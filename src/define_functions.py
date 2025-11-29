"""Core construction and solution routines for submodular-like QP box SDP.

Function summary:
    generate_random_instance(n, opts=None, seed=-99)
        Returns (n+1)x(n+1) block matrix Qc = [[0, c'; c, Q]] where
        Q is symmetric (optionally forced submodular: off-diagonals <= 0) and
        c follows option "c" (zero | nonpositive | free). Options dict keys:
            q:          structure of Q ("submodular" | "free")
            c:          distribution for c ("zero" | "nonpositive" | "free")
            q_density:  fraction of nonzero off-diagonal entries in Q (0.0–1.0,
                        default 1.0 keeps full density)
            qc_round:   round Q and c entries to 1/k grid (None or int k>=2, default None)
            psd:        intended later PSD enforcement style (informational here)
            rlt:        intended later RLT family (informational here)

    generate_hopefully_nontight_instance(n, x, X)
        Given a feasible (x, X) for the SDP relaxation, solves an auxiliary
        Fusion model to construct Q, c so that (x, X) is optimal using only
        PSD + basic RLT upper bounds, aiming for non-rank-1 X. Returns Qc.
        If Mosek status not PrimalAndDualFeasible, returns -inf matrix.

    build_and_solve_sdp(n=None, Qc=None, opts=None, fixed_values=None, x_opt=None)
        Builds and solves selected SDP relaxation variant over the box using
        Fusion. PSD variants: full cone, all 2x2, or leading 3x3 minors.
        RLT variants: none | diag | upper_bounds | full. Can fix entries of
        Y via fixed_values array [[i, j, value], ...]. If `x_opt` (QP optimum)
        is provided, uses it to compute `pval_xopt` and the relative gap; else
        falls back to the SDP primal from the first column of Y. Returns tuple:
            (rel_gap, eval_ratio, YY, ZZ, SS, fixed_values, opts, pval_sdp, pval_xopt)
        where:
            rel_gap:  primal/dual relative gap (nonnegative near convergence)
            eval_ratio: leading eigenvalue / second eigenvalue of Y for rank hint
            YY:      final (n+1)x(n+1) Y matrix
            ZZ:      duals of upper-bound constraints if used else []
            SS:      duals of PSD cone (Y) else []
            fixed_values: augmented with dual values if provided
            opts:    options dict passed through
            pval_sdp: objective from SDP primal (first column of Y)
            pval_xopt: objective from QP optimum if available, else None

All tolerances imported from define_constants; adjust there if needed.
"""
###############################################################################

# Import packages

import os
import sys
import math
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import pandas as pd
import mosek
from mosek.fusion import *
import gurobipy as gp
from gurobipy import GRB

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "% .10f" % x))

# Import copy

import copy

###############################################################################

# Set constants

from define_constants import *

###############################################################################

def generate_random_instance(n, opts = None, seed = -99):

    # If user has entered seed (not equal to -99), then set the seed

    if seed != -99:
        npr.seed(seed)

    if opts is None:

        opts = {
            "q":          "submodular",   # "submodular" or "free"
            "c":          "free",         # "zero" or "nonpositive" or "free"
            "q_density": 1.0,             # density of off-diagonal entries
            "qc_round": None              # round to 1/k (None=no rounding, or int k>=2)
        }

    else:
        # Ensure required keys present; do not add unrelated options here
        if "q" not in opts:
            opts["q"] = "submodular"
        if "c" not in opts:
            opts["c"] = "free"
        if "q_density" not in opts:
            opts["q_density"] = 1.0
        if "qc_round" not in opts:
            opts["qc_round"] = None
    # Basic validation / clipping for density
    try:
        density = float(opts["q_density"])
    except Exception:
        density = 1.0
    density = max(0.0, min(1.0, density))
    opts["q_density"] = density

    # Generate Q and c

    Q = npr.standard_normal((n, n))
    Q = 0.5 * (Q + Q.T)
    
    # Optional rounding to 1/k grid
    if opts["qc_round"] is not None:
        try:
            k = int(opts["qc_round"])
            if k >= 2:
                Q = np.round(Q * k) / k
        except (TypeError, ValueError):
            pass  # skip rounding if invalid
    
    if opts["q"] == "submodular":
        for i in range(0, n):
            for j in range(0, i):
                Q[i, j] = -np.abs(Q[i, j])
                Q[j, i] = Q[i, j]

    # Enforce approximate off-diagonal density via symmetric mask
    if density < 1.0:
        if n > 1:
            upper_rand = npr.uniform(size=(n, n)) < density
            upper_mask = np.triu(upper_rand, 1)
            mask = upper_mask + upper_mask.T + np.eye(n)
        else:
            mask = np.eye(n)
        Q = Q * mask

    if opts["c"] == "zero":
        c = np.zeros((n, 1))
    elif opts["c"] == "nonpositive":
        c = -np.abs(npr.standard_normal((n, 1)))
    else:
        c = npr.standard_normal((n, 1))
    
    # Apply rounding to c if specified
    if opts["qc_round"] is not None:
        try:
            k = int(opts["qc_round"])
            if k >= 2:
                c = np.round(c * k) / k
        except (TypeError, ValueError):
            pass
    
    # Normalize consistently: scale Q and c so that ||Qc|| = 1
    # Qc_tmp = np.block([[0, c.T], [c, Q]])
    # scale = npl.norm(Qc_tmp)
    # if scale > 0:
    #     Q = Q / scale
    #     c = c / scale
    Qc = np.block([[0, c.T], [c, Q]])

    # Solve the QP to optimality with Gurobi: min x'*Q*x + 2*c'*x s.t. 0 <= x <= 1

    model = gp.Model("QP_over_box")
    model.setParam('OutputFlag', 0)  # Suppress output
    model.setParam('OptimalityTol', 1e-8)  # Tighten optimality tolerance
    model.setParam('FeasibilityTol', 1e-9)  # Tighten feasibility tolerance
    # model.setParam('NonConvex', 2)
    
    # Create variables
    x_vars = model.addMVar(n, lb=0.0, ub=1.0, name="x")
    
    # Set objective: (1/2)*x'*Q*x + c'*x
    model.setObjective(x_vars @ Q @ x_vars + 2.0 * c.flatten() @ x_vars, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        print(f"Warning: Gurobi did not find optimal solution, status = {model.status}")
        x_opt = np.zeros((n, 1))
    else:
        x_opt = np.reshape(x_vars.X, (n, 1))

    return Qc, opts, x_opt

###############################################################################

def generate_hopefully_nontight_instance(n, x, X):

    # Generate Q,c such that (x,X) is optimal for SDP with only PSD
    # and RLT upper bounds. Note that (x,X) must be feasible on input.
    # Intention is that X is not rank-1

    # Follows paper by Emre Alper Yildirim and co-author

    M = Model('Optimality feasibility program')
    
    Y = M.variable('Y', Domain.greaterThan(0.0, n, n))
    S = M.variable('S', Domain.inPSDCone(n + 1))

    beta = S.slice([0, 0], [1, 1]) # Could also be S.index([0, 0])
    h    = S.slice([1, 0], [n + 1, 1])
    H    = S.slice([1, 1], [n + 1, n + 1])

    e = np.ones((n, 1))
    mat = x @ e.T - X
    M.constraint(Expr.dot(mat, Y), Domain.equalsTo(0.0))

    myexpr = Expr.add(beta, Expr.mul(2.0, Expr.dot(x, h)))
    myexpr = Expr.add(myexpr, Expr.dot(X, H))
    M.constraint(myexpr, Domain.equalsTo(0.0))

    myobj = 0
    myexpr = Expr.sub(H, Y)
    myexpr = Expr.sub(myexpr, Expr.transpose(Y))
    for i in range(0, n):
        for j in range(0, i):
            M.constraint(myexpr.index([i, j]), Domain.lessThan(0.0))
            myobj = Expr.add(myobj, myexpr.index([i, j]))

    M.objective(ObjectiveSense.Minimize, myobj)

    myexpr = Expr.sum(Y)
    for i in range(0, n + 1):
        myexpr = Expr.add(myexpr, S.index([i, i]))
    M.constraint(myexpr, Domain.equalsTo(1.0))

    # Solve

    M.solve()
    
    # Get return code. If we don't get 'PrimalAndDualFeasible', we balk

    return_code = M.getProblemStatus(SolutionType.Default).name

    if(return_code != 'PrimalAndDualFeasible'):
        print(return_code)
        print("Got unexpected return_code. Exiting...")
        return -np.inf * np.ones((n + 1, n + 1))

    # Get solution values

    YY = np.reshape(Y.level(), (n, n))
    HH = np.reshape(H.level(), (n, n))
    hh = np.reshape(h.level(), (n, 1))

    #  print(YY)
    #  print(HH)

    Q = 1 * (HH - YY - YY.T) # Factor 1 is by trial and error
    #  print(Q)
    evals, evecs = npl.eig(Q)
    #  print(evals)
    c = 1 * (hh + YY.T @ e) # Factor 1 is by trial and error
    #  print(c)

    Qc = np.block([[0, c.T], [c, Q]])

    return Qc

###############################################################################

# Define function to solve SDP relaxation of QPB ("quadratic programming
# over the box")

def build_and_solve_sdp(n = None, Qc = None, opts = None, fixed_values = None, x_opt = None):

    # Set default options

    if opts is None:

        opts = {
            "psd":     "full",         # "all_2x2" or "leading_3x3" or "full"
            "rlt":     "upper_bounds"  # "none" or "diag" or "upper_bounds" or "full"
        }

    else:
        # Ensure required keys present; do not add unrelated options here
        if "psd" not in opts:
            opts["psd"] = "full"
        if "rlt" not in opts:
            opts["rlt"] = "upper_bounds"

    # Setup basic model

    M = Model('SDP relaxation of QPB')

    if opts["psd"] == "all_2x2":
        # The following 8 lines of code setup Y but only enforce the PSD
        # condition on the 2x2 principal submatrices. I did this as a test,
        # but it led to gaps. I.e., it seems we need stronger conditions
        # than just the 2x2 submatrices
        Y = M.variable('Y', Domain.unbounded(n + 1, n + 1))
        M.constraint(Expr.sub(Y, Expr.transpose(Y)), Domain.equalsTo(0.0))
        for i in range(0, n + 1):
            for j in range(0, i):
                myexpr_row1 = Expr.hstack([Y.index([i, i]), Y.index([i, j])])
                myexpr_row2 = Expr.hstack([Y.index([j, i]), Y.index([j, j])])
                myexpr = Expr.vstack([myexpr_row1, myexpr_row2])
                M.constraint(myexpr, Domain.inPSDCone(2))

    elif opts["psd"] == "leading_3x3":
        Y = M.variable('Y', Domain.unbounded(n + 1, n + 1))
        M.constraint(Expr.sub(Y, Expr.transpose(Y)), Domain.equalsTo(0.0))
        for i in range(0, n + 1):
            for j in range(0, i):
                myexpr_row1 = Expr.hstack([Y.index([0, 0]), Y.index([0, i]), Y.index([0, j])])
                myexpr_row2 = Expr.hstack([Y.index([0, i]), Y.index([i, i]), Y.index([i, j])])
                myexpr_row3 = Expr.hstack([Y.index([0, j]), Y.index([i, j]), Y.index([j, j])])
                myexpr = Expr.vstack([myexpr_row1, myexpr_row2, myexpr_row3])
                M.constraint(myexpr, Domain.inPSDCone(3))
    else:
        Y = M.variable('Y', Domain.inPSDCone(n + 1))

    # Define Y = [Y00, x'; x, X] as usual
    Y00 = Y.slice([0, 0], [1, 1]) # Could also be Y.index([0, 0])
    x   = Y.slice([1, 0], [n + 1, 1])
    X   = Y.slice([1, 1], [n + 1, n + 1])
    
    # Setup objective

    M.objective(ObjectiveSense.Minimize, Expr.dot(Qc, Y))

    # Add all the constraints

    # Y00 = 1
    M.constraint(Y00, Domain.equalsTo(1.0))

    if opts["rlt"] == "full":

        for i in range(0, n):

            # Xii <= xi
            myexpr = Expr.sub(x.index([i, 0]), X.index([i, i]))
            M.constraint(myexpr, Domain.greaterThan(0.0))

            for j in range(0, i):

                # Xij >= 0
                myexpr = X.index([i, j])
                M.constraint(myexpr, Domain.greaterThan(0.0))

                # Xij >= xi + xj - 1
                myexpr = Expr.sub(X.index([i, j]), x.index([i, 0]))
                myexpr = Expr.sub(myexpr, x.index([j, 0]))
                myexpr = Expr.add(myexpr, 1)
                M.constraint(myexpr, Domain.greaterThan(0.0))

                # Xij <= xi
                myexpr = Expr.sub(x.index([i, 0]), X.index([i, j]))
                M.constraint(myexpr, Domain.greaterThan(0.0))

                # Xij <= xj
                myexpr = Expr.sub(x.index([j, 0]), X.index([i, j]))
                M.constraint(myexpr, Domain.greaterThan(0.0))

    elif opts["rlt"] == "upper_bounds":

        myexpr = Expr.mul(np.ones((n, 1)), Expr.transpose(x))
        myexpr = Expr.sub(myexpr, X)
        cons_upper_bound = M.constraint(myexpr, Domain.greaterThan(0.0))

    elif opts["rlt"] == "diag":

        for i in range(0, n):

            # Xii <= xi
            myexpr = Expr.sub(x.index([i, 0]), X.index([i, i]))
            M.constraint(myexpr, Domain.greaterThan(0.0))

    else:

        for i in range(0, n):
            # 0 <= Xii <= 1
            M.constraint(X.index([i, i]), Domain.lessThan(1.0))
            M.constraint(X.index([i, i]), Domain.greaterThan(0.0))

    # Triangle inequalities for n = 3. Currently disabled

    if n == -3:

        i = 0
        j = 1
        k = 2

        myexpr = Expr.add(X.index([i, j]), X.index([i, k]))
        myexpr = Expr.add(myexpr, X.index([j, k]))
        myexpr = Expr.add(myexpr, 1)
        myexpr = Expr.sub(myexpr, x.index([i, 0]))
        myexpr = Expr.sub(myexpr, x.index([j, 0]))
        myexpr = Expr.sub(myexpr, x.index([k, 0]))
        M.constraint(myexpr, Domain.greaterThan(0.0))

        myexpr = Expr.add(x.index([i, 0]), X.index([j, k]))
        myexpr = Expr.sub(myexpr, X.index([i, j]))
        myexpr = Expr.sub(myexpr, X.index([i, k]))
        M.constraint(myexpr, Domain.greaterThan(0.0))

        i = 1
        j = 0 
        k = 2

        myexpr = Expr.add(x.index([i, 0]), X.index([j, k]))
        myexpr = Expr.sub(myexpr, X.index([i, j]))
        myexpr = Expr.sub(myexpr, X.index([i, k]))
        M.constraint(myexpr, Domain.greaterThan(0.0))

        i = 2
        j = 0
        k = 1

        myexpr = Expr.add(x.index([i, 0]), X.index([j, k]))
        myexpr = Expr.sub(myexpr, X.index([i, j]))
        myexpr = Expr.sub(myexpr, X.index([i, k]))
        M.constraint(myexpr, Domain.greaterThan(0.0))

    # Fix specified values of Y matrix

    if fixed_values is not None:

        mylen = fixed_values.shape[0]
        fix_constraints = dict(zip(range(0, mylen), mylen * [None]))

        for k in range(0, mylen):
            i = int(fixed_values[k, 0])
            j = int(fixed_values[k, 1])
            val = fixed_values[k, 2]
            #  print(str(i) + ' ' + str(j) + ' ' + str(val))
            fix_constraints[k] = M.constraint(Y.index(i, j), Domain.equalsTo(val))

    # Setup Mosek options. Do I want to add tighter tolerances?

    M.setLogHandler(sys.stdout)
    M.setSolverParam("log", 0)
    M.setSolverParam("intpntCoTolPfeas", tol_mosek)
    M.setSolverParam("intpntCoTolDfeas", tol_mosek)
    M.setSolverParam("intpntCoTolRelGap", tol_mosek)

    # Solve

    M.solve()
    
    # Get return code. If we don't get 'PrimalAndDualFeasible', we balk

    return_code = M.getProblemStatus(SolutionType.Default).name

    if(return_code != 'PrimalAndDualFeasible'):
        print(return_code)
        print("Got unexpected return_code. Exiting...")
        return -np.inf, np.inf

    # Get solution values

    YY = np.reshape(Y.level(), (n + 1, n + 1))
    # Extract solution approximation from first column of Y
    yy = YY[:, [0]]

    diagXX = np.diagonal(YY)
    diagXX = diagXX[1 : (n + 1)]
    #  tmp = np.logical_and(diagXX >= 0.05, diagXX <= 0.95)
    #  if np.any(tmp):
    #      Q = Qc[1 : (n + 1), 1 : (n + 1)]
    #      tmp2 = Q[np.ix_(tmp, tmp)]
    #      evals, evecs = npl.eig(tmp2)
    #      if np.sort(evals)[0] < tol_general:
    #          print('Got unexpected (something about evals of submatrix of Q). Ignoring as of 2025-05-14')

    # Get primal and dual values

    # SDP primal via first column of Y
    pval_sdp = (yy.T @ Qc @ yy)[0, 0]

    # QP primal via x_opt (if provided)
    if x_opt is not None:
        x_opt_ext = np.vstack([[1.0], x_opt])  # Prepend 1 for augmented form
        pval_xopt = (x_opt_ext.T @ Qc @ x_opt_ext)[0, 0]
        pval = pval_xopt
    else:
        pval_xopt = None
        pval = pval_sdp
    dval = M.dualObjValue()
    #  print(pval)
    #  print(dval)

    #  RLT_dual = np.reshape(RLT.dual(), (n, n))
    #  tmp = 0.5 * RLT_dual @ np.ones((n, 1))
    #  RLT_dual = 0.5 * (RLT_dual + RLT_dual.T)
    #  R1 = np.hstack([np.zeros((1, 1)), tmp.T])
    #  R2 = np.hstack([tmp, -RLT_dual])
    #  R = np.vstack([R1, R2])

    #  S = np.reshape(Y.dual(), (n + 1, n + 1))
    #  if submodular_instance == 0:
    #      evals, evecs = npl.eig(S)
    #      print(np.sort(evals))
    #  print(Qc[1:(n+1), 1:(n+1)] - S[1:(n+1), 1:(n+1)] + 0.5*(RLT_dual + RLT_dual.T))
    #  print(S[1:(n+1), 1:(n+1)])

    #  mat = Qc - S - R
    #  print(Qc)
    #  print(S)
    #  print(R)

    # Calculate relative gap as usual. Except that if pval and dval
    # are really close, we purposely do not allow negative values

    if np.abs(pval - dval) < tol_general:
        rel_gap = abs(pval - dval) / np.maximum(1.0, np.abs(pval + dval) / 2.0)
    else:
        rel_gap = (pval - dval) / np.maximum(1.0, np.abs(pval + dval) / 2.0)

    # Calculate eigenvalue ratio as usual. Could put this in its own
    # function

    evals, evecs = npl.eig(YY)
    evals = -np.sort(-evals)
    eval_ratio = evals[0] / np.abs(evals[1])

    if opts["rlt"] == "upper_bounds":
        ZZ = np.reshape(cons_upper_bound.dual(), (n, n))
        SS = np.reshape(Y.dual(), (n + 1, n + 1))
    else:
        ZZ = []
        SS = []

    if fixed_values is not None:

        mylen = fixed_values.shape[0]
        fixed_values = np.hstack([fixed_values, -np.inf * np.ones((mylen, 1))])

        for k in range(0, mylen):
            fixed_values[k, 3] = fix_constraints[k].dual()[0]

    M.dispose()

    #  print(YY)
    #  print("")

    return rel_gap, eval_ratio, YY, ZZ, SS, fixed_values, opts, pval_sdp, pval_xopt
