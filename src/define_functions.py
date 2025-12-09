"""Core construction and solution routines for submodular-like QP box SDP.

Function summary:
    generate_random_instance(n, seed=-99)
        Returns (n+1)x(n+1) block matrix Qc = [[0, c'; c, Q]] where
        Q is symmetric with submodular structure (off-diagonals <= 0) and
        c is a random vector.

    build_and_solve_sdp(n=None, Qc=None, x_opt=None)
        Builds and solves the SDP relaxation with full PSD cone and upper_bounds
        RLT constraints (Xij <= xi for all i,j). x_opt is the QP optimum and is
        used to compute the relative gap. Returns tuple:
            (rel_gap,)
        where:
            rel_gap: primal/dual relative gap (nonnegative near convergence)

All tolerances imported from define_constants; adjust there if needed.
"""
###############################################################################

# Import packages

import sys
import numpy as np
import numpy.random as npr
from mosek.fusion import *
import gurobipy as gp
from gurobipy import GRB

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "% .10f" % x))

###############################################################################

# Set constants

from define_constants import *

###############################################################################

def generate_random_instance(n, seed = -99):

    # If user has entered seed (not equal to -99), then set the seed

    if seed != -99:
        npr.seed(seed)

    # Generate Q and c

    Q = npr.standard_normal((n, n))
    Q = 0.5 * (Q + Q.T)
    
    # Enforce submodular structure: off-diagonals are non-positive
    for i in range(0, n):
        for j in range(0, i):
            Q[i, j] = -np.abs(Q[i, j])
            Q[j, i] = Q[i, j]

    # Generate c as a random vector
    c = npr.standard_normal((n, 1))
    
    Qc = np.block([[0, c.T], [c, Q]])

    # Solve the QP to optimality with Gurobi: min x'*Q*x + 2*c'*x s.t. 0 <= x <= 1

    model = gp.Model("QP_over_box")
    model.setParam('OutputFlag', 0)  # Suppress output
    model.setParam('OptimalityTol', 1e-8)  # Tighten optimality tolerance
    model.setParam('FeasibilityTol', 1e-9)  # Tighten feasibility tolerance
    
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

    return Qc, x_opt

###############################################################################

# Define function to solve SDP relaxation of QPB ("quadratic programming
# over the box")

def build_and_solve_sdp(n = None, Qc = None, x_opt = None):

    # Setup basic model

    M = Model('SDP relaxation of QPB')

    # Y is constrained to the PSD cone
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

    # RLT upper bound constraints: Xij <= xi for all i,j
    myexpr = Expr.mul(np.ones((n, 1)), Expr.transpose(x))
    myexpr = Expr.sub(myexpr, X)
    M.constraint(myexpr, Domain.greaterThan(0.0))

    # Configure Mosek solver tolerances
    M.setLogHandler(sys.stdout)
    M.setSolverParam("log", 0)
    M.setSolverParam("intpntCoTolPfeas", tol_mosek)
    M.setSolverParam("intpntCoTolDfeas", tol_mosek)
    M.setSolverParam("intpntCoTolRelGap", tol_mosek)

    # Solve

    M.solve()
    
    # Verify solver convergence
    return_code = M.getProblemStatus(SolutionType.Default).name
    if(return_code != 'PrimalAndDualFeasible'):
        print(return_code)
        print("Got unexpected return_code. Exiting...")
        return -np.inf, np.inf

    # Get solution values
    # (not using SDP solution approximation for gap calculation)

    # Get primal and dual values

    # Compute objective from QP optimum (x_opt)
    x_opt_ext = np.vstack([[1.0], x_opt])  # Prepend 1 for augmented form
    pval = (x_opt_ext.T @ Qc @ x_opt_ext)[0, 0]
    dval = M.dualObjValue()

    # Compute relative gap (prevents negative gaps when values are very close)
    if np.abs(pval - dval) < tol_general:
        rel_gap = abs(pval - dval) / np.maximum(1.0, np.abs(pval + dval) / 2.0)
    else:
        rel_gap = (pval - dval) / np.maximum(1.0, np.abs(pval + dval) / 2.0)

    M.dispose()

    return (rel_gap,)
