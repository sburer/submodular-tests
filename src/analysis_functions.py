"""Extended construction/solution routines used by the structural and sparsity
studies reported in the paper's numerical section.

These provide the shared instance-generation and QP/SDP routines, and also
  - return the primal SDP solution matrix Y (needed for rank / active-set
    analysis), and
  - allow imposing only a selected subset of the RLT upper-bound constraints
    (needed for the sparsity study).

Function summary:
    generate_instance(n, seed, density=1.0)
        Random submodular (Q,c). If density<1.0, each off-diagonal pair of Q is
        zeroed independently with probability 1-density (sparse Q). Returns
        (Qc, Q, c, mask) where mask marks the retained off-diagonal pairs.

    solve_qp(n, Q, c)
        Global QP optimum over the box via Gurobi. Returns (x_opt, opt_value).

    solve_sdp(n, Qc, keep_pairs=None)
        SDP relaxation via Mosek. If keep_pairs (an (n,n) boolean array) is
        given, only the RLT bound X_ij <= x_j with keep_pairs[i,j] True is
        imposed (the diagonal is always imposed). Returns (dual_value, Y).

    numerical_rank(Y), active_pattern(Y, n), analyze(n, seed, density=1.0)
        Rank of Y, active-inequality decomposition (l, d, u), and a combined
        per-instance record, respectively.

Tolerances: solver tolerance is taken from define_constants (tol_mosek); the
rank/active/gap thresholds below are specific to the structural analysis.
"""
import os
import sys
import numpy as np
import numpy.random as npr

# Make sibling modules in src/ importable regardless of caller location.
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mosek.fusion import Model, Domain, Expr, ObjectiveSense, SolutionType
import gurobipy as gp
from gurobipy import GRB

from define_constants import tol_mosek

# Analysis-specific thresholds
RANK_TOL = 1.0e-6   # eigenvalue-ratio threshold for numerical rank of Y
ACT_TOL = 1.0e-6    # threshold for declaring an RLT inequality active
GAP_TOL = 1.0e-4    # relative gap above which an instance is a possible counterexample


def generate_instance(n, seed, density=1.0):
    """Random submodular (Q,c); density<1.0 yields a sparse Q."""
    if seed != -99:
        npr.seed(seed)
    Q = npr.standard_normal((n, n))
    Q = 0.5 * (Q + Q.T)
    mask = np.ones((n, n))
    for i in range(n):
        for j in range(i):
            keep = 1.0 if density >= 1.0 else (1.0 if npr.rand() < density else 0.0)
            val = -np.abs(Q[i, j]) * keep
            Q[i, j] = val
            Q[j, i] = val
            mask[i, j] = keep
            mask[j, i] = keep
    c = npr.standard_normal((n, 1))
    Qc = np.block([[0.0, c.T], [c, Q]])
    return Qc, Q, c, mask


def solve_qp(n, Q, c):
    """Global optimum of min x'Qx + 2c'x over [0,1]^n via Gurobi."""
    m = gp.Model("qp")
    m.setParam("OutputFlag", 0)
    m.setParam("OptimalityTol", 1e-8)
    m.setParam("FeasibilityTol", 1e-9)
    x = m.addMVar(n, lb=0.0, ub=1.0)
    m.setObjective(x @ Q @ x + 2.0 * c.flatten() @ x, GRB.MINIMIZE)
    m.optimize()
    if m.status != GRB.OPTIMAL:
        return None, None
    return np.reshape(x.X, (n, 1)), m.ObjVal


def solve_sdp(n, Qc, keep_pairs=None):
    """SDP relaxation of the box QP. Returns (dual_value, Y). If keep_pairs is
    given, only the selected RLT bounds X_ij <= x_j are imposed."""
    M = Model("sdp")
    Y = M.variable("Y", Domain.inPSDCone(n + 1))
    Y00 = Y.slice([0, 0], [1, 1])
    x = Y.slice([1, 0], [n + 1, 1])
    X = Y.slice([1, 1], [n + 1, n + 1])
    M.objective(ObjectiveSense.Minimize, Expr.dot(Qc, Y))
    M.constraint(Y00, Domain.equalsTo(1.0))
    if keep_pairs is None:
        expr = Expr.sub(Expr.mul(np.ones((n, 1)), Expr.transpose(x)), X)
        M.constraint(expr, Domain.greaterThan(0.0))
    else:
        for i in range(n):
            for j in range(n):
                if i == j or keep_pairs[i, j]:
                    M.constraint(Expr.sub(x.index(j, 0), X.index(i, j)),
                                 Domain.greaterThan(0.0))
    M.setSolverParam("log", 0)
    M.setSolverParam("intpntCoTolPfeas", tol_mosek)
    M.setSolverParam("intpntCoTolDfeas", tol_mosek)
    M.setSolverParam("intpntCoTolRelGap", tol_mosek)
    M.solve()
    if M.getProblemStatus(SolutionType.Default).name != "PrimalAndDualFeasible":
        M.dispose()
        return None, None
    dval = M.dualObjValue()
    Ymat = np.reshape(Y.level(), (n + 1, n + 1))
    M.dispose()
    return dval, Ymat


def numerical_rank(Ymat):
    """Numerical rank of Y via an eigenvalue-ratio threshold."""
    w = np.clip(np.linalg.eigvalsh(0.5 * (Ymat + Ymat.T)), 0, None)
    if w.max() <= 0:
        return 0
    return int(np.sum(w > RANK_TOL * w.max()))


def active_pattern(Ymat, n):
    """Active-inequality decomposition (l, d, u, a) after sorting by x.

    After sorting x ascending, for i<j the two RLT bounds on X_ij are
    X_ij <= x_i (s-upper, the smaller) and X_ij <= x_j (s-lower, the larger);
    diagonal bounds are X_ii <= x_i. Returns counts (l, d, u) of active
    s-lower, diagonal, and s-upper inequalities, and their sum a.
    """
    x = Ymat[1:, 0].copy()
    X = Ymat[1:, 1:].copy()
    order = np.argsort(x)
    x = x[order]
    X = X[np.ix_(order, order)]
    d = sum(abs(X[i, i] - x[i]) <= ACT_TOL for i in range(n))
    l = u = 0
    for i in range(n):
        for j in range(i + 1, n):  # x[i] <= x[j]
            if abs(X[i, j] - x[i]) <= ACT_TOL:  # bound by smaller -> s-upper
                u += 1
            if abs(X[i, j] - x[j]) <= ACT_TOL:  # bound by larger -> s-lower
                l += 1
    return l, d, u, l + d + u


def analyze(n, seed, density=1.0):
    """Solve one instance and return a record with the relative gap, rank of Y,
    and active-inequality decomposition. Returns None on solver failure."""
    Qc, Q, c, mask = generate_instance(n, seed, density)
    x_opt, pval = solve_qp(n, Q, c)
    if x_opt is None:
        return None
    dval, Ymat = solve_sdp(n, Qc)
    if dval is None:
        return None
    denom = max(1.0, abs(pval + dval) / 2.0)
    rel_gap = (pval - dval) / denom
    r = numerical_rank(Ymat)
    l, d, u, a = active_pattern(Ymat, n)
    return dict(seed=seed, n=n, rel_gap=rel_gap, r=r, l=l, d=d, u=u, a=a)
