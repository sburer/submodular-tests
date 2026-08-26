"""Three-product pricing example (Section 5.2, Tables 2 and 3).

Deterministic: the instance is the fixed data of Table 2, no randomness.

A mobile data provider sells three prepaid bundles under the linear demand
model d(p) = a - Bp and maximizes profit (p - m)'(a - Bp) over a price box.
B is not symmetric and (B + B')/2 is indefinite, so this is not a convex QP;
but the off-diagonals of (B + B')/2 are nonpositive, so it is QSMB, and since
n = 3 Theorem 1 makes the SDP relaxation exact.

Requires Mosek (SDP).  Gurobi is used only for an independent global check of
the QP optimum and is skipped if unavailable.

Run:  python run_pricing.py
"""

import itertools
import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix

# ---- Table 2 data ---------------------------------------------------------
A_VEC = np.array([208.0, 220.0, 342.0])
B_MAT = np.array([[25.0, -20.0, -1.0],
                  [-3.0, 4.0, -0.25],
                  [-1.0, -0.25, 8.0]])
MARGINAL_COST = np.array([3.0, 8.0, 14.0])
BASELINE_PRICE = np.array([8.0, 18.0, 32.0])
LO = np.array([6.0, 14.0, 24.0])
HI = np.array([10.0, 22.0, 36.0])
BUNDLES = ["10 GB", "50 GB", "100 GB"]
N = 3


def demand(p):
    return A_VEC - B_MAT @ p


def profit(p):
    return float((p - MARGINAL_COST) @ demand(p))


def solve_sdp():
    """Relaxation (2) for the QSMB form of the pricing problem."""
    S = (B_MAT + B_MAT.T) / 2
    g = A_VEC + B_MAT.T @ MARGINAL_COST
    D = np.diag(HI - LO)
    Q = D @ S @ D
    c = 2 * D @ S @ LO - D @ g
    const = LO @ S @ LO - g @ LO + MARGINAL_COST @ A_VEC

    with Model() as M:
        Y = M.variable("Y", Domain.inPSDCone(N + 1))
        x = Y.slice([0, 1], [1, N + 1]).reshape([N])
        X = Y.slice([1, 1], [N + 1, N + 1])
        M.constraint(Y.index(0, 0), Domain.equalsTo(1.0))
        for i in range(N):
            for j in range(N):
                M.constraint(Expr.sub(X.index(i, j), x.index(i)), Domain.lessThan(0.0))
        M.objective(ObjectiveSense.Minimize,
                    Expr.add(Expr.dot(Matrix.dense(Q), X), Expr.dot(c, x)))
        M.solve()
        val = M.primalObjValue()
        Yv = np.array(Y.level()).reshape(N + 1, N + 1)
    xv = Yv[0, 1:]
    return Q, c, const, val, LO + (HI - LO) * xv, Yv


def main():
    S = (B_MAT + B_MAT.T) / 2
    print("=" * 72)
    print("Section 5.2 -- pricing with n = 3 products")
    print("=" * 72)

    print("\n[1] the instance is QSMB but not a convex QP")
    print(f"    eigenvalues of (B+B')/2 : {np.round(np.linalg.eigvalsh(S), 3)}")
    print(f"    indefinite              : "
          f"{np.linalg.eigvalsh(S)[0] < 0 < np.linalg.eigvalsh(S)[-1]}")
    print(f"    off-diagonals <= 0      : "
          f"{all(S[i, j] <= 0 for i in range(N) for j in range(N) if i != j)}")

    worst = min(min(demand(np.array(p))) for p in itertools.product(*zip(LO, HI)))
    print(f"\n[2] demand is nonnegative over the price box (min over corners): {worst}")

    d0 = demand(BASELINE_PRICE); p0 = profit(BASELINE_PRICE)
    print("\n[3] Table 2 -- baseline")
    print(f"    {'Bundle':<9}{'Marg cost':>11}{'Price':>9}{'Demand':>10}{'Range':>14}")
    for i, b in enumerate(BUNDLES):
        print(f"    {b:<9}{MARGINAL_COST[i]:>11.0f}{BASELINE_PRICE[i]:>9.0f}"
              f"{d0[i]:>10.1f}{f'[{LO[i]:.0f},{HI[i]:.0f}]':>14}")
    print(f"    baseline profit: ${p0/1000:.3f} million per month")

    Q, c, const, val, pstar, Yv = solve_sdp()
    dstar = demand(pstar); pf = profit(pstar)
    ev = np.linalg.eigvalsh(Yv)
    rank = int((ev > 1e-7 * max(1.0, ev.max())).sum())

    print("\n[4] Table 3 -- optimum from the SDP relaxation (exact by Theorem 1)")
    print(f"    SDP objective (x-space) : {val:.6f}")
    print(f"    implied max profit      : {-(val + const):.4f} thousand")
    print(f"    rank of Y(x,X)          : {rank}  (rank one => certified tight)")
    print(f"    p*                      : {np.round(pstar, 4)}")
    print(f"    d(p*)                   : {np.round(dstar, 4)}")
    print(f"    {'Bundle':<9}{'Price':>8}{'Demand':>10}{'Margin':>9}{'Profit(M)':>12}")
    tot = 0.0
    for i, b in enumerate(BUNDLES):
        m = pstar[i] - MARGINAL_COST[i]; pr = m * dstar[i] / 1000.0; tot += pr
        print(f"    {b:<9}{pstar[i]:>8.0f}{dstar[i]:>10.1f}{m:>9.0f}{pr:>12.3f}")
    print(f"    {'total':<9}{'':>8}{'':>10}{'':>9}{tot:>12.3f}")
    print(f"    optimal profit: ${pf/1000:.3f} million per month, "
          f"an increase of {100*(pf/p0-1):.2f}% over the baseline")

    try:
        import gurobipy as gp
        from gurobipy import GRB
        g = A_VEC + B_MAT.T @ MARGINAL_COST
        mdl = gp.Model(); mdl.Params.OutputFlag = 0; mdl.Params.NonConvex = 2
        p = mdl.addVars(N, lb=LO, ub=HI)
        pv = [p[i] for i in range(N)]
        mdl.setObjective(
            gp.quicksum(pv[i] * S[i, j] * pv[j] for i in range(N) for j in range(N))
            - gp.quicksum(g[i] * pv[i] for i in range(N))
            + float(MARGINAL_COST @ A_VEC), GRB.MINIMIZE)
        mdl.optimize()
        print(f"\n[5] independent global check (Gurobi): max profit "
              f"{-mdl.ObjVal:.4f} at p = {[round(p[i].X, 4) for i in range(N)]}")
    except Exception as e:
        print(f"\n[5] Gurobi check skipped ({type(e).__name__})")


if __name__ == "__main__":
    main()
