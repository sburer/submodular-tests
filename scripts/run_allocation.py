"""Distributionally robust stratified sampling allocation (Section 5.3, Table 4).

Deterministic: mu, Sigma, the costs and the budget are the fixed data of
Section 5.3, and the allocation search is a complete enumeration.

For a fixed allocation m the inner problem

    sup_P  E_P[ max_i  xi_i (1 - xi_i) / m_i ]

is a moment problem of the form studied in Section 3.  Its dual is

    min  y0 + mu'y + Sigma.Y
    s.t. y0 + y'xi + xi'Y xi >= f_k(xi)   for all xi in [0,1]^n, all k
         Y_ij <= 0 (i != j)   for the ambiguity set P   [E[xi_i xi_j] >= Sigma_ij]
         Y psd                for the ambiguity set Q   [E[xi xi'] <= Sigma]

Each box-nonnegativity condition is imposed through the dual of relaxation (2),
which is exact here because n = 3 (Theorem 1) and each Q_k is submodular.

The outer minimization enumerates all 406 integer allocations with m_i >= 1 and
m_1 + m_2 + m_3 = 30.

Requires Mosek.

Run:  python run_allocation.py            (full enumeration, 812 SDPs)
      python run_allocation.py --table    (Table 4 only, 4 SDPs)
"""

import argparse
import numpy as np
import mosek.fusion as mf
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix

MU = np.array([0.5, 0.5, 0.5])
SIGMA = np.array([[0.49, 0.25, 0.25],
                  [0.25, 0.29, 0.25],
                  [0.25, 0.25, 0.26]])
COST = np.array([1.0, 1.0, 1.0])
BUDGET = 30
N = 3


def worstcase(m, ambiguity):
    """sup_P E_P[max_i xi_i(1-xi_i)/m_i] for ambiguity set 'P' or 'Q'."""
    m = np.asarray(m, dtype=float)
    with Model() as M:
        y0 = M.variable("y0", 1, Domain.unbounded())
        y = M.variable("y", N, Domain.unbounded())
        if ambiguity == "P":
            Y = M.variable("Y", Domain.unbounded([N, N]))
            M.constraint(Expr.sub(Y, Y.transpose()),
                         Domain.equalsTo(0.0).withShape([N, N]))
            for i in range(N):
                for j in range(N):
                    if i != j:
                        M.constraint(Y.index(i, j), Domain.lessThan(0.0))
        elif ambiguity == "Q":
            Y = M.variable("Y", Domain.inPSDCone(N))
        else:
            raise ValueError(ambiguity)

        for k in range(N):
            ek = np.zeros(N); ek[k] = 1.0
            Qk = Expr.add(Y, Matrix.dense(np.outer(ek, ek) / m[k]))
            ck = Expr.sub(y, ek / m[k])
            U = M.variable(f"U{k}", Domain.greaterThan(0.0, [N, N]))
            Ue = Expr.mul(U, np.ones(N))
            top = Expr.reshape(Expr.mul(0.5, Expr.sub(ck, Ue)), [1, N])
            blk = Expr.add(Qk, Expr.mul(0.5, Expr.add(U, U.transpose())))
            S = Expr.vstack(Expr.hstack(Expr.reshape(y0, [1, 1]), top),
                            Expr.hstack(Expr.reshape(top, [N, 1]), blk))
            M.constraint(S, Domain.inPSDCone(N + 1))

        M.objective(ObjectiveSense.Minimize,
                    Expr.add(Expr.add(y0.index(0), Expr.dot(MU, y)),
                             Expr.dot(Matrix.dense(SIGMA), Y)))
        M.solve()
        return M.primalObjValue()


def allocations(budget=BUDGET, n=N):
    return [(i, j, budget - i - j)
            for i in range(1, budget - 1)
            for j in range(1, budget - i)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true",
                    help="skip the enumeration; report Table 4 only")
    args = ap.parse_args()

    print("=" * 72)
    print("Section 5.3 -- distributionally robust stratified sampling allocation")
    print("=" * 72)
    print("\n[1] the information each ambiguity set pins down")
    print(f"    E[xi_i(1-xi_i)] = mu_i - Sigma_ii = {MU - np.diag(SIGMA)}")
    print(f"    Var[xi_i]       = Sigma_ii - mu_i^2 = {np.diag(SIGMA) - MU**2}")
    D = SIGMA - 0.25 * np.ones((N, N))
    print(f"    Sigma - 0.25 ee' eigenvalues      = {np.round(np.linalg.eigvalsh(D), 4)}")
    print(f"    => the point mass at (0.5,0.5,0.5) is Q-feasible: "
          f"{(np.linalg.eigvalsh(D) >= -1e-12).all()}")

    allocs = allocations()
    if args.table:
        best = {"P": (4, 13, 13), "Q": (10, 10, 10)}
        print(f"\n[2] enumeration skipped (--table); using the reported optima")
    else:
        print(f"\n[2] enumerating all {len(allocs)} integer allocations "
              f"with m_i >= 1 and sum m_i = {BUDGET}")
        best = {}
        for amb in ("P", "Q"):
            vals = {m: worstcase(m, amb) for m in allocs}
            mstar = min(vals, key=vals.get)
            ties = [m for m, v in vals.items() if abs(v - vals[mstar]) < 1e-8]
            best[amb] = mstar
            print(f"    ambiguity {amb}: optimum {mstar} with value {vals[mstar]:.6f}"
                  f"   (unique: {len(ties) == 1})")

    mP, mQ = best["P"], best["Q"]
    print("\n[3] Table 4 -- worst-case objective at each allocation")
    print(f"    {'':<14}{'m*_P = ' + str(mP):>20}{'m*_Q = ' + str(mQ):>20}")
    for amb in ("P", "Q"):
        vP = worstcase(mP, amb); vQ = worstcase(mQ, amb)
        star = lambda v, opt: f"{v:.4f}" + ("*" if opt else " ")
        print(f"    {amb:<14}{star(vP, amb=='P'):>20}{star(vQ, amb=='Q'):>20}")
    print("    (* marks the optimum for that ambiguity set)")
    print("\n[4] the tie at m*_Q: max_i xi_i(1-xi_i) <= 1/4 for any distribution")
    print(f"    on [0,1]^n, so 1/(4*{mQ[0]}) = {0.25/mQ[0]:.4f} is an upper bound at the")
    print( "    equal allocation for any ambiguity set, and both P and Q attain it.")


if __name__ == "__main__":
    main()
