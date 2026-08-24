"""Numerical half of the n = 4 counterexample verification (requires Mosek).

Provides run_numerics(), used by verify_counterexample.py:

    B1  optimal value of relaxation (2), the rank of its optimal solution, and
        its active-inequality signature (l, d, u)
    B2  Table 6 of the paper: the cut ladder
    B3  a rational certificate that a gap survives PSD + full RLT + triangle
"""

from fractions import Fraction as F
from itertools import combinations

import numpy as np
import mosek.fusion as mf
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix

from verify_counterexample import Q_INT, C_INT, N

Q = np.array(Q_INT, dtype=float)
C = np.array(C_INT, dtype=float)


def triangle_rows(n=N):
    """The 4 triangle inequalities per triple, as (G, g0) with G.X + g0'x <= 0."""
    rows = []
    for i, j, k in combinations(range(n), 3):
        for (a, b, c_) in [(i, j, k), (j, i, k), (k, i, j)]:
            G = np.zeros((n, n)); g0 = np.zeros(n)
            # X_ab + X_ac <= x_a + X_bc
            G[b, c_] -= 0.5; G[c_, b] -= 0.5
            G[a, b] += 0.5; G[b, a] += 0.5
            G[a, c_] += 0.5; G[c_, a] += 0.5
            g0[a] -= 1.0
            rows.append((G, g0, 0.0))
        # x_i + x_j + x_k <= X_ij + X_ik + X_jk + 1
        G = np.zeros((n, n)); g0 = np.zeros(n)
        for (a, b) in [(i, j), (i, k), (j, k)]:
            G[a, b] -= 0.5; G[b, a] -= 0.5
        g0[i] = g0[j] = g0[k] = 1.0
        rows.append((G, g0, -1.0))
    return rows


def solve_relaxation(rlt="upper", triangle=False):
    """Optimal value of the SDP relaxation with the requested strengthenings."""
    with Model() as M:
        Y = M.variable("Y", Domain.inPSDCone(N + 1))
        x = Y.slice([0, 1], [1, N + 1]).reshape([N])
        X = Y.slice([1, 1], [N + 1, N + 1])
        M.constraint(Y.index(0, 0), Domain.equalsTo(1.0))
        for i in range(N):
            for j in range(N):
                M.constraint(Expr.sub(X.index(i, j), x.index(i)),
                             Domain.lessThan(0.0))          # X_ij <= x_i
                if rlt == "full":
                    M.constraint(X.index(i, j), Domain.greaterThan(0.0))
                    M.constraint(Expr.sub(X.index(i, j),
                                          Expr.sub(Expr.add(x.index(i), x.index(j)), 1.0)),
                                 Domain.greaterThan(0.0))
        if triangle:
            for G, g0, rhs in triangle_rows():
                M.constraint(Expr.add(Expr.dot(Matrix.dense(G), X), Expr.dot(g0, x)),
                             Domain.lessThan(-rhs))
        M.objective(ObjectiveSense.Minimize,
                    Expr.add(Expr.dot(Matrix.dense(Q), X), Expr.dot(C, x)))
        M.solve()
        return M.primalObjValue(), np.array(Y.level()).reshape(N + 1, N + 1)


def signature(Yv, tol=1e-5):
    """(rank, l, d, u) for the optimal solution, sorted by x."""
    x = Yv[0, 1:].copy(); X = Yv[1:, 1:].copy()
    order = np.argsort(x)
    x = x[order]; X = X[np.ix_(order, order)]
    ev = np.linalg.eigvalsh(Yv)
    rank = int((ev > tol * max(1.0, ev.max())).sum())
    l = d = u = 0
    for i in range(N):
        for j in range(N):
            if abs(X[i, j] - x[i]) <= tol:
                if i > j: l += 1
                elif i == j: d += 1
                else: u += 1
    return rank, l, d, u, x, X


def run_numerics():
    print("\n" + "=" * 72)
    print("Part B -- numerical (Mosek)")
    print("=" * 72)

    v_upper, Yv = solve_relaxation("upper", False)
    rank, l, d, u, xs, Xs = signature(Yv)
    print("\n[B1] relaxation (2)")
    print(f"     optimal value            : {v_upper:.7f}")
    print(f"     rank of Y(x,X)           : {rank}")
    print(f"     active (l, d, u)         : ({l}, {d}, {u}),  a = {l+d+u}")
    print(f"     Lemma 1 counting bound   : r(r+1) = {rank*(rank+1)} <= 2(a+1) = {2*(l+d+u+1)}")
    print(f"     x (sorted)               : {np.round(xs, 6)}")
    R = np.array([[xs[i] - Xs[i, j] for j in range(N)] for i in range(N)])
    act = np.sort(R[R <= 1e-5]); sl = np.sort(R[R > 1e-5])
    print(f"     activity is unambiguous  : largest active residual {act.max():.1e},"
          f" smallest slack residual {sl.min():.1e}")

    v_full, _ = solve_relaxation("full", False)
    v_tri, Ytri = solve_relaxation("full", True)
    print("\n[B2] Table 6 -- cut ladder")
    print(f"     {'PSD + RLT upper bounds, i.e. (2)':<48}{v_upper:>12.7f}")
    print(f"     {'PSD + full RLT':<48}{v_full:>12.7f}")
    print(f"     {'PSD + full RLT + triangle inequalities':<48}{v_tri:>12.7f}")
    print( "     true optimal value of the box QP is 0")

    print("\n[B3] rational certificate against PSD + full RLT + triangle")
    cert = rational_certificate(Ytri)
    if cert is None:
        print("     no certificate found at the denominators tried")
    else:
        obj, den, t = cert
        print(f"     mixing parameter t       : {t}")
        print(f"     denominator              : {den}")
        print(f"     exact objective value    : {obj}  = {float(obj):.3e}")
        print( "     satisfies exactly: PSD, all 16 RLT (both directions),")
        print( "     and all 16 triangle inequalities, with negative objective.")


def rational_certificate(Ytri, denominators=(10**8,), ts=None):
    """Mix the computed optimum with a strictly feasible point, round, verify."""
    if ts is None:
        ts = [F(k, 10**7) for k in range(1, 400)]
    xopt = Ytri[0, 1:]; Xopt = Ytri[1:, 1:]
    xstr = np.full(N, 0.5)
    Xstr = 0.25 * np.ones((N, N)) + 0.125 * np.eye(N)
    Qf = [[F(v) for v in row] for row in Q_INT]
    Cf = [F(v) for v in C_INT]
    tri = triangle_rows()

    for den in denominators:
        for t in ts:
            tf = float(t)
            xm = (1 - tf) * xopt + tf * xstr
            Xm = (1 - tf) * Xopt + tf * Xstr
            x = [F(round(v * den), den) for v in xm]
            X = [[F(round(Xm[i][j] * den), den) for j in range(N)] for i in range(N)]
            for i in range(N):                       # symmetrize exactly
                for j in range(i + 1, N):
                    X[j][i] = X[i][j]
            if not all(X[i][j] <= x[i] for i in range(N) for j in range(N)):
                continue
            if not all(X[i][j] >= 0 for i in range(N) for j in range(N)):
                continue
            if not all(X[i][j] >= x[i] + x[j] - 1 for i in range(N) for j in range(N)):
                continue
            ok = True
            for G, g0, rhs in tri:
                s = (sum(F(G[i][j]).limit_denominator(4) * X[i][j]
                         for i in range(N) for j in range(N))
                     + sum(F(g0[i]).limit_denominator(4) * x[i] for i in range(N)))
                if s > -F(rhs).limit_denominator(4):
                    ok = False; break
            if not ok:
                continue
            from verify_counterexample import det_exact
            Y = [[F(1)] + x] + [[x[i]] + X[i] for i in range(N)]
            if any(det_exact([r[:k] for r in Y[:k]]) < 0 for k in range(1, N + 2)):
                continue
            obj = (sum(Cf[i] * x[i] for i in range(N))
                   + sum(Qf[i][j] * X[i][j] for i in range(N) for j in range(N)))
            if obj < 0:
                return obj, den, t
    return None
