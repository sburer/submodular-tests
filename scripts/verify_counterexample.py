"""Verification of the n = 4 counterexample (Example 4, Proposition 12, Table 6).

Reproduces every computational claim the paper makes about the instance

    Q = [[  8, -14,   0,   0],        c = (12, 29, 0, 0)
         [-14,  25, -25,   0],
         [  0, -25,  25, -14],
         [  0,   0, -14,   8]]

Part A (exact, no solver required)
    A1  the box QP has optimal value 0, attained only at x = 0, by complete
        enumeration of the 3^4 KKT active-set patterns over the rationals
    A2  (xbar, Xbar) is feasible for the relaxation with objective exactly
        -109/1024, via the leading principal minors of Y(xbar, Xbar)
    A3  (xbar, Xbar) also satisfies the RLT lower bounds, so the full SDP-RLT
        relaxation attains the same value

Part B (numerical, requires Mosek)
    B1  the optimal value of the relaxation, and the rank and active-set
        signature of its optimal solution
    B2  Table 6: the cut ladder
    B3  a rational certificate that a gap survives PSD + full RLT + triangle

Run:  python verify_counterexample.py            (all parts)
      python verify_counterexample.py --exact    (Part A only, no Mosek)
"""

import argparse
import itertools
from fractions import Fraction as F

import numpy as np

# --------------------------------------------------------------------------
# The instance (equation (3) of the paper)
# --------------------------------------------------------------------------

Q_INT = [[8, -14, 0, 0], [-14, 25, -25, 0], [0, -25, 25, -14], [0, 0, -14, 8]]
C_INT = [12, 29, 0, 0]
N = 4

# The feasible point of Proposition 12
XBAR = [F(3, 32), F(3, 16), F(9, 16), F(3, 4)]
XXBAR = [
    [F(3, 32),   F(3, 32),    F(3, 32),   F(61, 1024)],
    [F(3, 32),   F(125, 1024), F(3, 16),  F(3, 16)],
    [F(3, 32),   F(3, 16),    F(231, 512), F(9, 16)],
    [F(61, 1024), F(3, 16),   F(9, 16),   F(3, 4)],
]


def det_exact(M):
    """Exact determinant by cofactor expansion (M is small)."""
    n = len(M)
    if n == 1:
        return M[0][0]
    tot = 0
    for j in range(n):
        minor = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        tot += ((-1) ** j) * M[0][j] * det_exact(minor)
    return tot


def solve_exact(A, b):
    """Solve A z = b exactly.  Returns None if singular or inconsistent."""
    m = len(A)
    if m == 0:
        return []
    M = [list(map(F, A[i])) + [F(b[i])] for i in range(m)]
    piv = []
    r = 0
    for col in range(m):
        p = next((i for i in range(r, m) if M[i][col] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = M[r][col]
        M[r] = [v / inv for v in M[r]]
        for i in range(m):
            if i != r and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * b_ for a, b_ in zip(M[i], M[r])]
        piv.append(col)
        r += 1
    for i in range(r, m):                      # inconsistent row 0 = nonzero
        if M[i][m] != 0:
            return None
    if r < m:                                  # singular but consistent
        return "degenerate"
    z = [F(0)] * m
    for i, col in enumerate(piv):
        z[col] = M[i][m]
    return z


def a1_box_qp():
    """Enumerate the 3^4 KKT active-set patterns over Q."""
    Q = [[F(v) for v in row] for row in Q_INT]
    c = [F(v) for v in C_INT]

    def f(x):
        return sum(c[i] * x[i] for i in range(N)) + sum(
            Q[i][j] * x[i] * x[j] for i in range(N) for j in range(N))

    kkt, singular, degenerate = [], 0, 0
    for pattern in itertools.product([0, 1, "free"], repeat=N):
        free = [i for i in range(N) if pattern[i] == "free"]
        fixed = {i: F(pattern[i]) for i in range(N) if pattern[i] != "free"}
        # stationarity in the free block: 2 Q_FF z = -c_F - 2 Q_FB x_B
        A = [[2 * Q[i][j] for j in free] for i in free]
        rhs = [-c[i] - 2 * sum(Q[i][k] * v for k, v in fixed.items()) for i in free]
        if free:
            if det_exact([[F(v) for v in row] for row in A]) == 0:
                singular += 1
            sol = solve_exact(A, rhs)
            if sol is None:
                continue
            if sol == "degenerate":
                degenerate += 1
                continue
            if any(not (0 <= z <= 1) for z in sol):
                continue
        else:
            sol = []
        x = [None] * N
        for i, v in fixed.items():
            x[i] = v
        for idx, i in enumerate(free):
            x[i] = sol[idx]
        kkt.append((f(x), tuple(x)))

    kkt.sort(key=lambda t: t[0])
    best_val, best_x = kkt[0]
    minimizers = sorted({x for v, x in kkt if v == best_val})
    return best_val, minimizers, singular, degenerate, len(kkt)


def a2_feasible_point():
    """Feasibility and objective value of (xbar, Xbar)."""
    Q = [[F(v) for v in row] for row in Q_INT]
    c = [F(v) for v in C_INT]

    rlt_upper = all(XXBAR[i][j] <= XBAR[i] for i in range(N) for j in range(N))
    symmetric = all(XXBAR[i][j] == XXBAR[j][i] for i in range(N) for j in range(N))

    Y = [[F(1)] + XBAR] + [[XBAR[i]] + XXBAR[i] for i in range(N)]
    minors = [det_exact([row[:k] for row in Y[:k]]) for k in range(1, N + 2)]

    obj = (sum(c[i] * XBAR[i] for i in range(N))
           + sum(Q[i][j] * XXBAR[i][j] for i in range(N) for j in range(N)))
    return rlt_upper, symmetric, minors, obj


def a3_rlt_lower():
    """(xbar, Xbar) also satisfies the RLT lower bounds."""
    nonneg = all(XXBAR[i][j] >= 0 for i in range(N) for j in range(N))
    mccormick = all(XXBAR[i][j] >= XBAR[i] + XBAR[j] - 1
                    for i in range(N) for j in range(N))
    return nonneg, mccormick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", action="store_true",
                    help="run only the exact part (no Mosek required)")
    args = ap.parse_args()

    print("=" * 72)
    print("Part A -- exact rational arithmetic, no solver")
    print("=" * 72)

    Qa = np.array(Q_INT, dtype=float)
    print("\n[A0] instance")
    print("     off-diagonals <= 0 (submodular):",
          all(Q_INT[i][j] <= 0 for i in range(N) for j in range(N) if i != j))
    print("     eigenvalues of Q:", np.round(np.linalg.eigvalsh(Qa), 2))

    val, minimizers, singular, degenerate, n_kkt = a1_box_qp()
    print("\n[A1] box QP by complete KKT enumeration (3^4 = 81 patterns)")
    print(f"     KKT points found            : {n_kkt}")
    print(f"     singular reduced Hessians   : {singular}")
    print(f"     degenerate (consistent) ones: {degenerate}")
    print(f"     optimal value               : {val}")
    print(f"     minimizers                  : {minimizers}")
    assert val == 0 and minimizers == [(F(0),) * N], "QP claim failed"

    rlt_u, sym, minors, obj = a2_feasible_point()
    print("\n[A2] the feasible point (xbar, Xbar)")
    print(f"     Xbar symmetric              : {sym}")
    print(f"     all 16 Xbar_ij <= xbar_i    : {rlt_u}")
    print( "     leading principal minors of Y(xbar,Xbar):")
    for m in minors:
        print(f"         {m}")
    print(f"     all strictly positive       : {all(m > 0 for m in minors)}")
    print(f"     objective c'xbar + Q.Xbar   : {obj}  ( = {float(obj):.10f} )")
    assert rlt_u and all(m > 0 for m in minors) and obj == F(-109, 1024)

    nonneg, mcc = a3_rlt_lower()
    print("\n[A3] RLT lower bounds at the same point")
    print(f"     Xbar_ij >= 0                : {nonneg}")
    print(f"     Xbar_ij >= xbar_i+xbar_j-1  : {mcc}")
    print( "     => the full SDP-RLT relaxation admits it and attains -109/1024")
    assert nonneg and mcc

    print("\n[A] CONCLUSION: QP optimal value 0, relaxation <= -109/1024,")
    print("    so the relaxation gap is at least 109/1024 > 0.")

    if args.exact:
        print("\n(--exact given; skipping the numerical part)")
        return

    from counterexample_numerics import run_numerics
    run_numerics()


if __name__ == "__main__":
    main()
