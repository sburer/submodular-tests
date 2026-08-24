"""
Geometric illustration for the SECOND n = 4 counterexample: the one the
triangle inequalities do NOT repair.

This is the companion to `gap_figure.py`, which draws the same projection for
Rick's instance.  There, cutting the relaxation with a single triangle
inequality moved its leftmost point from -13/4 back to 0, closing the gap
exactly.  Here the same cut removes 99.9% of the gap and then stops, and the
point of this figure is to make that residual visible.

--------------------------------------------------------------------------
The instance
--------------------------------------------------------------------------
    Q = [   8  -14    0    0 ]         c = (12, 29, 0, 0)
        [ -14   25  -25    0 ]
        [   0  -25   25  -14 ]
        [   0    0  -14    8 ]

Submodular (all off-diagonals <= 0), indefinite, and structurally a path
1-2-3-4: the non-adjacent couplings Q_13, Q_14, Q_24 all vanish.  It was found
by an adversarial search seeded at the optimal solution of Rick's instance.
The QP optimal value is 0, attained at x = 0 (complete enumeration of the 3^4
KKT patterns in exact rational arithmetic); the relaxation of the paper attains
approximately -0.1220566.

--------------------------------------------------------------------------
What is being drawn
--------------------------------------------------------------------------
Exactly the projection of `gap_figure.py`: the full bodies in the lifted space
of (x, X) are mapped to the plane

    t(x, X) = c'x + Q . X                        (the objective value itself)
    g(x, X) = x_3 + X_14 - X_13 - X_34           (a triangle-inequality slack)

so that the leftmost point of each shadow IS the optimal value of the
corresponding problem, and g >= 0 is the triangle inequality
X_13 + X_34 <= x_3 + X_14.  (At the relaxation's optimum exactly two triangle
inequalities are violated, on the triples {1,2,4} and {1,3,4}, each by 0.0346 --
the same two triples as in Rick's instance.)

Three bodies are drawn:

    * our relaxation    -- PSD + the RLT upper bounds X_ij <= x_i;
    * + TRI             -- the same, plus full RLT and all 16 triangle
                           inequalities;
    * exact hull QPB_4.

Adding the ETRI1/2/3 inequalities of Anstreicher and Puges, and their conic
strengthening, changes the middle body's leftmost point by less than 1e-7, so
the picture would be unchanged; only TRI is drawn to keep this script
self-contained.

The main panel shows the large gap that the triangle inequalities remove.  The
inset magnifies the neighbourhood of the origin by a factor of about 1000 to
show what they leave behind: the TRI-strengthened body still reaches
t = -0.0001432 < 0, while the hull stops exactly at t = 0.  That residual is
not solver noise -- it is certified exactly in the accompanying note by a
rational point that satisfies the PSD condition, all RLT, all 16 TRI and all
400 ETRI inequalities, with objective -2483/20000000.

Dependencies: numpy, scipy, matplotlib, cvxpy (Clarabel).  No Gurobi or Mosek.

Usage:  python gap_figure2.py --outbase gap_projection2
"""

import argparse
from itertools import combinations, product

import cvxpy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 4
Q = np.array([[  8., -14.,   0.,   0.],
              [-14.,  25., -25.,   0.],
              [  0., -25.,  25., -14.],
              [  0.,   0., -14.,   8.]])
C = np.array([12., 29., 0., 0.])

G = np.zeros((N, N))
G[0, 3] = G[3, 0] = 0.5
G[0, 2] = G[2, 0] = -0.5
G[2, 3] = G[3, 2] = -0.5
GC = np.zeros(N)
GC[2] = 1.0

t_of = lambda x: C @ x + x @ Q @ x
g_of = lambda x: GC @ x + x @ G @ x

RELAX_TIP = (-0.1220566, -0.0346212)   # image of the relaxation's optimum
TRI_TIP = (-0.0001432, 0.0)            # image of the TRI-strengthened optimum
HULL_TIP = (0.0, 0.0)                  # image of the true optimum x = 0

PATTERNS = list(product([0.0, 1.0, None], repeat=N))


# ----------------------------------------------------------------------
# Support function of the exact hull QPB_4: global max of a quadratic over the
# box, by complete KKT enumeration (singular reduced Hessians handled by least
# squares) with a vectorized multistart ascent as a safety net.
# ----------------------------------------------------------------------
def box_qp_max(A, b, starts=300, iters=500, seed=1):
    best, best_x = -np.inf, None
    for pat in PATTERNS:
        free = [i for i in range(N) if pat[i] is None]
        fixed = [i for i in range(N) if pat[i] is not None]
        x = np.zeros(N)
        for i in fixed:
            x[i] = pat[i]
        if free:
            M = 2 * A[np.ix_(free, free)]
            rhs = -b[free] - (2 * A[np.ix_(free, fixed)] @ x[fixed] if fixed else 0.0)
            sol, _, rank, _ = np.linalg.lstsq(M, rhs, rcond=None)
            if rank < len(free) and np.linalg.norm(M @ sol - rhs) > 1e-7:
                continue
            x[free] = sol
        if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
            continue
        x = np.clip(x, 0, 1)
        v = x @ A @ x + b @ x
        if v > best:
            best, best_x = v, x.copy()
    for z in product([0.0, 1.0], repeat=N):
        y = np.array(z)
        v = y @ A @ y + b @ y
        if v > best:
            best, best_x = v, y
    rng = np.random.default_rng(seed)
    Xs = rng.random((starts, N))
    step = 1.0 / (2 * np.abs(A).sum() + np.abs(b).sum() + 1)
    for _ in range(iters):
        Xs = np.clip(Xs + step * (2 * Xs @ A + b), 0, 1)
    vals = np.einsum("ij,jk,ik->i", Xs, A, Xs) + Xs @ b
    k = int(vals.argmax())
    if vals[k] > best:
        best, best_x = vals[k], Xs[k]
    return best_x


def hull_point(a, b):
    x = box_qp_max(a * Q + b * G, a * C + b * GC)
    return np.array([t_of(x), g_of(x)])


# ----------------------------------------------------------------------
# Support functions of the two relaxations, as parametrized problems.
# ----------------------------------------------------------------------
def make_sdp(with_tri):
    A, B = cp.Parameter(), cp.Parameter()
    Y = cp.Variable((N + 1, N + 1), PSD=True)
    x, X = Y[0, 1:], Y[1:, 1:]
    cons = ([Y[0, 0] == 1, X == X.T]
            + [X[i, j] <= x[i] for i in range(N) for j in range(N)])
    if with_tri:
        cons += [X[i, j] >= 0 for i in range(N) for j in range(i, N)]
        cons += [X[i, j] >= x[i] + x[j] - 1 for i in range(N) for j in range(i, N)]
        for (i, j, k) in combinations(range(N), 3):
            for (p, q, r) in [(i, j, k), (j, i, k), (k, i, j)]:
                cons.append(X[p, q] + X[p, r] <= x[p] + X[q, r])
            cons.append(x[i] + x[j] + x[k] <= X[i, j] + X[i, k] + X[j, k] + 1)
    t = cp.sum(cp.multiply(Q, X)) + C @ x
    g = cp.sum(cp.multiply(G, X)) + GC @ x
    prob = cp.Problem(cp.Maximize(A * t + B * g), cons)

    def point(a, b):
        A.value, B.value = float(a), float(b)
        prob.solve(solver=cp.CLARABEL)
        return np.array([t.value, g.value])
    return point


# ----------------------------------------------------------------------
# Trace a shadow's boundary, refining where consecutive support points are far
# apart in the metric of whichever window we are about to draw.
# ----------------------------------------------------------------------
def trace_boundary(support_fn, sx, sy, budget=260, eps=0.004):
    pts = {}

    def at(phi):
        if phi not in pts:
            pts[phi] = support_fn(np.cos(phi), np.sin(phi))
        return pts[phi]

    for phi in list(np.linspace(0, 2 * np.pi, 49, endpoint=False)) + [2 * np.pi - 1e-9]:
        at(phi)
    dist = lambda p, q: np.hypot((p[0] - q[0]) / sx, (p[1] - q[1]) / sy)
    for _ in range(budget):
        angles = sorted(pts)
        gaps = [(dist(pts[angles[i]], pts[angles[i + 1]]), angles[i], angles[i + 1])
                for i in range(len(angles) - 1) if angles[i + 1] - angles[i] > 1e-7]
        if not gaps:
            break
        worst = max(gaps)
        if worst[0] < eps:
            break
        at(0.5 * (worst[1] + worst[2]))
    return np.array([pts[phi] for phi in sorted(pts)])


BODIES = [("relax", "#4C72B0", r"Our relaxation", 0.15),
          ("tri", "#DD8452", r"$+$ triangle inequalities", 0.30),
          ("hull", "#55A868", r"Exact convex hull $\mathrm{QPB}_4$", 0.45)]


def make_figure(outbase):
    fns = {"relax": make_sdp(False), "tri": make_sdp(True), "hull": hull_point}
    main = {k: trace_boundary(f, 0.70, 0.10) for k, f in fns.items()}
    zoom = {k: trace_boundary(f, 7.0e-4, 2.4e-4) for k, f in fns.items()}

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    loop = lambda P: np.vstack([P, P[0]])

    # blue and green as filled bodies; the TRI body is drawn as an outline on
    # top, since at this scale it is indistinguishable from the hull.
    for key, col, lab, al in BODIES:
        P = loop(main[key])
        if key == "tri":
            ax.plot(P[:, 0], P[:, 1], color=col, lw=2.4, ls=(0, (5, 2)), zorder=5, label=lab)
            continue
        ax.fill(P[:, 0], P[:, 1], color=col, alpha=al, zorder=1)
        ax.plot(P[:, 0], P[:, 1], color=col, lw=1.7, zorder=3, label=lab)
    ax.axhline(0.0, color="#C44E52", ls="--", lw=1.2, zorder=4)
    ax.axvline(0.0, color="0.5", ls=":", lw=1.0, zorder=1)
    ax.plot(*RELAX_TIP, marker="*", ms=13, color="#4C72B0", mec="k", mew=0.6, zorder=6)
    ax.plot(*HULL_TIP, marker="o", ms=5.5, color="#55A868", mec="k", mew=0.6, zorder=6)
    ax.annotate("relaxation optimum\n" r"$\approx -0.1221$", xy=RELAX_TIP, xytext=(-0.155, 0.019),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(RELAX_TIP[0], -0.041), xytext=(0.0, -0.041),
                arrowprops=dict(arrowstyle="<->", lw=1.1))
    ax.text(-0.061, -0.0475, "gap removed by TRI", ha="center", fontsize=9)
    ax.text(-0.163, -0.0135, "triangle inequality", ha="left", va="center",
            fontsize=9, color="#C44E52")
    ax.annotate("TRI leaves a residual\nhere: see inset", xy=(0.0, -0.004),
                xytext=(0.075, -0.033), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xlim(-0.17, 0.60)
    ax.set_ylim(-0.052, 0.055)
    ax.set_xlabel(r"objective value  $c^{T}x + Q\bullet X$")
    ax.set_ylabel(r"triangle slack  $x_3 + X_{14} - X_{13} - X_{34}$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=8.5,
              frameon=False, borderaxespad=0.0, handlelength=1.6, columnspacing=1.4)

    # ---- inset: what the triangle inequalities leave behind ----------------
    axi = ax.inset_axes([0.575, 0.34, 0.40, 0.50])
    for key, col, lab, al in BODIES:
        P = loop(zoom[key])
        axi.fill(P[:, 0], P[:, 1], color=col, alpha=al, zorder=1)
        axi.plot(P[:, 0], P[:, 1], color=col, lw=1.4, zorder=3)
    axi.axvline(0.0, color="0.5", ls=":", lw=1.0)
    axi.plot(*TRI_TIP, marker="D", ms=4.5, color="#DD8452", mec="k", mew=0.5, zorder=6)
    axi.plot(*HULL_TIP, marker="o", ms=4.5, color="#55A868", mec="k", mew=0.5, zorder=6)
    axi.annotate("", xy=(TRI_TIP[0], 6.0e-5), xytext=(0.0, 6.0e-5),
                 arrowprops=dict(arrowstyle="<->", lw=0.9))
    axi.text(-7.0e-5, 8.0e-5, r"$1.43\times10^{-4}$", ha="center", fontsize=8)
    axi.set_xlim(-2.6e-4, 5.0e-4)
    axi.set_ylim(-4.0e-5, 1.9e-4)
    axi.set_xticks([-2e-4, 0, 4e-4])
    axi.set_xticklabels([r"$-2{\cdot}10^{-4}$", "0", r"$4{\cdot}10^{-4}$"], fontsize=7)
    axi.set_yticks([])
    axi.tick_params(length=2)
    axi.set_title("what TRI leaves behind", fontsize=8.5, pad=3)
    for s in axi.spines.values():
        s.set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outbase}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.02)
    return main, zoom


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render the second n = 4 counterexample figure.")
    ap.add_argument("--outbase", default="gap_projection2",
                    help="output path without extension (writes .pdf and .png)")
    args = ap.parse_args()
    main, zoom = make_figure(args.outbase)
    for k, _, _, _ in BODIES:
        P = main[k]
        i = int(P[:, 0].argmin())
        print(f"{k:6s} leftmost point of the shadow: t = {P[i,0]:.7f}")
    print(f"wrote {args.outbase}.pdf and {args.outbase}.png")
