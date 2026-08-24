"""
Geometric illustration of why the SDP relaxation is tight for *submodular*
quadratic minimization over the box, and why the RLT *lower* bounds may be
dropped in that case.

--------------------------------------------------------------------------
What is being drawn
--------------------------------------------------------------------------
For n = 3 we lift x in [0,1]^3 to a 4x4 moment matrix

        Y = [ 1   x^T ]        with   X = (X_ij) = "E[x_i x_j]".
            [ x    X  ]

We compare three convex bodies, all living in the same lifted space, ordered
by inclusion (each is a relaxation of the previous one):

    CP(J)   subset   L_RLT   subset   L_upper

  * CP(J)   -- the exact convex hull conv{ (1,x)(1,x)^T : x in [0,1]^3 }.
              Minimizing a linear objective over CP(J) gives the *true* optimal
              value of the box QP.
  * L_RLT   -- PSD cone intersected with the FULL McCormick / RLT inequalities
              for every pair i <= j:
                    X_ij <= x_i ,  X_ij <= x_j ,          (upper bounds)
                    X_ij >= 0 ,    X_ij >= x_i + x_j - 1 . (lower bounds)
              This is the Anstreicher-Burer relaxation (tight for n = 2).
  * L_upper -- PSD cone intersected with ONLY the RLT upper bounds
              (X <= x e^T). This is the relaxation studied in the paper
              [eq. (2)]: for submodular data the objective is nonincreasing in
              the off-diagonals, so the lower bounds are never active and can
              be discarded.

--------------------------------------------------------------------------
Why the picture demonstrates the theorem
--------------------------------------------------------------------------
Fix the first and second moments  x_i  and  X_ii  and treat the three
off-diagonals (X_12, X_13, X_23) as the free coordinates.  A submodular
objective is  min  Q . Y  with  Q_12, Q_13, Q_23 <= 0, i.e. it *maximizes* a
nonnegative combination of (X_12, X_13, X_23).  Hence the optimizer sits on the
part of the boundary whose outward normal lies in the nonnegative orthant.

Key fact (verified numerically in this script): the support functions of the
three bodies AGREE for every nonnegative direction, and disagree only for
directions that have a negative component.  Therefore:

    * submodular objective  ->  optimum on the shared boundary  ->  all three
      bodies give the same value  ->  the SDP relaxation is TIGHT, and dropping
      the lower bounds does not change the optimum;
    * non-submodular objective  ->  optimum on a boundary where the bodies
      separate  ->  a gap appears.

To see this in 2-D we project onto  (u, v) = (X_12 + X_13, X_13 + X_23).  The
projection is linear, so inclusions are preserved and the shadow's support in a
2-D direction (a, b) equals the 3-D support in direction (a, a+b, b); thus
(a, b) >= 0  <=>  submodular direction.

--------------------------------------------------------------------------
Choice of instance
--------------------------------------------------------------------------
We fix the marginals to  x_i = X_ii = 5/8  for all i.  Because X_ii = x_i, any
distribution consistent with these moments is supported on {0,1}^3 (equality in
E[x_i^2] <= E[x_i] forces binary support), so CP(J) is the correlation polytope
and is computed *exactly* over the 8 binary vertices -- no discretization.

The value 5/8 is chosen only for visual clarity: it makes both gaps visible.
(At x_i = 1/2 the RLT lower bounds are implied by the PSD constraint, so L_RLT
and L_upper coincide; at x_i = 2/3 the L_RLT-vs-CP gap is a very thin sliver.)
The qualitative conclusion -- coincidence on the submodular boundary, a gap
elsewhere -- is independent of this particular choice.

Dependencies: numpy, scipy, matplotlib, and Mosek (Fusion API) for the two SDP
bodies.  The CP body uses only scipy.optimize.linprog.
"""

import itertools
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, SolutionType

# ----------------------------------------------------------------------
# Instance: the fixed first and second moments (see module docstring).
# ----------------------------------------------------------------------
X_MARGINAL = 5.0 / 8.0                 # x_i  = 5/8 for i = 1,2,3
DIAG = 5.0 / 8.0                       # X_ii = 5/8 (so support is binary)
x = np.array([X_MARGINAL] * 3)
Xii = np.array([DIAG] * 3)

# The 8 vertices of {0,1}^3 (the support of every feasible distribution here).
BINARY_VERTICES = np.array(list(itertools.product([0, 1], repeat=3)), dtype=float)

# Linear map to the 2-D drawing plane:  (u, v) = (X_12 + X_13, X_13 + X_23).
PROJ = np.array([[1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0]])

# Index bookkeeping: the three off-diagonal slots and which x-indices they pair.
OFFDIAG = [(1, 2), (1, 3), (2, 3)]     # positions in the 4x4 matrix Y
PAIR = [(0, 1), (0, 2), (1, 2)]        # corresponding (i, j) into x


# ----------------------------------------------------------------------
# Support function of CP(J), computed EXACTLY over the binary vertices.
#
# max  d . (X_12, X_13, X_23)   over distributions p on {0,1}^3 with the fixed
# marginals.  For a vertex w in {0,1}^3 the off-diagonal moment contribution is
# (w1 w2, w1 w3, w2 w3); the objective is linear in p, so this is an LP.
# ----------------------------------------------------------------------
def cp_support(direction):
    # Off-diagonal moment vector contributed by each binary vertex.
    M = np.array([[w[0] * w[1], w[0] * w[2], w[1] * w[2]] for w in BINARY_VERTICES])
    c = -(M @ direction)                       # linprog minimizes; we want max
    A_eq = np.vstack([np.ones(len(BINARY_VERTICES)),      # weights sum to 1
                      BINARY_VERTICES.T])                 # match each marginal x_i
    b_eq = np.concatenate([[1.0], x])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return None
    p = res.x
    return M.T @ p                             # the maximizing (X_12, X_13, X_23)


# ----------------------------------------------------------------------
# Support function of an SDP body (PSD cone + a chosen set of RLT inequalities).
#
# The diagonal and first row/column of Y are pinned to the fixed moments; the
# three off-diagonals are free.  `full_rlt=True` adds the lower bounds.
# ----------------------------------------------------------------------
def sdp_support(direction, full_rlt):
    M = Model()
    Y = M.variable("Y", Domain.inPSDCone(4))
    M.constraint(Y.index(0, 0), Domain.equalsTo(1.0))          # top-left entry = 1
    for i in range(3):
        M.constraint(Y.index(0, i + 1), Domain.equalsTo(x[i]))     # first row = x
        M.constraint(Y.index(i + 1, i + 1), Domain.equalsTo(Xii[i]))  # diagonal fixed
    for (a, b), (i, j) in zip(OFFDIAG, PAIR):
        Xab = Y.index(a, b)
        # Upper (McCormick) bounds -- always present:
        M.constraint(Expr.sub(min(x[i], x[j]), Xab), Domain.greaterThan(0.0))
        if full_rlt:
            # Lower (McCormick) bounds -- only in the full-RLT body:
            M.constraint(Xab, Domain.greaterThan(0.0))                    # X_ij >= 0
            M.constraint(Expr.sub(Xab, x[i] + x[j] - 1.0),
                         Domain.greaterThan(0.0))                          # X_ij >= x_i+x_j-1
    # Maximize the linear functional d . (X_12, X_13, X_23).
    obj = Expr.add(Expr.add(Expr.mul(float(direction[0]), Y.index(1, 2)),
                            Expr.mul(float(direction[1]), Y.index(1, 3))),
                   Expr.mul(float(direction[2]), Y.index(2, 3)))
    M.objective(ObjectiveSense.Maximize, obj)
    M.setSolverParam("log", 0)
    M.solve()
    if M.getProblemStatus(SolutionType.Default).name not in ("PrimalAndDualFeasible", "Optimal"):
        M.dispose()
        return None
    v = np.array([Y.index(1, 2).level()[0], Y.index(1, 3).level()[0], Y.index(2, 3).level()[0]])
    M.dispose()
    return v


# ----------------------------------------------------------------------
# Trace the boundary of a body's 2-D projection by sweeping directions.
#
# For a 2-D angle phi -> direction (a, b), the shadow's supporting point is the
# projection of the 3-D support point in direction (a, a+b, b).
# ----------------------------------------------------------------------
def projected_boundary(support_fn, n_angles=480):
    pts = []
    for phi in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        a, b = np.cos(phi), np.sin(phi)
        p3 = support_fn(np.array([a, a + b, b]))
        if p3 is not None:
            pts.append(PROJ @ p3)
    return np.array(pts)


def sanity_check_coincidence(n_dirs=400, seed=0):
    """Confirm the three support functions agree on nonnegative directions and
    separate elsewhere (this is the geometric content of the theorem)."""
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n_dirs, 3))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    max_pos = max_mix = 0.0
    for d in D:
        u = sdp_support(d, full_rlt=False)   # largest body
        c = cp_support(d)                    # smallest body
        if u is None or c is None:
            continue
        gap = d @ u - d @ c                  # support gap L_upper - CP >= 0
        if np.all(d >= 0):
            max_pos = max(max_pos, gap)
        else:
            max_mix = max(max_mix, gap)
    return max_pos, max_mix


# ----------------------------------------------------------------------
# Build the figure.
# ----------------------------------------------------------------------
def make_figure(outbase):
    upper = projected_boundary(lambda d: sdp_support(d, full_rlt=False))
    rlt = projected_boundary(lambda d: sdp_support(d, full_rlt=True))
    cp = projected_boundary(cp_support)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    # Draw the three nested regions, outer to inner.
    for reg, color, label, alpha in [
        (upper, "#4C72B0", "Our relaxation", 0.16),
        (rlt,   "#DD8452", "PSD + full McCormick", 0.28),
        (cp,    "#55A868", "Exact convex hull", 0.55),
    ]:
        loop = np.vstack([reg, reg[0]])
        ax.fill(loop[:, 0], loop[:, 1], color=color, alpha=alpha, zorder=1)
        ax.plot(loop[:, 0], loop[:, 1], color=color, lw=1.6, zorder=2, label=label)

    ax.set_xlabel(r"$X_{12} + X_{13}$")
    ax.set_ylabel(r"$X_{13} + X_{23}$")
    ax.set_aspect("equal")
    ax.set_xlim(0.2, 1.4)
    ax.set_ylim(0.2, 1.4)
    # Legend order: tightest (CP) -> middle (RLT) -> loosest (upper); reverse
    # of the draw order above, which is outer-to-inner for correct layering.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left",
              bbox_to_anchor=(0.02, 0.96), fontsize=8, framealpha=0.95)
    ax.set_title(r"Projection to $(X_{12}+X_{13},\,X_{13}+X_{23})$", pad=12)
    fig.tight_layout()
    # Save both a vector PDF (for LaTeX) and a raster PNG (for quick viewing),
    # cropped tightly to the drawn content with no extra whitespace.
    for ext in ("pdf", "png"):
        fig.savefig(f"{outbase}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0)
    return upper, rlt, cp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render the tightness-projection figure.")
    ap.add_argument("--outbase", default="tightness_projection",
                    help="output path without extension (writes .pdf and .png)")
    args = ap.parse_args()

    mp, mm = sanity_check_coincidence()
    print(f"instance: x_i = X_ii = {X_MARGINAL:.4f}")
    print(f"support gap on nonnegative (submodular) directions: {mp:.2e}  (should be ~0)")
    print(f"support gap on mixed-sign  directions:              {mm:.3f}  (a genuine gap)")
    make_figure(args.outbase)
    print(f"wrote {args.outbase}.pdf and {args.outbase}.png")
