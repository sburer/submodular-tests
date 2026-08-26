"""Sparsity study (paper Section 6 / Conclusions): can RLT bounds on the
zero-Q_ij pairs be dropped without loosening the SDP relaxation?

For each instance, solves the full SDP relaxation (all n^2 RLT bounds) and a
reduced one that keeps only the bounds X_ij <= x_j on pairs with Q_ij != 0
(the diagonal is always kept). Reports how close the two bounds are and how
often the reduced relaxation remains tight.

Two settings are examined:
  (a) random sparse Q at several densities, and
  (b) an explicitly block-separable Q coupling only the pairs (1,2),(3,4),(5,6).

Requires Mosek and Gurobi to be installed and licensed.
"""
import os
import sys

import numpy as np
import numpy.random as npr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from analysis_functions import generate_instance, solve_qp, solve_sdp


def keep_mask(Q, n):
    keep = (np.abs(Q) > 1e-12)
    np.fill_diagonal(keep, True)
    return keep


def run_random(n, seeds, density):
    worst_diff = 0.0
    worst_seed = None
    still_tight = 0
    N = 0
    kept_frac = []
    for s in seeds:
        Qc, Q, c, mask = generate_instance(n, s, density=density)
        x_opt, pval = solve_qp(n, Q, c)
        if x_opt is None:
            continue
        keep = keep_mask(Q, n)
        dval_full, _ = solve_sdp(n, Qc)
        dval_red, _ = solve_sdp(n, Qc, keep_pairs=keep)
        if dval_full is None or dval_red is None:
            continue
        N += 1
        offdiag = n * n - n
        kept_frac.append((int(keep.sum()) - n) / offdiag if offdiag else 1.0)
        denom = max(1.0, abs(pval))
        if abs(dval_full - dval_red) / denom > worst_diff:
            worst_diff = abs(dval_full - dval_red) / denom
            worst_seed = s
        if (pval - dval_red) / denom < 1e-6:
            still_tight += 1
    print(f"--- random sparse: n={n}, density={density}, N={N} ---")
    print(f"    avg fraction of off-diagonal RLT bounds kept: {np.mean(kept_frac):.2f}")
    print(f"    max |dval_full - dval_reduced| / |pval|: {worst_diff:.3e} (seed {worst_seed})")
    print(f"    reduced SDP still tight (gap<1e-6): {still_tight}/{N}")


def run_block(seeds):
    print("=== explicit block Q: only pairs (1,2),(3,4),(5,6) nonzero, n=6 ===")
    worst = 0.0
    worst_seed = None
    tight = 0
    N = 0
    for s in seeds:
        npr.seed(s)
        Q = np.zeros((6, 6))
        np.fill_diagonal(Q, npr.standard_normal(6))
        for (i, j) in [(0, 1), (2, 3), (4, 5)]:
            v = -abs(npr.standard_normal())
            Q[i, j] = Q[j, i] = v
        c = npr.standard_normal((6, 1))
        Qc = np.block([[0.0, c.T], [c, Q]])
        x_opt, pval = solve_qp(6, Q, c)
        if x_opt is None:
            continue
        keep = keep_mask(Q, 6)
        df, _ = solve_sdp(6, Qc)
        dr, _ = solve_sdp(6, Qc, keep_pairs=keep)
        if df is None or dr is None:
            continue
        N += 1
        denom = max(1.0, abs(pval))
        if abs(df - dr) / denom > worst:
            worst = abs(df - dr) / denom
            worst_seed = s
        if (pval - dr) / denom < 1e-6:
            tight += 1
    print(f"    kept {int(keep.sum()) - 6}/30 off-diagonal RLT bounds")
    print(f"    max |dval_full - dval_reduced| / |pval|: {worst:.3e} (seed {worst_seed})")
    print(f"    reduced SDP still tight: {tight}/{N}")


if __name__ == "__main__":
    for dens in (0.2, 0.3, 0.5):
        run_random(8, range(1, 401), dens)
    run_block(range(1, 401))
