"""Systematic tightness search and solution-structure study (paper Section 5.1).

For each fixed dimension n, generates many random submodular instances, solves
the QP to optimality and the SDP relaxation, and records:
  - the relative gap (to search for counterexamples to tightness), and
  - the rank of the optimal SDP solution Y and its active-inequality pattern.

Prints, per dimension: number of instances, number of possible counterexamples
(relative gap > 1e-4), the maximum relative gap, and the rank distribution.

Requires Mosek and Gurobi to be installed and licensed.
"""
import os
import sys
import time
from collections import Counter

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from analysis_functions import analyze, GAP_TOL

# Dimension -> number of instances (seeds 1..count). Adjust as desired.
PLAN = {4: 20000, 5: 8000, 6: 4000, 7: 2000}


def run(plan=PLAN):
    for n, count in plan.items():
        t0 = time.time()
        worst = -np.inf
        worst_seed = None
        counterexamples = 0
        N = 0
        rank_ctr = Counter()
        for s in range(1, count + 1):
            res = analyze(n, s)
            if res is None:
                continue
            N += 1
            rank_ctr[res["r"]] += 1
            if res["rel_gap"] > worst:
                worst = res["rel_gap"]
                worst_seed = s
            if res["rel_gap"] > GAP_TOL:
                counterexamples += 1
                print(f"  POSSIBLE COUNTEREXAMPLE n={n} seed={s} "
                      f"gap={res['rel_gap']:.3e} r={res['r']} "
                      f"l={res['l']} d={res['d']} u={res['u']}", flush=True)
        dt = time.time() - t0
        print(f"n={n}: {N} instances, max rel_gap={worst:.3e} (seed {worst_seed}), "
              f"counterexamples={counterexamples}, rank_dist={dict(rank_ctr)} "
              f"({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    run()
