"""Run multiple random SDP relaxation instances and collect results.

This script:
  1. Generates a random (Q,c) instance via `generate_random_instance`.
  2. Builds and solves the selected SDP relaxation using `build_and_solve_sdp`.
  3. Prints key diagnostics: relative gap, eigenvalue ratio, objective values, and
	 extracted solution vector approximation.

Requires Mosek (and optionally Gurobi if using that option) to be installed and
licensed in the current Python environment.
"""

import os
import sys
import numpy as np

# Ensure we can import source modules (add `src` folder to path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
	sys.path.append(SRC_PATH)

from define_functions import generate_random_instance, build_and_solve_sdp
import matplotlib.pyplot as plt
import time

def run_many_instances(n: int = 4, seeds = range(1, 11)):
	"""Run multiple random instances and return a results dictionary.

	Each seed generates a new (Q,c), solves SDP, and solves QP to optimality.
	Returns a dict keyed by seed with metrics and solutions.
	"""

	results = {}

	def progress_bar(iteration, total, prefix="Progress", length=40):
		pct = (iteration / total)
		filled = int(length * pct)
		bar = "#" * filled + "-" * (length - filled)
		print(f"\r{prefix}: |{bar}| {iteration}/{total} ({pct*100:5.1f}%)", end="")

	total = len(list(seeds))
	for idx, seed in enumerate(seeds, start=1):
		opts = {
			"q": "submodular",
			"c": "free",
			"psd": "full",
			"rlt": "upper_bounds",
		}

		# Update progress before each run
		progress_bar(idx-1, total, prefix=f"Running seeds {seeds}")
		Qc, opts, x_opt = generate_random_instance(n, opts=opts, seed=seed)

		rel_gap, eval_ratio, YY, ZZ, SS, fixed_values, opts = build_and_solve_sdp(
			n=n, Qc=Qc, opts=opts, fixed_values=None
		)

		# Extract solution approximation (first column of Y)
		y = YY[:, [0]]
		x_est = y[1:].flatten()

		# Compute primal objective value and optimal QP objective
		pval = float((y.T @ Qc @ y)[0, 0])
		Q = Qc[1:, 1:]
		c = Qc[1:, [0]]
		opt_val = float(x_opt.T @ Q @ x_opt + 2.0 * c.T @ x_opt)

		results[int(seed)] = {
			"rel_gap": rel_gap,
			"eval_ratio": float(eval_ratio),
			"pval": pval,
			"opt_val": opt_val,
			"x_est": x_est.tolist(),
			"x_opt": x_opt.flatten().tolist(),
			"opts": opts,
		}

		# Progress update after processing the seed
		progress_bar(idx, total, prefix=f"Running seeds {seeds}")

	print("\nCompleted runs.")
	return results


if __name__ == "__main__":
	results = run_many_instances(n=20, seeds=range(1, 10001))
	# Quick preview of results dict keys
	print("Seeds processed:", sorted(results.keys()))

	# Plot rel_gap vs eval_ratio
	seeds = sorted(results.keys())
	rel_gaps = [results[s]["rel_gap"] for s in seeds]
	eval_ratios = [results[s]["eval_ratio"] for s in seeds]

	plt.figure(figsize=(6,4))
	plt.scatter(eval_ratios, rel_gaps, c='tab:blue', alpha=0.7)
	plt.xlabel("Eigenvalue Ratio (lambda1 / |lambda2|)")
	plt.ylabel("Relative Gap")
	plt.xscale('log')
	plt.yscale('log')
	plt.title(f"Relative Gap vs Eigenvalue Ratio (n={4}, seeds {seeds[0]}..{seeds[-1]})")
	plt.grid(True, linestyle='--', alpha=0.4)
	out_path = os.path.join(PROJECT_ROOT, "results_rel_gap_vs_eval_ratio.png")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	print(f"Saved plot to {out_path}")

