"""Run multiple random SDP relaxation instances and collect results.

This script:
  1. Generates a random (Q,c) instance via `generate_random_instance`.
  2. Solves the QP to optimality with Gurobi.
  3. Builds and solves the SDP relaxation using `build_and_solve_sdp`.
  4. Computes the primal-dual relative gap for each instance.
  5. Creates a density plot of relative gaps on log scale.
  6. Reports instances with gaps exceeding the threshold.

Requires Mosek and Gurobi to be installed and licensed.
"""

import os
import sys
import numpy as np
import numpy.random as npr

# Ensure we can import source modules (add `src` folder to path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
	sys.path.append(SRC_PATH)

from define_functions import generate_random_instance, build_and_solve_sdp
from define_constants import tol_rel_gap
import matplotlib.pyplot as plt

def run_many_instances(n_lower: int = 4, n_upper: int = 20, seeds = range(1, 1001), reproducible: bool = True):
	"""Run multiple random instances with varying sizes and return results dict.

	For each seed, randomly samples problem size n from [n_lower, n_upper] (inclusive).
	Solves QP to optimality with Gurobi and builds/solves SDP relaxation.
	Returns a dict keyed by seed with metrics and the size used for that instance.

	Args:
	    reproducible: If True (default), uses the seed value for reproducibility.
	                  If False, runs non-deterministically with different randomness each time.
	"""

	results = {}
	config = {"n_lower": n_lower, "n_upper": n_upper}

	def progress_bar(iteration, total, prefix="Progress", length=40):
		pct = (iteration / total)
		filled = int(length * pct)
		bar = "#" * filled + "-" * (length - filled)
		print(f"\r{prefix}: |{bar}| {iteration}/{total} ({pct*100:5.1f}%)", end="")

	total = len(list(seeds))
	for idx, seed in enumerate(seeds, start=1):
		# Update progress before each run
		progress_bar(idx-1, total, prefix=f"Running seeds {seeds}")
		# Set seed BEFORE generating any randomness (if reproducible mode enabled)
		if reproducible:
			npr.seed(seed)
		# Randomly sample problem size
		n = npr.randint(n_lower, n_upper + 1)
		Qc, x_opt = generate_random_instance(n, seed=seed if reproducible else -99)

		rel_gap = build_and_solve_sdp(
			n=n, Qc=Qc, x_opt=x_opt
		)

		results[int(seed)] = {
			"rel_gap": rel_gap,
			"n": n,
		}

		# Progress update after processing the seed
		progress_bar(idx, total, prefix=f"Running seeds {seeds}")

	print("\nCompleted runs.")
	return results, config


if __name__ == "__main__":
	results, config = run_many_instances(n_lower=4, n_upper=20, seeds=range(1, 1001))
	n_lower, n_upper = config["n_lower"], config["n_upper"]
	# Quick preview of results dict keys
	# print("Seeds processed:", sorted(results.keys()))

	# Plot histogram of relative gaps
	seeds = sorted(results.keys())
	rel_gaps = [results[s]["rel_gap"] for s in seeds]
	# Filter to finite values
	rel_gaps_finite = [r for r in rel_gaps if np.isfinite(r)]

	plt.figure(figsize=(8,5))
	if len(rel_gaps_finite) > 0:
		# Create density plot using kernel density estimation
		from scipy import stats
		# Filter out zeros for log scale
		rel_gaps_positive = [r for r in rel_gaps_finite if r > 0]
		if len(rel_gaps_positive) > 1:
			# Use log-transformed data for KDE, then transform back
			log_gaps = np.log10(rel_gaps_positive)
			kde = stats.gaussian_kde(log_gaps)
			x_range = np.linspace(min(log_gaps), max(log_gaps), 500)
			density = kde(x_range)
			x_plot = 10**x_range
			plt.plot(x_plot, density, color='tab:blue', linewidth=2)
			plt.fill_between(x_plot, density, alpha=0.3, color='tab:blue')
			plt.xlabel("Relative Gap")
			plt.ylabel("Density")
			plt.xscale('log')
			plt.title(f"Distribution of Relative Gaps (n sampled from [{n_lower},{n_upper}], seeds {seeds[0]}..{seeds[-1]})")
			plt.grid(True, linestyle='--', alpha=0.4, axis='y')
		else:
			plt.text(0.5, 0.5, "Insufficient positive data for density plot", ha='center', va='center')
			plt.title(f"Distribution of Relative Gaps (n sampled from [{n_lower},{n_upper}], seeds {seeds[0]}..{seeds[-1]}) - Insufficient Data")
	else:
		plt.text(0.5, 0.5, "No finite data to plot", ha='center', va='center')
		plt.title(f"Distribution of Relative Gaps (n sampled from [{n_lower},{n_upper}], seeds {seeds[0]}..{seeds[-1]}) - No Data")
	out_path = os.path.join(PROJECT_ROOT, "results/rel_gap_distribution.png")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	print(f"Saved plot to {out_path}")

	# Identify seeds with large relative gaps
	threshold = tol_rel_gap 
	for s in seeds:
		if results[s]["rel_gap"] > threshold:
			print(f"Seed {s}: gap={results[s]['rel_gap']:.3e}, n={results[s]['n']}")

