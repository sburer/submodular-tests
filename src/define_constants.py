"""Global numerical tolerance constants used across optimization routines.

These values standardize stopping criteria and numerical checks in algorithms
operating on submodular or convex relaxations. Adjust cautiously; tightening
them may increase runtime, loosening them may degrade solution guarantees.

Definitions:
	tol_general: Baseline absolute tolerance for generic float comparisons.
	tol_rel_gap: Relative optimality gap threshold to declare convergence.
	tol_eval_ratio: Maximum allowed (evaluation / improvement) ratio signaling
		diminishing returns and practical termination.
	tol_mosek: Tolerance tuned for MOSEK solver convergence checks (often
		smaller due to interior-point precision).
"""

# Core tolerances (modify only if numerics require different precision)
tol_general = 1.0e-6    # General absolute tolerance for comparisons
tol_rel_gap = 1.0e-4    # Relative optimality gap to declare problem solved
tol_eval_ratio = 10**4  # Evaluation/improvement ratio threshold for termination
tol_mosek = 1.0e-10     # Solver-specific tolerance for MOSEK
