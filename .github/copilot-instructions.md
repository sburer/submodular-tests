# Copilot Instructions: Submodular Box QP Project

## Project Overview

Research codebase for solving box-constrained quadratic programs with submodular structure and analyzing SDP relaxation tightness. Core workflow: generate random instances → solve QP with Gurobi → solve SDP relaxation with Mosek → compute primal-dual gaps → visualize distribution.

## Architecture & Data Flow

Three-layer structure with strict separation:
- **[src/define_constants.py](../src/define_constants.py)**: Numerical tolerances (`tol_general`, `tol_rel_gap`, `tol_mosek`) - modify here to adjust precision across all solvers
- **[src/define_functions.py](../src/define_functions.py)**: Core solvers - `generate_random_instance()` builds QP and solves with Gurobi, `build_and_solve_sdp()` handles Mosek SDP formulation
- **[scripts/run_tests.py](../scripts/run_tests.py)**: Experiment runner - orchestrates instance generation, solving, and plotting

**Critical: Scripts add `src/` to `sys.path` dynamically**. Always import from source modules using `from define_functions import ...`, never relative imports.

## Domain-Specific Patterns

### Submodular Structure Enforcement
```python
# Q must have non-positive off-diagonals (enforced in generate_random_instance)
Q[i, j] = -np.abs(Q[i, j])  # i != j
```

### Augmented Matrix Convention
QP parameters stored as `Qc = [[0, c.T], [c, Q]]` (n+1)×(n+1) block matrix. Objective evaluation requires prepending 1: `x_opt_ext = np.vstack([[1.0], x_opt])`.

### Relative Gap Formula
```python
rel_gap = (pval - dval) / max(1.0, |pval + dval| / 2.0)
```
Never allow negative gaps - use absolute value when `|pval - dval| < tol_general`.

### SDP Relaxation Structure
The SDP relaxation lifts the QP into matrix space with variable `Y = [[Y00, x^T], [x, X]]` where:
- `Y00 = 1` (normalization constraint)
- `Y ⪰ 0` (PSD cone constraint)
- **RLT upper bound constraints**: `X[i,j] ≤ x[i]` for all i,j

**RLT rationale**: Since box constraints enforce `0 ≤ x[i] ≤ 1`, valid outer products satisfy `x[i]·x[j] ≤ x[i]` (because `x[j] ≤ 1`). These McCormick envelope constraints tighten the relaxation by cutting away infeasible lifted matrices. Implemented in [define_functions.py](../src/define_functions.py#L118-L120) as:
```python
myexpr = Expr.mul(np.ones((n, 1)), Expr.transpose(x))  # Broadcast x[i] to all j
myexpr = Expr.sub(myexpr, X)  # x[i] - X[i,j] ≥ 0
```

## Running Experiments

**Primary command** (from project root):
```bash
cd scripts && python run_tests.py
```

**Reproducibility**: Default is seeded (`reproducible=True`). Problem sizes randomly sampled from `[n_lower, n_upper]` per seed.

**Custom runs**: Import and call `run_many_instances()` with parameters:
```python
results, config = run_many_instances(n_lower=5, n_upper=25, seeds=range(1,101))
```

**Output**: Saves `results/rel_gap_distribution.png` (log-scale density plot) and prints outliers exceeding `tol_rel_gap`.

## Solver Configuration

### Gurobi (QP Solver)
Tight tolerances set in `generate_random_instance()`:
- `OptimalityTol=1e-8`, `FeasibilityTol=1e-9`
- Always suppress output: `model.setParam('OutputFlag', 0)`

### Mosek (SDP Solver)
Configured in `build_and_solve_sdp()` with `tol_mosek` (1e-10 default):
```python
M.setSolverParam("intpntCoTolPfeas", tol_mosek)
M.setSolverParam("intpntCoTolDfeas", tol_mosek)
M.setSolverParam("intpntCoTolRelGap", tol_mosek)
```
Check `M.getProblemStatus() == 'PrimalAndDualFeasible'` before extracting solutions.

## Code Conventions

- **NumPy print formatting**: Configured for high precision (`%.10f`) in [define_functions.py](../src/define_functions.py#L27-L28)
- **Progress tracking**: Custom `progress_bar()` in [run_tests.py](../scripts/run_tests.py#L47-L52) with terminal carriage returns
- **Seed handling**: Pass `seed=-99` to disable seeding (non-deterministic mode)
- **Results structure**: Dict keyed by seed with `{"rel_gap": float, "n": int}` values

## License Requirements

**Critical**: Both Gurobi and Mosek require valid commercial/academic licenses. Without them, imports will fail immediately. Test with `import gurobipy` and `from mosek.fusion import *` before running experiments.
