# Submodular Box QP and SDP Relaxation

This project solves random instances of quadratic programming (QP) problems over the box $[0,1]^n$ and computes SDP relaxations to analyze the primal-dual gap.

## Overview

The main script `scripts/run_tests.py` performs the following:

1. **Generates random instances**: Creates random (Q, c) pairs with submodular structure
2. **Solves QP to optimality**: Uses Gurobi to find the exact optimal solution
3. **Solves SDP relaxation**: Builds and solves the SDP relaxation using Mosek
4. **Computes relative gaps**: Calculates primal-dual relative gaps for each instance
5. **Visualizes results**: Creates a log-scale density plot of gap distribution
6. **Reports outliers**: Prints instances with gaps exceeding the tolerance threshold

## Requirements

- Python 3.7+
- [Gurobi](https://www.gurobi.com/) (with valid license)
- [Mosek](https://www.mosek.com/) (with valid license)
- NumPy
- SciPy
- Matplotlib

## Project Structure

```
src/
  define_constants.py    # Problem tolerances and constants
  define_functions.py    # Core functions for QP and SDP solving
scripts/
  run_tests.py          # Main script for running experiments
results/
  rel_gap_distribution.png  # Output plot (generated on run)
```

## Usage

### Basic Usage (Reproducible)

Run with default settings (reproducible results):

```bash
cd scripts
python run_tests.py
```

This runs 1000 seeds (1-1000) with problem sizes sampled from [4, 20].

### Non-Reproducible Runs

To run with different random instances each time (non-deterministic):

```python
from run_tests import run_many_instances

results, config = run_many_instances(reproducible=False)
```

### Custom Configuration

Adjust problem parameters:

```python
from run_tests import run_many_instances

results, config = run_many_instances(
    n_lower=5,           # Minimum problem size
    n_upper=25,          # Maximum problem size
    seeds=range(1, 101), # Run 100 instances
    reproducible=True    # Use seeded randomness
)
```

## Example Output

### Command-Line Output

When you run the script, you'll see progress tracking and final results:

```
Running seeds range(1, 101): |########################################| 100/100 (100.0%)
Completed runs.
Saved plot to /path/to/project/results/rel_gap_distribution.png
```

### Output Image

The script generates `results/rel_gap_distribution.png`, a density plot showing:

- **X-axis**: Relative gap (log scale)
- **Y-axis**: Density (kernel density estimation)
- **Title**: Shows the range of problem sizes (n) and seed range tested

Example plot characteristics:
- Most gaps cluster near zero (good SDP bounds)
- Log scale reveals the full distribution from very small to moderate gaps
- Blue shaded area under the density curve

## Key Functions

### `run_many_instances(n_lower, n_upper, seeds, reproducible)`

Main entry point for running the experiment.

**Parameters:**
- `n_lower` (int): Minimum problem size (default: 4)
- `n_upper` (int): Maximum problem size (default: 20)
- `seeds` (iterable): Seed values to iterate (default: range(1, 1001))
- `reproducible` (bool): Use seeded randomness (default: True)

**Returns:**
- `results` (dict): Keyed by seed with 'rel_gap' and 'n' values
- `config` (dict): Configuration used (n_lower, n_upper)

### `generate_random_instance(n, seed)`

Generates a random QP instance with submodular structure.

**Returns:**
- `Qc`: (n+1)×(n+1) augmented matrix [[0, c'], [c, Q]]
- `x_opt`: Optimal solution from Gurobi

### `build_and_solve_sdp(n, Qc, x_opt)`

Builds and solves the SDP relaxation.

**Returns:**
- `rel_gap`: Primal-dual relative gap

## Understanding the Gap

The **relative gap** measures how tight the SDP bound is:

$$\text{rel\_gap} = \frac{p_{\text{val}} - d_{\text{val}}}{\max(1, |p_{\text{val}} + d_{\text{val}}|/2)}$$

where:
- $p_{\text{val}}$ = objective value at optimal QP solution
- $d_{\text{val}}$ = dual objective value of SDP relaxation

A gap close to 0 means the SDP provides a very tight bound; a larger gap indicates the relaxation is looser.

## Configuration

Modify [src/define_constants.py](src/define_constants.py) to adjust solver tolerances and gap thresholds.

## License

See LICENSE file (if applicable).
