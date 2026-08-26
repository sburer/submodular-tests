# Submodular Box QP and SDP Relaxation

Code accompanying *On the Semidefinite Representability of Continuous Quadratic
Submodular Minimization With Applications to Pricing and Moment Problems*.

The paper studies quadratic submodular minimization over the box $[0,1]^n$
(QSMB) and a polynomially sized SDP relaxation of it, proves that the
relaxation is tight for $n \le 3$, and exhibits an explicit $n = 4$ instance on
which it is not. This repository reproduces every computational claim the paper
makes.

## What reproduces what

Each row is an artifact printed in the paper, the script that produces it, and
what that script needs. Run everything from the `scripts/` directory.

| Paper artifact | Script | Needs |
| --- | --- | --- |
| Figure 1 — nested relaxations, submodular boundary | `python tightness_figure.py --outbase ../results/tightness_projection` | Mosek |
| Figure 2 — relative-gap distribution, 1,000 instances | `python run_tests.py` | Gurobi, Mosek |
| Section 5.1 — 34,000 fixed-dimension instances, rank-one finding | `python run_structure_search.py` | Gurobi, Mosek |
| Tables 2 and 3 — three-product pricing | `python run_pricing.py` | Mosek (Gurobi optional) |
| Table 4 — stratified sampling allocation | `python run_allocation.py` | Mosek |
| Figure 3 — historical MATLAB grounded-path reproduction | `python run_laplacian.py` | Mosek |
| Table 5 — Laplacian energy percentage gap *(see Known issues)* | `python run_laplacian.py --energy` | Mosek; writes `results/table5.csv` and `results/table5_raw.csv` |
| Example 4, Proposition 12, Table 6 — the counterexample | `python verify_counterexample.py` | Mosek |
| — its exact half alone, no solver | `python verify_counterexample.py --exact` | none |
| Figure 4 — the counterexample projected | `python gap_figure2.py --outbase ../results/gap_projection2` | cvxpy + Clarabel |
| Conclusions — sparsity remark | `python run_sparsity.py` | Gurobi, Mosek |

`verify_counterexample.py --exact` is worth singling out: the paper's central
new claim, that the relaxation is not tight for $n \ge 4$, is checked in exact
rational arithmetic with no solver and no floating point. It enumerates all
$3^4$ KKT active-set patterns over $\mathbb{Q}$, confirms that the four
singular reduced Hessians have inconsistent stationarity systems, and verifies
the feasible point of Proposition 12 through its five leading principal minors.

## Reproducibility

Every randomized experiment is seeded and deterministic:

- `run_tests.py` — seeds 1..1000, one instance per seed
- `run_structure_search.py` — seeds 1..N at each dimension $n \in \{4,5,6,7\}$
- `run_sparsity.py` — seeded per instance
- `run_laplacian.py --energy` — `numpy.random.default_rng([seed, n, graph])`,
  default `--seed 1`
- `tightness_figure.py`, `gap_figure2.py`, `run_pricing.py`,
  `run_allocation.py`, `verify_counterexample.py` — no randomness at all

## Known issues

**Figure 3 is reproducible with its supplied MATLAB inputs.** Run
`python run_laplacian.py` to use the matrix and
moments constructed by Karthik's `cmmmoments.m`. The moments are the stated
independent-uniform moments, but the path matrix has diagonal 2 at both
endpoints. Equivalently, it is the standard path Laplacian plus unit penalties
on the first and last coordinates:

`xi' L_code xi = sum_edges (xi_i-xi_j)^2 + xi_1^2 + xi_n^2`.

This grounded path explains the archived intercepts P = 2/3 and Q = 1/2 and
the crossing at alpha = 0.2. It is not the standard graph Laplacian currently
described in the paper; the promoted script records the historical computation
explicitly rather than silently presenting the degenerate standard-Laplacian
result.

**Table 5 reproduces qualitatively, not numerically.** The sampling scheme
behind the published table was not recorded, so this repository fixes its own:
$\mu_i \sim U(0,1)$ and $\sigma_i = u_i \sqrt{\mu_i(1-\mu_i)}$ with
$u_i \sim U(0,1)$, which satisfies Lemma 3. With `--seed 1`:

|  | path | star | complete |
| --- | --- | --- | --- |
| $n = 2$ | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| $n = 10$ | 0.658 (0.774) | 1.198 (1.697) | 1.782 (1.056) |
| $n = 20$ | 0.632 (0.591) | 1.406 (1.997) | 2.159 (0.752) |
| $n = 50$ | 0.650 (0.401) | 1.888 (2.034) | 2.314 (0.433) |

Both claims the text draws from the table hold: the gap grows with $n$, and it
is largest for the complete graph, then the star, then the path. The complete
graph agrees closely with the published numbers; the path and star run higher,
which is what one would expect from a different marginal sampling scheme, since
those graphs have far fewer edges to average over.

The default Table 5 run uses `--seed 1` and saves the twelve summary cells to
`results/table5.csv`. The complete seeded observations are saved to
`results/table5_raw.csv`; use `--energy-outbase` to choose another output base.

## Requirements

- Python 3.9+, NumPy, SciPy, Matplotlib
- [Mosek](https://www.mosek.com/) with a licence — all SDPs except `gap_figure2.py`
- [Gurobi](https://www.gurobi.com/) with a licence — global solution of the
  nonconvex box QPs
- [cvxpy](https://www.cvxpy.org/) with Clarabel — `gap_figure2.py` only

Nothing beyond NumPy is needed for `verify_counterexample.py --exact`.

## Project structure

```
src/
  define_constants.py       tolerances
  analysis_functions.py     shared instance generation, solves, and structural analysis
scripts/
  run_tests.py              Figure 2
  run_structure_search.py   Section 5.1 fixed-dimension search
  run_sparsity.py           Conclusions sparsity remark
  run_pricing.py            Tables 2 and 3
  run_allocation.py         Table 4
  run_laplacian.py          Figure 3 reproduction and Table 5
  verify_counterexample.py  Example 4, Proposition 12, Table 6
  counterexample_numerics.py  the Mosek half of the above
  tightness_figure.py       Figure 1
  gap_figure2.py            Figure 4
results/
  tightness_projection.pdf/.png
  rel_gap_distribution.png
  bounds1.csv/.pdf/.png
  table5.csv
  table5_raw.csv
  gap_projection2.pdf/.png
```

## The relative gap

Several scripts report

$$\text{rel gap} = \frac{p^* - d^*}{\max(1, |p^* + d^*|/2)}$$

where $p^*$ is the optimal value of the QP and $d^*$ the dual objective of the
SDP relaxation. The dual value is used because weak duality makes it a rigorous
lower bound, so the relative gap is nonnegative and vanishes exactly when the
relaxation is tight.

## License

MIT. See [LICENSE](LICENSE).
