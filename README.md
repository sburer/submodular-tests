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
| Figure 3 — worst-case subquantile *(see Known issues)* | `python run_laplacian.py --subquantile` | Mosek |
| Table 5 — Laplacian energy percentage gap *(see Known issues)* | `python run_laplacian.py --energy` | Mosek |
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

**Figure 3 does not currently reproduce.** `run_laplacian.py --subquantile`
returns a worst-case subquantile of 0 for both ambiguity sets at every
$\alpha$, rather than the two crossing curves the paper plots. The reason
appears to be structural rather than a coding error. The Laplacian satisfies
$Le = 0$, so constant vectors have zero energy, and with $\mu,\Sigma$ set to
the first two moments of the independent uniform vector on $[0,1]^n$ both
relaxed ambiguity sets admit a distribution whose energy vanishes identically:

- under $\mathcal{Q}$, the point mass at $\xi = e/2$, since
  $\Sigma - \tfrac14 ee^T = \tfrac1{12} I \succeq 0$;
- under $\mathcal{P}$, the perfectly correlated $\xi = t\,e$ with
  $\mathbb{E}[t] = 1/2$ and $\mathbb{E}[t^2] = 1/3$, which meets the diagonal
  moments exactly and has $\mathbb{E}[\xi_i \xi_j] = 1/3 \ge 1/4 = \Sigma_{ij}$.

Both are excluded by $\mathcal{R}$, which fixes $\mathbb{E}[\xi\xi^T] = \Sigma$
exactly, but readmitted by the relaxations. Either the figure used different
data than the text describes, or it plots a different quantity. Unresolved.

**Table 5 reproduces qualitatively, not numerically.** The sampling scheme
behind the published table was not recorded, so this repository fixes its own:
$\mu_i \sim U(0,1)$ and $\sigma_i = u_i \sqrt{\mu_i(1-\mu_i)}$ with
$u_i \sim U(0,1)$, which satisfies Lemma 3. With `--seed 1`:

|  | path | star | complete |
| --- | --- | --- | --- |
| $n = 2$ | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| $n = 10$ | 0.599 (0.993) | 1.381 (2.508) | 1.771 (1.176) |
| $n = 20$ | 0.619 (0.595) | 1.752 (2.282) | 2.162 (0.757) |
| $n = 50$ | 0.649 (0.339) | 2.010 (2.198) | 2.328 (0.503) |

Both claims the text draws from the table hold: the gap grows with $n$, and it
is largest for the complete graph, then the star, then the path. The complete
graph agrees closely with the published numbers; the path and star run higher,
which is what one would expect from a different marginal sampling scheme, since
those graphs have far fewer edges to average over.

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
  define_functions.py       instance generation, QP and SDP solves
  analysis_functions.py     extended solve returning Y; rank and active-set analysis
scripts/
  run_tests.py              Figure 2
  run_structure_search.py   Section 5.1 fixed-dimension search
  run_sparsity.py           Conclusions sparsity remark
  run_pricing.py            Tables 2 and 3
  run_allocation.py         Table 4
  run_laplacian.py          Figure 3 and Table 5
  verify_counterexample.py  Example 4, Proposition 12, Table 6
  counterexample_numerics.py  the Mosek half of the above
  tightness_figure.py       Figure 1
  gap_figure2.py            Figure 4
results/
  tightness_projection.pdf/.png
  rel_gap_distribution.png
  bounds1.png
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
