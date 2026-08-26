"""Run the grounded-path subquantile and Laplacian-energy experiments.

The subquantile experiment uses independent-uniform moments and a path matrix
with diagonal 2 at both endpoints. This is the standard path Laplacian plus
unit boundary penalties (a grounded path), not the singular graph Laplacian.

Run from this directory:

    python run_laplacian.py
    python run_laplacian.py --energy

The default output is ``../results/bounds1.{csv,png,pdf}``.
Mosek is required.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix


def subquantile(L: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                alpha: float, ambiguity: str) -> float:
    """Solve the P/Q moment-dual subquantile SDP for a quadratic matrix L."""
    n = L.shape[0]
    Ak = [np.zeros((n, n)), -L]
    with Model() as model:
        y0 = model.variable("y0", 1, Domain.unbounded())
        y = model.variable("y", n, Domain.unbounded())
        threshold = model.variable("threshold", 1, Domain.unbounded())
        if ambiguity == "P":
            Y = model.variable("Y", Domain.unbounded([n, n]))
            model.constraint(
                Expr.sub(Y, Y.transpose()),
                Domain.equalsTo(0.0).withShape([n, n]),
            )
            for i in range(n):
                for j in range(n):
                    if i != j:
                        model.constraint(Y.index(i, j), Domain.lessThan(0.0))
        elif ambiguity == "Q":
            Y = model.variable("Y", Domain.inPSDCone(n))
        else:
            raise ValueError(ambiguity)

        for k in (0, 1):
            ck = Expr.constTerm(0.0) if k == 0 else threshold.index(0)
            if ambiguity == "P":
                Z = model.variable(f"Z{k}", Domain.greaterThan(0.0, [n, n]))
                topleft = Expr.sub(y0.index(0), ck)
                offdiag = Expr.mul(0.5, Expr.sub(y, Expr.mul(Z.transpose(), np.ones(n))))
                block = Expr.sub(
                    Expr.add(Y, Expr.mul(0.5, Expr.add(Z, Z.transpose()))),
                    Matrix.dense(Ak[k]),
                )
            else:
                z = model.variable(f"z{k}", n, Domain.greaterThan(0.0))
                w = model.variable(f"w{k}", n, Domain.greaterThan(0.0))
                topleft = Expr.sub(Expr.sub(y0.index(0), Expr.sum(z)), ck)
                offdiag = Expr.mul(0.5, Expr.add(Expr.sub(y, w), z))
                block = Expr.sub(Y, Matrix.dense(Ak[k]))
            sdp_block = Expr.vstack(
                Expr.hstack(Expr.reshape(topleft, [1, 1]), Expr.reshape(offdiag, [1, n])),
                Expr.hstack(Expr.reshape(offdiag, [n, 1]), block),
            )
            model.constraint(sdp_block, Domain.inPSDCone(n + 1))

        sdpobj = Expr.add(
            Expr.add(y0.index(0), Expr.dot(mu, y)),
            Expr.dot(Matrix.dense(Sigma), Y),
        )
        model.objective(
            ObjectiveSense.Minimize,
            Expr.sub(Expr.mul(1.0 / (1.0 - alpha), sdpobj), threshold.index(0)),
        )
        model.solve()
        return -model.primalObjValue()


def graph_laplacian(kind: str, n: int) -> np.ndarray:
    """Return an unweighted path, star, or complete graph Laplacian."""
    adjacency = np.zeros((n, n))
    if kind == "path":
        for i in range(n - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    elif kind == "star":
        for i in range(1, n):
            adjacency[0, i] = adjacency[i, 0] = 1.0
    elif kind == "complete":
        adjacency = np.ones((n, n)) - np.eye(n)
    else:
        raise ValueError(kind)
    return np.diag(adjacency.sum(1)) - adjacency


def graph_edges(kind: str, n: int) -> list[tuple[int, int]]:
    L = graph_laplacian(kind, n)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if L[i, j] != 0]


def energy_sdp(L: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the full SDP bound e* for the marginal-moment experiment."""
    n = L.shape[0]
    diag_target = mu**2 + sigma**2
    with Model() as model:
        S = model.variable("S", Domain.unbounded([n, n]))
        model.constraint(Expr.sub(S, S.transpose()), Domain.equalsTo(0.0).withShape([n, n]))
        for i in range(n):
            model.constraint(S.index(i, i), Domain.equalsTo(float(diag_target[i])))
            for j in range(n):
                model.constraint(Expr.sub(S.index(i, j), float(mu[j])), Domain.lessThan(0.0))
        Y = Expr.vstack(
            Expr.hstack(Expr.constTerm([[1.0]]), Expr.reshape(Expr.constTerm(mu), [1, n])),
            Expr.hstack(Expr.reshape(Expr.constTerm(mu), [n, 1]), S),
        )
        model.constraint(Y, Domain.inPSDCone(n + 1))
        model.objective(ObjectiveSense.Maximize, Expr.dot(Matrix.dense(-L), S))
        model.solve()
        return -model.primalObjValue()


def energy_bivariate(kind: str, n: int, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the sum of independent bivariate edge lower bounds."""
    total = 0.0
    for i, j in graph_edges(kind, n):
        best = min(mu[i], mu[j], mu[i] * mu[j] + sigma[i] * sigma[j])
        total += (mu[i] ** 2 + sigma[i] ** 2) + (mu[j] ** 2 + sigma[j] ** 2) - 2 * best
    return total


def grounded_path_laplacian(n: int) -> np.ndarray:
    """Return the path Laplacian with unit penalties at both endpoints."""
    L = graph_laplacian("path", n)
    L[0, 0] += 1.0
    L[-1, -1] += 1.0
    return L


def subquantile_inputs(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the grounded path matrix and independent-uniform moments."""
    L = grounded_path_laplacian(n)
    mu = np.full(n, 0.5)
    Sigma = 0.25 * np.ones((n, n)) + (1.0 / 12.0) * np.eye(n)
    return L, mu, Sigma


def run(alphas: list[float], n: int) -> list[dict]:
    L, mu, Sigma = subquantile_inputs(n)
    rows = []
    for alpha in alphas:
        for ambiguity in ("P", "Q"):
            raw_value = float(subquantile(L, mu, Sigma, alpha, ambiguity))
            value = 0.0 if abs(raw_value) < 1.0e-6 else raw_value
            rows.append(
                {
                    "alpha": float(alpha),
                    "ambiguity": ambiguity,
                    "value": value,
                    "raw_value": raw_value,
                }
            )
        p_value = rows[-2]["value"]
        q_value = rows[-1]["value"]
        print(f"alpha={alpha:.2f}  P={p_value:.9f}  Q={q_value:.9f}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {
        ambiguity: sorted(
            (row for row in rows if row["ambiguity"] == ambiguity),
            key=lambda row: row["alpha"],
        )
        for ambiguity in ("P", "Q")
    }
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(
        [row["alpha"] for row in values["P"]],
        [row["value"] for row in values["P"]],
        ".-",
        color="#0072BD",
        linewidth=0.8,
        markersize=2.2,
        label="P",
    )
    ax.plot(
        [row["alpha"] for row in values["Q"]],
        [row["value"] for row in values["Q"]],
        "+-",
        color="#D95319",
        linewidth=0.8,
        markersize=4.0,
        markeredgewidth=0.7,
        label="Q",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.7)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.set_yticks(np.linspace(0.0, 0.7, 8))
    ax.set_xlabel("α")
    ax.set_ylabel("Lower bound on worst-case (1-α)-subquantile")
    ax.legend(loc="upper right")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def run_energy(sizes: list[int], reps: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Run Table 5 and return summary rows plus reproducible raw rows."""
    kinds = ("path", "star", "complete")
    print(f"{reps} random instances per cell, seed {seed}")
    print("mu_i ~ U(0,1),  sigma_i = u_i sqrt(mu_i(1-mu_i)),  u_i ~ U(0,1)")
    print(f"\n{'':>8}" + "".join(f"{kind:>22}" for kind in kinds))
    summary = []
    raw = []
    for n in sizes:
        cells = []
        for kind in kinds:
            L = graph_laplacian(kind, n)
            kind_seed = {"path": 1, "star": 2, "complete": 3}[kind]
            rng = np.random.default_rng([seed, n, kind_seed])
            gaps = []
            for rep in range(1, reps + 1):
                mu = rng.uniform(0.0, 1.0, n)
                sigma = rng.uniform(0.0, 1.0, n) * np.sqrt(mu * (1.0 - mu))
                e = energy_sdp(L, mu, sigma)
                eb = energy_bivariate(kind, n, mu, sigma)
                gap = max(0.0, 100.0 * (e - eb) / e) if e > 1.0e-12 else 0.0
                gaps.append(gap)
                raw.append({"seed": seed, "n": n, "graph": kind, "rep": rep,
                            "full_bound": e, "edge_bound": eb, "gap_pct": gap})
            g = np.asarray(gaps)
            cells.append(f"{g.mean():.3f} ({g.std(ddof=0):.3f})")
            summary.append({"seed": seed, "n": n, "graph": kind, "reps": reps,
                            "mean_gap_pct": g.mean(), "std_gap_pct": g.std(ddof=0)})
        print(f"n = {n:<4}" + "".join(f"{cell:>22}" for cell in cells))
    return summary, raw


def write_energy_csv(outbase: Path, summary: list[dict], raw: list[dict]) -> None:
    """Save the publication table and all seeded observations."""
    outputs = ((outbase.with_suffix(".csv"), summary),
               (outbase.parent / f"{outbase.name}_raw.csv", raw))
    for path, rows in outputs:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--energy", action="store_true", help="run the Table 5 energy experiment")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 10, 20, 50])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--energy-outbase", type=Path, default=Path("../results/table5"))
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument(
        "--outbase",
        type=Path,
        default=Path("../results/bounds1"),
    )
    args = parser.parse_args()
    if args.energy:
        args.energy_outbase.parent.mkdir(parents=True, exist_ok=True)
        summary, raw = run_energy(args.sizes, args.reps, args.seed)
        write_energy_csv(args.energy_outbase, summary, raw)
        print(f"wrote {args.energy_outbase}.csv and {args.energy_outbase}_raw.csv")
        return
    alphas = (
        args.alphas
        if args.alphas is not None
        else [round(0.01 * k, 2) for k in range(100)]
    )
    rows = run(alphas, args.n)
    args.outbase.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.outbase.with_suffix(".csv"), rows)
    write_plot(args.outbase, rows)
    print(f"wrote {args.outbase}.csv, .png, and .pdf")


if __name__ == "__main__":
    main()
