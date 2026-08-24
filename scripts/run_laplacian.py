"""Quadratic energy of Laplacian matrices (Section 5.4, Figure 3 and Table 5).

Two experiments.

(1) --subquantile   Figure 3.  Worst-case (1-alpha)-subquantile of the energy
    E(xi) = xi' L xi for a path graph on 50 vertices, with mu and Sigma the
    first two moments of the independent uniform vector on [0,1]^n, bounded
    using ambiguity set P via SDP (12) and ambiguity set Q via SDP (13).

    For fixed threshold x the inner problem is
        S(x) = sup_P E_P[ max(0, x - E(xi)) ],
    a two-piece maximum with A_1 = 0, b_1 = 0, c_1 = 0 and A_2 = -L, b_2 = 0,
    c_2 = x.  Then -A_k is submodular and A_k is negative semidefinite for both
    k, so both SDPs apply.  The subquantile is sup_x [ x - S(x)/(1-alpha) ],
    obtained by minimizing over x jointly with the SDP variables.
    Deterministic.

(2) --energy        Table 5.  Minimum expected energy e* under marginal moment
    information, via (17) with A = -L, against the lower bound ebar* obtained
    by bounding each edge term with the bivariate result of Proposition 8 and
    summing.  Reports mean (sd) of 100 * (e* - ebar*) / e* over 100 random
    instances for path, star and complete graphs at n = 2, 10, 20, 50.

    mu and sigma are drawn to satisfy Lemma 3, 0 <= sigma_i <= sqrt(mu_i(1-mu_i)):
        mu_i    ~ U(0,1)
        sigma_i = u_i * sqrt(mu_i (1 - mu_i)),  u_i ~ U(0,1)
    Seeded; --seed changes the stream.

Requires Mosek.

Run:  python run_laplacian.py --subquantile --outbase ../results/bounds1
      python run_laplacian.py --energy
"""

import argparse
import numpy as np
import mosek.fusion as mf
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix


# --------------------------------------------------------------------------
# graphs
# --------------------------------------------------------------------------

def laplacian(kind, n):
    A = np.zeros((n, n))
    if kind == "path":
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
    elif kind == "star":
        for i in range(1, n):
            A[0, i] = A[i, 0] = 1.0
    elif kind == "complete":
        A = np.ones((n, n)) - np.eye(n)
    else:
        raise ValueError(kind)
    return np.diag(A.sum(1)) - A


def edges(kind, n):
    L = laplacian(kind, n)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if L[i, j] != 0]


# --------------------------------------------------------------------------
# (1) worst-case subquantile  -- Figure 3
# --------------------------------------------------------------------------

def subquantile(L, mu, Sigma, alpha, ambiguity):
    """sup_x [ x - S(x)/(1-alpha) ] with S from ambiguity set P or Q."""
    n = L.shape[0]
    Ak = [np.zeros((n, n)), -L]          # A_1 = 0, A_2 = -L
    with Model() as M:
        y0 = M.variable("y0", 1, Domain.unbounded())
        y = M.variable("y", n, Domain.unbounded())
        xth = M.variable("x", 1, Domain.unbounded())     # the threshold
        if ambiguity == "P":
            Y = M.variable("Y", Domain.unbounded([n, n]))
            M.constraint(Expr.sub(Y, Y.transpose()), Domain.equalsTo(0.0).withShape([n, n]))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        M.constraint(Y.index(i, j), Domain.lessThan(0.0))
        else:
            Y = M.variable("Y", Domain.inPSDCone(n))

        for k in (0, 1):
            ck = Expr.constTerm(0.0) if k == 0 else xth.index(0)
            if ambiguity == "P":
                Z = M.variable(f"Z{k}", Domain.greaterThan(0.0, [n, n]))
                topleft = Expr.sub(y0.index(0), ck)
                offdiag = Expr.mul(0.5, Expr.sub(y, Expr.mul(Z.transpose(), np.ones(n))))
                block = Expr.sub(Expr.add(Y, Expr.mul(0.5, Expr.add(Z, Z.transpose()))),
                                 Matrix.dense(Ak[k]))
            else:
                z = M.variable(f"z{k}", n, Domain.greaterThan(0.0))
                w = M.variable(f"w{k}", n, Domain.greaterThan(0.0))
                topleft = Expr.sub(Expr.sub(y0.index(0), Expr.sum(z)), ck)
                offdiag = Expr.mul(0.5, Expr.add(Expr.sub(y, w), z))
                block = Expr.sub(Y, Matrix.dense(Ak[k]))
            S = Expr.vstack(
                Expr.hstack(Expr.reshape(topleft, [1, 1]), Expr.reshape(offdiag, [1, n])),
                Expr.hstack(Expr.reshape(offdiag, [n, 1]), block))
            M.constraint(S, Domain.inPSDCone(n + 1))

        sdpobj = Expr.add(Expr.add(y0.index(0), Expr.dot(mu, y)),
                          Expr.dot(Matrix.dense(Sigma), Y))
        M.objective(ObjectiveSense.Minimize,
                    Expr.sub(Expr.mul(1.0 / (1.0 - alpha), sdpobj), xth.index(0)))
        M.solve()
        return -M.primalObjValue()


def run_subquantile(nvert, alphas, outbase):
    L = laplacian("path", nvert)
    mu = np.full(nvert, 0.5)
    Sigma = 0.25 * np.ones((nvert, nvert)) + (1.0 / 3 - 0.25) * np.eye(nvert)
    print(f"path graph on {nvert} vertices; mu_i = 1/2, Sigma_ii = 1/3, Sigma_ij = 1/4")
    print(f"\n{'alpha':>8}{'bound from Q':>16}{'bound from P':>16}{'stronger':>12}")
    rows = []
    for a in alphas:
        vQ = subquantile(L, mu, Sigma, a, "Q")
        vP = subquantile(L, mu, Sigma, a, "P")
        rows.append((a, vQ, vP))
        print(f"{a:>8.2f}{vQ:>16.5f}{vP:>16.5f}{('P' if vP > vQ else 'Q'):>12}")
    if outbase:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        A = np.array(rows)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(A[:, 0], A[:, 1], "o-", label=r"ambiguity set $\mathcal{Q}$")
        ax.plot(A[:, 0], A[:, 2], "s-", label=r"ambiguity set $\mathcal{P}$")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"worst-case $(1-\alpha)$-subquantile")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"{outbase}.{ext}", dpi=200)
        print(f"\nwrote {outbase}.pdf and {outbase}.png")
    return rows


# --------------------------------------------------------------------------
# (2) minimum expected energy  -- Table 5
# --------------------------------------------------------------------------

def energy_sdp(L, mu, sigma):
    """e* via (17) with A = -L: e* = -max{ (-L).Sigma : ... }."""
    n = L.shape[0]
    diag_target = mu ** 2 + sigma ** 2
    with Model() as M:
        S = M.variable("S", Domain.unbounded([n, n]))
        M.constraint(Expr.sub(S, S.transpose()), Domain.equalsTo(0.0).withShape([n, n]))
        for i in range(n):
            M.constraint(S.index(i, i), Domain.equalsTo(float(diag_target[i])))
            for j in range(n):
                M.constraint(Expr.sub(S.index(i, j), float(mu[j])), Domain.lessThan(0.0))
        Y = Expr.vstack(
            Expr.hstack(Expr.constTerm([[1.0]]), Expr.reshape(Expr.constTerm(mu), [1, n])),
            Expr.hstack(Expr.reshape(Expr.constTerm(mu), [n, 1]), S))
        M.constraint(Y, Domain.inPSDCone(n + 1))
        M.objective(ObjectiveSense.Maximize, Expr.dot(Matrix.dense(-L), S))
        M.solve()
        return -M.primalObjValue()


def energy_bivariate(kind, n, mu, sigma):
    """ebar*: sum over edges of the bivariate bound of Proposition 8."""
    tot = 0.0
    for (i, j) in edges(kind, n):
        best = min(mu[i], mu[j], mu[i] * mu[j] + sigma[i] * sigma[j])
        tot += (mu[i] ** 2 + sigma[i] ** 2) + (mu[j] ** 2 + sigma[j] ** 2) - 2 * best
    return tot


def run_energy(sizes, kinds, reps, seed):
    print(f"{reps} random instances per cell, seed {seed}")
    print("mu_i ~ U(0,1),  sigma_i = u_i sqrt(mu_i(1-mu_i)),  u_i ~ U(0,1)")
    print(f"\n{'':>8}" + "".join(f"{k:>22}" for k in kinds))
    table = {}
    for n in sizes:
        cells = []
        for kind in kinds:
            L = laplacian(kind, n)
            gaps = []
            rng = np.random.default_rng([seed, n, abs(hash(kind)) % (2**31)])
            for _ in range(reps):
                mu = rng.uniform(0.0, 1.0, n)
                sigma = rng.uniform(0.0, 1.0, n) * np.sqrt(mu * (1.0 - mu))
                e = energy_sdp(L, mu, sigma)
                eb = energy_bivariate(kind, n, mu, sigma)
                if e > 1e-12:
                    # ebar* is a valid lower bound on e*, so the gap is >= 0;
                    # clamp solver noise at n = 2, where the two agree exactly.
                    gaps.append(max(0.0, 100.0 * (e - eb) / e))
            g = np.array(gaps)
            cells.append(f"{g.mean():.3f} ({g.std(ddof=0):.3f})")
            table[(n, kind)] = (g.mean(), g.std(ddof=0))
        print(f"n = {n:<4}" + "".join(f"{c:>22}" for c in cells))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subquantile", action="store_true", help="Figure 3")
    ap.add_argument("--energy", action="store_true", help="Table 5")
    ap.add_argument("--nvert", type=int, default=50)
    ap.add_argument("--outbase", default=None)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 10, 20, 50])
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    args = ap.parse_args()
    if not (args.subquantile or args.energy):
        args.subquantile = args.energy = True

    if args.subquantile:
        print("=" * 72)
        print("Section 5.4, Figure 3 -- worst-case (1-alpha)-subquantile")
        print("=" * 72)
        alphas = args.alphas if args.alphas else [round(0.05 * k, 2) for k in range(0, 19)]
        run_subquantile(args.nvert, alphas, args.outbase)

    if args.energy:
        print("\n" + "=" * 72)
        print("Section 5.4, Table 5 -- percentage gap 100 (e* - ebar*) / e*")
        print("=" * 72)
        run_energy(args.sizes, ["path", "star", "complete"], args.reps, args.seed)


if __name__ == "__main__":
    main()
