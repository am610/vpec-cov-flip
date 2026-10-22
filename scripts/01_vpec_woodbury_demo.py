#!/usr/bin/env python
"""
Example: peculiar-velocity covariance from FLIP in the factorized form,
compared against a dense Cholesky solve.

    C = D + fs8^2 * F P F^T,   D = diag(sigma_mu^2),   F = diag(dmu/dvpec)

where P is the FLIP velocity-velocity covariance at fs8 = 1. P is not
low-rank, so it is replaced by K columns, C ~ D + B B^T, either by

  mc    K random realizations / sqrt(K) (Kessler's proposal, any rank of P)
  eig   the top-K eigenmodes (deterministic, needs an eigendecomposition)

and the likelihood is evaluated with the Woodbury identity (an N x N Cholesky
becomes a K x K one). The script reports how K controls the error in chi2 and
log|C|, and in the fs8 likelihood profile.

Usage (from the project root):
    python scripts/01_vpec_woodbury_demo.py                      # synthetic sample
    python scripts/01_vpec_woodbury_demo.py --n 3000 --zmax 0.12
    python scripts/01_vpec_woodbury_demo.py --fitres my.FITRES --zmax 0.1
"""

import argparse
import logging
import os
import sys
import time

# Cap BLAS threads before numpy loads: shared login nodes expose hundreds of cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import scipy.linalg

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import vpec_cov  # noqa: E402
from factorized_likelihood import FactorizedCovariance  # noqa: E402

log = logging.getLogger("vpec_demo")


def dense_loglike_terms(C, r):
    """chi2 = r^T C^-1 r and log|C| from a dense Cholesky of C."""
    L = scipy.linalg.cholesky(C, lower=True)
    y = scipy.linalg.solve_triangular(L, r, lower=True)
    return float(y @ y), float(2.0 * np.sum(np.log(np.diag(L))))


def load_sample(args, rng):
    if args.fitres:
        t = vpec_cov.read_fitres(args.fitres, ("RA", "DEC", "zCMB"))
        keep = (t["zCMB"] > args.zmin) & (t["zCMB"] < args.zmax)
        ra, dec, z = t["RA"][keep], t["DEC"][keep], t["zCMB"][keep]
        log.info(f"{args.fitres}: {keep.sum()} of {len(keep)} SNe with {args.zmin} < zCMB < {args.zmax}")
    else:
        ra = rng.uniform(0.0, 360.0, args.n)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, args.n)))
        z = rng.uniform(args.zmin, args.zmax, args.n)
        log.info(f"synthetic sample: {args.n} SNe, uniform on the sky, {args.zmin} < z < {args.zmax}")
    return ra, dec, z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fitres", help="FITRES file with RA, DEC, zCMB (default: synthetic sample)")
    ap.add_argument("--n", type=int, default=1500, help="synthetic sample size")
    ap.add_argument("--zmin", type=float, default=0.02)
    ap.add_argument("--zmax", type=float, default=0.10)
    ap.add_argument("--fs8", type=float, default=0.4, help="true fs8 for the mock residual")
    ap.add_argument("--sigma-mu", type=float, default=0.12, help="diagonal sigma_mu (stat + intrinsic)")
    ap.add_argument("--k-list", type=int, nargs="+", default=[5, 10, 25, 50, 100, 200, 400, 800])
    ap.add_argument("--n-seeds", type=int, default=20, help="MC repeats per K")
    ap.add_argument("--k-profile", type=int, default=200, help="K for the fs8 profile panel")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ra, dec, z = load_sample(args, rng)
    n = len(z)

    t0 = time.time()
    P = vpec_cov.flip_vv_covariance(ra, dec, z)
    t_flip = time.time() - t0
    F = vpec_cov.vpec_to_dmu(z)
    P_mu = P * np.outer(F, F)  # per unit fs8^2, in mag^2
    log.info(f"FLIP covariance for N={n}: {t_flip:.1f} s")

    ev = vpec_cov.eigen_spectrum(P_mu)
    modes = {f: vpec_cov.n_modes_for_variance(ev, f) for f in (0.90, 0.99, 0.999)}
    log.info("eigenmodes holding 90/99/99.9%% of the variance: %d / %d / %d (N=%d)"
             % (modes[0.90], modes[0.99], modes[0.999], n))

    # Reference: dense C at the true fs8 and a mock residual drawn from it.
    d = np.full(n, args.sigma_mu**2)
    C_true = np.diag(d) + args.fs8**2 * P_mu
    r = scipy.linalg.cholesky(C_true, lower=True) @ rng.standard_normal(n)

    t0 = time.time()
    chi2_ref, ld_ref = dense_loglike_terms(C_true, r)
    t_dense = time.time() - t0
    log.info(f"dense Cholesky of C ({n} x {n}): {t_dense*1e3:.1f} ms, chi2={chi2_ref:.2f}, logdet={ld_ref:.2f}")

    eig = vpec_cov.eigh_sorted(P_mu)

    # Factors at fs8 = 1; B(fs8) = fs8 * B_1 reuses the same columns at every MCMC step.
    rows = []
    for k in args.k_list:
        if k > n:
            continue
        B_eig = args.fs8 * vpec_cov.eigen_B(P_mu, k, eig)
        t0 = time.time()
        fc = FactorizedCovariance(d, B_eig)
        chi2, _, _ = fc.loglike(r)
        t_w = time.time() - t0
        rows.append(("eig", k, 0.0, chi2 / chi2_ref - 1.0, fc.logdet() - ld_ref, t_w))

        errs_chi2, errs_ld = [], []
        for s in range(args.n_seeds):
            B_mc = args.fs8 * vpec_cov.monte_carlo_B(P_mu, k, np.random.default_rng(1000 * k + s), eig)
            fc = FactorizedCovariance(d, B_mc)
            chi2, _, _ = fc.loglike(r)
            errs_chi2.append(chi2 / chi2_ref - 1.0)
            errs_ld.append(fc.logdet() - ld_ref)
        rows.append(("mc", k, float(np.std(errs_chi2)), float(np.mean(errs_chi2)), float(np.mean(errs_ld)), t_w))

    print()
    print(f"{'method':>6} {'K':>5} {'chi2 rel.err':>13} {'(MC scatter)':>13} {'dlogdet':>10} {'Woodbury ms':>12}")
    for m, k, sd, e_chi2, e_ld, t_w in rows:
        print(f"{m:>6} {k:5d} {e_chi2:13.2e} {sd:13.2e} {e_ld:10.3f} {t_w*1e3:12.1f}")
    print(f"\ndense Cholesky reference: {t_dense*1e3:.1f} ms\n")

    # fs8 likelihood profile: dense versus Woodbury with the cached B_1 columns.
    fs8_grid = np.linspace(0.05, 0.9, 35)
    k_prof = min(args.k_profile, n)
    prof_dense = np.array([sum(dense_loglike_terms(np.diag(d) + f**2 * P_mu, r)) for f in fs8_grid])
    B1_eig = vpec_cov.eigen_B(P_mu, k_prof, eig)
    B1_mc = vpec_cov.monte_carlo_B(P_mu, k_prof, np.random.default_rng(args.seed + 7), eig)
    prof = {}
    for name, B1 in (("eig", B1_eig), ("mc", B1_mc)):
        vals = []
        for f in fs8_grid:
            fc = FactorizedCovariance(d, f * B1)
            vals.append(fc.quadratic_form(r) + fc.logdet())
        prof[name] = np.array(vals)
    best = fs8_grid[np.argmin(prof_dense)]
    log.info(f"fs8 minimum on the grid: dense {best:.3f}, eig(K={k_prof}) "
             f"{fs8_grid[np.argmin(prof['eig'])]:.3f}, mc(K={k_prof}) {fs8_grid[np.argmin(prof['mc'])]:.3f} "
             f"(true {args.fs8})")

    np.savetxt(os.path.join(args.out, "k_convergence.csv"),
               [(0 if m == "eig" else 1, k, sd, e1, e2, t) for m, k, sd, e1, e2, t in rows],
               header="method(0=eig,1=mc) K chi2_relerr_scatter chi2_relerr dlogdet t_woodbury_s", fmt="%g")

    make_figure(args, n, ev, rows, fs8_grid, prof_dense, prof, k_prof)


def make_figure(args, n, ev, rows, fs8_grid, prof_dense, prof, k_prof):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 16, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    pos = np.clip(ev, 0, None)
    ax[0].semilogx(np.arange(1, n + 1), np.cumsum(pos) / pos.sum(), color="k")
    ax[0].set_xlabel("number of eigenmodes K")
    ax[0].set_ylabel("cumulative variance fraction")
    ax[0].text(0.05, 0.9, "(a)", transform=ax[0].transAxes)

    for m, label, style in (("eig", "top-K eigenmodes", "o-"), ("mc", "random realizations", "s-")):
        sel = [(k, e, sd) for mm, k, sd, e, _, _ in rows if mm == m]
        ks = [s[0] for s in sel]
        err = [abs(s[1]) if m == "eig" else np.hypot(abs(s[1]), s[2]) for s in sel]
        ax[1].loglog(ks, err, style, label=label)
    ax[1].set_xlabel("number of columns K")
    ax[1].set_ylabel(r"relative error of $\chi^2$")
    ax[1].legend(fontsize=13, frameon=False)
    ax[1].text(0.05, 0.1, "(b)", transform=ax[1].transAxes)

    ax[2].plot(fs8_grid, prof_dense - prof_dense.min(), "k-", label="dense Cholesky")
    for name, label, style in (("eig", "eigenmodes", "--"), ("mc", "realizations", ":")):
        ax[2].plot(fs8_grid, prof[name] - prof[name].min(), style, label=f"{label}, K={k_prof}")
    ax[2].axvline(args.fs8, color="gray", lw=0.8)
    ax[2].set_xlabel(r"$f\sigma_8$")
    ax[2].set_ylabel(r"$-2\Delta\ln\mathcal{L}$")
    ax[2].set_ylim(-1, 25)
    ax[2].legend(fontsize=13, frameon=False)
    ax[2].text(0.05, 0.9, "(c)", transform=ax[2].transAxes)

    fig.tight_layout()
    path = os.path.join(args.out, "vpec_woodbury_demo.pdf")
    fig.savefig(path)
    log.info(f"wrote {path}")


if __name__ == "__main__":
    main()
