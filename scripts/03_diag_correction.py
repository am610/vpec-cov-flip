#!/usr/bin/env python
"""
Does adding the diagonal of the dropped modes to D fix the log|C| bias?

For each K, method (eig, rsvd, mc) and correction (off, on), fs8 is fitted on many
mock residuals drawn from the dense C, with the approximate Woodbury
likelihood and with the exact dense Cholesky. Reported per setting:

  mean shift   <fs8_approx - fs8_dense> / sigma_fs8      (bias)
  rms shift    rms of the same quantity                    (scatter of the error)
  dlogdet      log|C_approx| - log|C|  at the true fs8
  sigma ratio  sigma_fs8 approx / dense (curvature at the minimum)

Usage (from the project root):
    python scripts/03_diag_correction.py --n 1500 --n-mock 30
"""

import argparse
import logging
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import scipy.linalg

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import vpec_cov  # noqa: E402
from factorized_likelihood import FactorizedCovariance  # noqa: E402

log = logging.getLogger("diag_correction")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--zmin", type=float, default=0.02)
    ap.add_argument("--zmax", type=float, default=0.10)
    ap.add_argument("--fs8", type=float, default=0.4)
    ap.add_argument("--sigma-mu", type=float, default=0.12)
    ap.add_argument("--k-list", type=int, nargs="+", default=[25, 50, 100, 200, 400, 800])
    ap.add_argument("--n-mock", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)
    n = args.n
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    z = rng.uniform(args.zmin, args.zmax, n)
    F = vpec_cov.vpec_to_dmu(z)
    P = vpec_cov.flip_vv_covariance(ra, dec, z) * np.outer(F, F)
    d = np.full(n, args.sigma_mu**2)
    eig = vpec_cov.eigh_sorted(P)

    L_true = scipy.linalg.cholesky(np.diag(d) + args.fs8**2 * P, lower=True)
    mocks = [L_true @ rng.standard_normal(n) for _ in range(args.n_mock)]
    _, ld_true = np.linalg.slogdet(np.diag(d) + args.fs8**2 * P)

    t0 = time.time()
    dense_fit, dense_sig = [], []
    for r in mocks:
        fun = lambda f, r=r: vpec_cov.m2lnl_dense(f, r, d, P)
        f_hat = vpec_cov.fit_fs8(fun)
        dense_fit.append(f_hat)
        dense_sig.append(vpec_cov.sigma_from_curvature(fun, f_hat))
    dense_fit, dense_sig = np.array(dense_fit), np.array(dense_sig)
    log.info(f"dense fits on {args.n_mock} mocks: {time.time()-t0:.0f} s; fs8_hat = "
             f"{dense_fit.mean():.3f} +- {dense_fit.std():.3f}, sigma_fs8 = {dense_sig.mean():.3f}")

    print(f"\n{'method':>6} {'corr':>5} {'K':>5} {'mean shift/sig':>15} {'rms shift/sig':>14} "
          f"{'sig ratio':>10} {'dlogdet':>9}")
    rows = []
    for method in ("eig", "rsvd", "mc"):
        for k in args.k_list:
            if k > n:
                continue
            if method == "eig":
                B1 = vpec_cov.eigen_B(P, k, eig)
            elif method == "rsvd":
                B1 = vpec_cov.randomized_B(P, k, np.random.default_rng(1000 * k))
            else:
                B1 = vpec_cov.monte_carlo_B(P, k, np.random.default_rng(1000 * k), eig)
            rem1 = vpec_cov.remainder_diag(P, B1)
            for corr in (0, 1):
                rem = rem1 if corr else np.zeros(n)
                shifts, ratios = [], []
                for r, f_d, s_d in zip(mocks, dense_fit, dense_sig):
                    fun = lambda f, r=r: vpec_cov.m2lnl_woodbury(f, r, d, B1, rem)
                    f_hat = vpec_cov.fit_fs8(fun)
                    shifts.append((f_hat - f_d) / s_d)
                    ratios.append(vpec_cov.sigma_from_curvature(fun, f_hat) / s_d)
                fc = FactorizedCovariance(d + args.fs8**2 * rem, args.fs8 * B1)
                dld = fc.logdet() - ld_true
                shifts = np.array(shifts)
                row = (("eig", "rsvd", "mc").index(method), corr, k, shifts.mean(),
                       np.sqrt(np.mean(shifts**2)), float(np.mean(ratios)), dld)
                rows.append(row)
                print(f"{method:>6} {'on' if corr else 'off':>5} {k:5d} {row[3]:15.3f} {row[4]:14.3f} "
                      f"{row[5]:10.3f} {dld:9.2f}", flush=True)

    np.savetxt(os.path.join(args.out, f"diag_correction_N{n}.csv"), rows,
               header="method(0=eig,1=rsvd,2=mc) corr(0=off,1=on) K mean_shift_sig rms_shift_sig sigma_ratio dlogdet",
               fmt="%g")


if __name__ == "__main__":
    main()
