#!/usr/bin/env python
"""
How many eigenmodes of the FLIP velocity covariance are needed as the sample
grows? Reports K for 90, 99 and 99.9% of the variance versus N, and the dense
Cholesky time, for synthetic samples at fixed volume (z in [zmin, zmax]).

Usage (from the project root):
    python scripts/02_rank_vs_N.py --n-list 250 500 1000 2000 3000
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

log = logging.getLogger("rank_vs_N")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-list", type=int, nargs="+", default=[250, 500, 1000, 2000, 3000])
    ap.add_argument("--zmin", type=float, default=0.02)
    ap.add_argument("--zmax", type=float, default=0.10)
    ap.add_argument("--sigma-mu", type=float, default=0.12)
    ap.add_argument("--fs8", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)

    print(f"{'N':>6} {'K90':>6} {'K99':>6} {'K99.9':>6} {'K99/N':>7} {'dense ms':>9}")
    rows = []
    for n in args.n_list:
        ra = rng.uniform(0.0, 360.0, n)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
        z = rng.uniform(args.zmin, args.zmax, n)
        F = vpec_cov.vpec_to_dmu(z)
        P_mu = vpec_cov.flip_vv_covariance(ra, dec, z) * np.outer(F, F)
        ev = vpec_cov.eigen_spectrum(P_mu)
        k = [vpec_cov.n_modes_for_variance(ev, f) for f in (0.90, 0.99, 0.999)]
        C = np.diag(np.full(n, args.sigma_mu**2)) + args.fs8**2 * P_mu
        t0 = time.time()
        scipy.linalg.cholesky(C, lower=True)
        t = time.time() - t0
        rows.append((n, *k, t))
        print(f"{n:6d} {k[0]:6d} {k[1]:6d} {k[2]:6d} {k[1]/n:7.2f} {t*1e3:9.1f}", flush=True)

    np.savetxt(os.path.join(args.out, "rank_vs_N.csv"), rows,
               header="N K90 K99 K999 t_dense_cholesky_s", fmt="%g")


if __name__ == "__main__":
    main()
