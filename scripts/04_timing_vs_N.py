#!/usr/bin/env python
"""
Cost of one likelihood evaluation versus sample size N for fixed K.

  dense      Cholesky of the N x N matrix C = D + B B^T
  woodbury   FactorizedCovariance rebuilt for every call (cost N K^2 + K^3),
             needed when D changes between calls (sigv or a fs8-dependent
             remainder term)
  cached     B^T D^-1 B and B^T D^-1 r computed once; only the K x K matrix
             I + fs8^2 G is factorized per call, so the cost does not grow with N
             (valid when D is fixed and fs8 only rescales B)

Only array shapes matter for the timing, so B is random.

Usage (from the project root):
    python scripts/04_timing_vs_N.py --n-list 1000 2000 5000 10000 20000 --k-list 200 400
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

from factorized_likelihood import FactorizedCovariance  # noqa: E402

log = logging.getLogger("timing")


def best_of(fun, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fun()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-list", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000])
    ap.add_argument("--k-list", type=int, nargs="+", default=[200, 400])
    ap.add_argument("--max-dense", type=int, default=20000, help="skip the dense solve above this N")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(1)
    rows = []
    print(f"{'N':>6} {'K':>5} {'dense ms':>10} {'woodbury ms':>12} {'cached ms':>10} {'speedup (cached)':>17}")
    for n in args.n_list:
        d = np.full(n, 0.12**2)
        r = rng.standard_normal(n) * 0.2
        for k in args.k_list:
            B = rng.standard_normal((n, k)) * 0.02
            repeats = 3 if n <= 5000 else 1

            if n <= args.max_dense:
                C = np.diag(d) + B @ B.T
                t_dense = best_of(lambda: scipy.linalg.cholesky(C, lower=True), repeats)
                del C
            else:
                t_dense = np.nan

            def woodbury():
                fc = FactorizedCovariance(d, B)
                fc.loglike(r)

            t_w = best_of(woodbury, repeats)

            G = B.T @ (B / d[:, None])
            q = B.T @ (r / d)
            rr = float(r @ (r / d))
            logdet_d = float(np.sum(np.log(d)))

            def cached(f=0.4):
                M = np.eye(k) + f**2 * G
                Lc = scipy.linalg.cholesky(M, lower=True)
                y = scipy.linalg.solve_triangular(Lc, f * q, lower=True)
                return rr - y @ y + logdet_d + 2.0 * np.sum(np.log(np.diag(Lc)))

            t_c = best_of(cached, 20)
            rows.append((n, k, t_dense, t_w, t_c))
            print(f"{n:6d} {k:5d} {t_dense*1e3:10.1f} {t_w*1e3:12.1f} {t_c*1e3:10.2f} {t_dense/t_c:17.0f}", flush=True)

    np.savetxt(os.path.join(args.out, "timing_vs_N.csv"), rows,
               header="N K t_dense_s t_woodbury_s t_cached_s", fmt="%g")


if __name__ == "__main__":
    main()
