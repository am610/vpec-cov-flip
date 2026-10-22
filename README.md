# VPEC_COV: peculiar-velocity covariance for SN cosmology in factorized form

Quick start: clone FLIP into `third_party/` (see Setup), then open `notebooks/vpec_cov_flip_woodbury.ipynb`, or run `python scripts/01_vpec_woodbury_demo.py`. Numbers and conclusions are summarized in `FINDINGS.txt`.

Prototype for including the correlated peculiar-velocity covariance cov(vpec)
in SNANA-style SN analyses, using FLIP for the covariance and the factorized
(Woodbury) likelihood from SNANA PR #1797 for the solve.

Status: exploratory, outside SNANA. Nothing here changes SNANA. The integration
into `create_cov` and the forecast project are to be agreed in the PV team and
DESC-TD first.

## Idea

FLIP gives the velocity-velocity covariance between SNe from (RA, DEC, zCMB),
a power spectrum and a cosmology:

    C_vv = fs8^2 * P + sigv^2 * I

P is dense and not low-rank. Two ways to put it into the factorized form
C = D + B B^T (D diagonal, B is N x K):

- mc: draw K vectors x_j ~ N(0, C) and set B = [x_1 ... x_K] / sqrt(K).
  Each column is a systematic shift vector like a calibration term, and
  E[B B^T] = C for any rank of C (proposal by R. Kessler).
- eig: keep the top-K eigenmodes (deterministic, needs an eigendecomposition).
- rsvd: randomized range finder on C (about 4 products C @ [N x K]), which
  approaches eig without the full eigendecomposition.

In all cases the diagonal of what B B^T misses can be added back to D
(remainder correction), which removes most of the log|C| bias.

Velocity is converted to a distance-modulus shift with the low-z factor
5/ln10 * (1+z) / (z (1+z/2)) / c. Since fs8 only rescales P, the columns are
computed once and B(fs8) = fs8 * B_1 is reused at every MCMC step.

## Layout

    notebooks/vpec_cov_flip_woodbury.ipynb   walkthrough with plots (start here)
    src/vpec_cov.py                FITRES reader, FLIP call, B builders, likelihood helpers
    src/factorized_likelihood.py   Woodbury likelihood (copied from the factorized-covariance prototype)
    scripts/01_vpec_woodbury_demo.py   K convergence of chi2, log|C| and the fs8 profile
    scripts/02_rank_vs_N.py            modes needed versus sample size
    scripts/03_diag_correction.py      fs8 bias of eig, rsvd, mc with and without remainder term
    scripts/04_timing_vs_N.py          dense versus factorized cost per likelihood evaluation
    third_party/flip               git clone of corentinravoux/flip (not tracked)
    outputs/                       figure, CSV tables, logs

## Setup

    cd third_party && git clone https://github.com/corentinravoux/flip.git
    pip install -r requirements.txt     # FLIP is imported from third_party, no install

## Run

    python scripts/01_vpec_woodbury_demo.py                  # synthetic sample, ~15 s
    python scripts/01_vpec_woodbury_demo.py --fitres my.FITRES --zmax 0.1
    python scripts/02_rank_vs_N.py --n-list 250 500 1000 2000 3000

FITRES input needs RA, DEC and zCMB (zCMB without the vpec correction, as
requested by the FLIP team). Use spectroscopic redshifts only. BLAS threads
default to 8 so the scripts are safe on a shared login node.

## Results (synthetic sample, z in 0.02 to 0.10, FLIP test power spectrum)

Modes needed for a given fraction of the variance (script 02):

      N    K90   K99   K99.9
    250     88   202    236
   1000    166   617    860
   4000    213  1385   2830
  10000    232  1831   5414

K90 saturates near 230, K99 and K99.9 keep growing but slower than N.

Error on fs8 at N=1500 (script 03, 30 mock residuals, shift in units of the
statistical error, solid = with remainder correction):

    method        K=100    K=200    K=400    mean shift
    eig + rem     0.21     0.11     0.04     none
    rsvd + rem    0.24     0.11     0.07     none
    mc + rem      1.11     0.69     0.52     -1.0, -0.6, -0.4 sigma

- eig and rsvd with the remainder correction give an fs8 error of about 0.1 sigma at
  K=200 and no bias. The log|C| error drops from about -12 to about -0.2 at K=200.
- Random realizations (mc) are biased low in fs8 at these K. The sign and size
  are the same for different seeds, so it is systematic, not sampling noise.

Cost per likelihood evaluation, 8 threads (script 04):

      N      dense      woodbury K=200   cached K=200
   2000     31 ms           2.7 ms          0.30 ms
  10000    2.4 s           13 ms            0.35 ms
  20000     13 s           52 ms            0.27 ms

"cached" precomputes B^T D^-1 B and is valid when D is fixed and fs8 only rescales
B. Below N of about 2000 a dense Cholesky is already fast, so the gain is for large N.

## Open questions

1. Accuracy at N above 5000 with the production power spectrum and the real
   sample geometry, and for w0, wa instead of fs8.
2. Whether mc can be corrected for its bias, or the SNANA implementation should
   use rsvd (needs the dense covariance in memory, 0.8 GB at N=10^4).
3. How D and the remainder term should be handled when sigv is fitted.
4. Impact on the w0, wa FoM for the M25 and K25 samples, using archived outputs.

## Credits

- FLIP is developed by C. Ravoux, B. Carreres and collaborators (MIT license). This repository only calls it and does not include it.
- The random-realization construction of B was proposed by R. Kessler.
- The Woodbury likelihood follows the factorized-covariance implementation in SNANA (PR #1797).

## References

- FLIP: arXiv:2501.16852, github.com/corentinravoux/flip
- Velocity-only likelihood: arXiv:2303.01198
- Factorized covariance: SNANA PR #1797
