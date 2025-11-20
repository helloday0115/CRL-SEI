"""weibull_calibrator.py
Functions to fit Weibull to class-wise tail distances (e.g., distances from class prototype),
compute survival scores and derive distance thresholds for a target quantile.

Dependencies: scipy (scipy.stats.weibull_min)

Example workflow:
  - After training: extract embeddings for validation samples and compute distances to class centers.
  - For each class, call fit_weibull_tail(distances, tail_size=20)
  - At inference, compute distance d of a test sample to predicted class center, compute SF = weibull_min.sf(d, c, loc, scale).
  - Decide unknown/known by comparing to quantile threshold (q): accept if SF >= 1-q (or compare distance threshold).

Usage:
  from weibull_calibrator import fit_weibull_tail, weibull_score, compute_threshold_from_quantile
"""
import numpy as np
from scipy.stats import weibull_min

def fit_weibull_tail(distances, tail_size=20, force_loc0=True):
    d = np.sort(np.asarray(distances).ravel())[::-1]  # descending
    tail = d[:tail_size]
    # MLE fit (force loc=0 often improves stability)
    if force_loc0:
        c, loc, scale = weibull_min.fit(tail, floc=0)
    else:
        c, loc, scale = weibull_min.fit(tail)
    return c, loc, scale

def weibull_score(distance, c, loc, scale):
    # survival function (1 - CDF)
    return weibull_min.sf(distance, c, loc=loc, scale=scale)

def compute_distance_threshold_for_quantile(c, loc, scale, q):
    # returns distance threshold such that CDF(threshold) = q
    return weibull_min.ppf(q, c, loc=loc, scale=scale)

if __name__ == '__main__':
    # demo synthetic distances
    rng = np.random.RandomState(0)
    distances = np.abs(rng.normal(loc=1.5, scale=0.3, size=1000))
    c, loc, scale = fit_weibull_tail(distances, tail_size=50)
    print('weibull params', c, loc, scale)
    print('thresh @ 0.95', compute_distance_threshold_for_quantile(c, loc, scale, 0.95))
