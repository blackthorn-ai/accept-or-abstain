"""Group-held-out splits (protocol section 5)."""
from __future__ import annotations

import numpy as np


def random_part(idx: np.ndarray, rng: np.random.Generator, fracs=(0.70, 0.15, 0.15)):
    """Split an index array at random into fit F, random calibration C_r and random test R."""
    idx = rng.permutation(idx)
    n_f = int(round(fracs[0] * len(idx)))
    n_c = int(round(fracs[1] * len(idx)))
    return idx[:n_f], idx[n_f : n_f + n_c], idx[n_f + n_c :]


def group_holdout(groups: np.ndarray, rng: np.random.Generator, frac_test=0.15, frac_cal=0.15):
    """Assign whole groups to held-out test H, held-out calibration C_h and the rest I.

    Groups are shuffled and taken in order until each share of the molecules is reached.
    """
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    n = len(groups)
    H_g, C_g, I_g = [], [], []
    n_h = n_c = 0
    for j in order:
        if n_h < frac_test * n:
            H_g.append(uniq[j]); n_h += counts[j]
        elif n_c < frac_cal * n:
            C_g.append(uniq[j]); n_c += counts[j]
        else:
            I_g.append(uniq[j])
    H = np.flatnonzero(np.isin(groups, H_g))
    C = np.flatnonzero(np.isin(groups, C_g))
    I = np.flatnonzero(np.isin(groups, I_g))
    return H, C, I


def named_holdout(groups: np.ndarray, rng: np.random.Generator, n_test: int, n_cal: int):
    """Hold out a given number of named groups (e.g. ligands) for test and for calibration."""
    uniq = np.unique(groups)
    order = rng.permutation(uniq)
    H_g, C_g = order[:n_test], order[n_test : n_test + n_cal]
    H = np.flatnonzero(np.isin(groups, H_g))
    C = np.flatnonzero(np.isin(groups, C_g))
    I = np.flatnonzero(~np.isin(groups, np.r_[H_g, C_g]))
    return H, C, I, list(H_g), list(C_g)
