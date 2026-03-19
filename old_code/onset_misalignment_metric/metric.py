'''Alignment of onsets comparison.'''

import numpy as np

def mean_absolute_onset_error(reference, estimated, window=0.05):
    """
    Compute mean absolute onset error (seconds) for matched onsets.
    reference: list of reference times
    estimated: list of predicted times
    window: maximum allowed distance for a match
    """
    ref = np.array(reference)
    est = np.array(estimated)
    
    used_ref = np.zeros(len(ref), dtype=bool)
    errors = []

    for e in est:
        # find nearest unused reference onset
        diffs = np.abs(ref - e)
        diffs[used_ref] = np.inf
        idx = np.argmin(diffs)
        if diffs[idx] <= window:
            errors.append(diffs[idx])
            used_ref[idx] = True  # avoid double-matching

    if errors:
        return np.mean(errors)
    else:
        return None  # no matches