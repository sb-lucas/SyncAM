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

def note_based_onset_error(reference_notes, estimated_notes, window=0.05, pitch_tolerance=0):
    """
    Compute onset error for individual notes, matching by pitch and onset time.
    
    Args:
        reference_notes: list of note dictionaries with 'onset' and 'pitch' keys
        estimated_notes: list of note dictionaries with 'onset' and 'pitch' keys  
        window: maximum time distance for onset matching (seconds)
        pitch_tolerance: maximum pitch difference for matching (semitones)
    
    Returns:
        dict containing:
            - 'note_errors': list of individual note errors
            - 'mean_error': mean of all note errors
            - 'matched_count': number of matched notes
            - 'total_reference': total number of reference notes
            - 'total_estimated': total number of estimated notes
    """
    ref_notes = reference_notes if isinstance(reference_notes, list) else reference_notes
    est_notes = estimated_notes if isinstance(estimated_notes, list) else estimated_notes
    
    used_ref = np.zeros(len(ref_notes), dtype=bool)
    note_errors = []
    matched_pairs = []

    for i, est_note in enumerate(est_notes):
        best_match_idx = -1
        best_error = float('inf')
        
        for j, ref_note in enumerate(ref_notes):
            if used_ref[j]:
                continue
                
            # Check pitch matching
            pitch_diff = abs(est_note['pitch'] - ref_note['pitch'])
            if pitch_diff > pitch_tolerance:
                continue
                
            # Check onset time matching
            onset_error = abs(est_note['onset'] - ref_note['onset'])
            if onset_error <= window and onset_error < best_error:
                best_match_idx = j
                best_error = onset_error
        
        if best_match_idx >= 0:
            note_errors.append(best_error)
            used_ref[best_match_idx] = True
            matched_pairs.append((i, best_match_idx, best_error))
    
    result = {
        'note_errors': note_errors,
        'mean_error': np.mean(note_errors) if note_errors else None,
        'matched_count': len(note_errors),
        'total_reference': len(ref_notes),
        'total_estimated': len(est_notes),
        'matched_pairs': matched_pairs
    }
    
    return result