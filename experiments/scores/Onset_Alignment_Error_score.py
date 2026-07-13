import numpy as np
import pretty_midi
import mir_eval
import madmom

import sys
import os
sys.path.append(os.path.abspath(".."))
from feature_extractors.extract_onset import get_onsets_from_midi, get_madmom_onsets

import numpy as np
import mir_eval
from scipy.optimize import linear_sum_assignment

def Onset_Alignment_Error_score(audio_file, midi_file, window=0.03, tau=0.03):
    """
    Computes the Mean Absolute Error between matched onsets and 
    penalizes unmatched events.
    
    Parameters:
        audio_file (str): Path to the audio file.
        midi_file (str): Path to the MIDI file.
        window (float): Max distance to consider a 'match' (e.g., 30ms).
        penalty_seconds (float): Constant error added for every unmatched onset.
        tau (float): Time constant for exponential decay of timing score.
    """
    # 0. Extract onsets
    audio_onsets = get_madmom_onsets(audio_file)
    midi_onsets = get_onsets_from_midi(midi_file)

    if len(audio_onsets) == 0 or len(midi_onsets) == 0:
        return float('inf')

    # 1. Build Cost Matrix (Absolute difference in time)
    # Rows = MIDI, Cols = Audio
    cost_matrix = np.abs(midi_onsets[:, np.newaxis] - audio_onsets)

    # 2. Solve optimal matching (Hungarian Algorithm)
    midi_indices, audio_indices = linear_sum_assignment(cost_matrix)

    # 3. Filter matches by the Tolerance Window
    matches = []
    unmatched_midi = len(midi_onsets)
    unmatched_audio = len(audio_onsets)

    for m_idx, a_idx in zip(midi_indices, audio_indices):
        diff = cost_matrix[m_idx, a_idx]
        if diff <= window:
            matches.append(diff)
            unmatched_midi -= 1
            unmatched_audio -= 1

    # 4. Calculate components
    num_matches = len(matches)
    recall = num_matches / len(audio_onsets)
    
    # Timing Quality (Exponential Decay)
    if num_matches > 0:
        mae = np.mean(matches)
        timing_score = np.exp(-mae / tau)
    else:
        timing_score = 0.0

    # 5. Combine: S = Recall * Timing_Score
    final_score = recall * timing_score

    return final_score