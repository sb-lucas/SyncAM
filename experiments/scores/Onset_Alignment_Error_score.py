import numpy as np
import pretty_midi
import mir_eval
import sys
import os
from scipy.optimize import linear_sum_assignment

def Onset_Alignment_Error_score(audio_features, midi_features, window=0.05, tau=0.05):
    """
    Computes the Mean Absolute Error between matched onsets and 
    penalizes unmatched events.
    
    Parameters:
        audio_file (str): Path to the audio file.
        midi_file (str): Path to the MIDI file.
        window (float): Max distance to consider a 'match' (e.g., 50ms).
        penalty_seconds (float): Constant error added for every unmatched onset.
        tau (float): Time constant for exponential decay of timing score.
    """

    # 0. Extract onsets
    audio_onsets = audio_features
    midi_onsets = midi_features

    if len(audio_onsets) == 0 or len(midi_onsets) == 0:
        return float('inf')

    # 1. Build Cost Matrix (Absolute difference in time)
    # Rows = MIDI, Cols = Audio
    cost_matrix = np.abs(midi_onsets[:, np.newaxis] - audio_onsets)

    # 2. Solve optimal matching (Hungarian Algorithm)
    midi_indices, audio_indices = linear_sum_assignment(cost_matrix)

    # 3. Filter matches by the Tolerance Window
    matches = []

    for m_idx, a_idx in zip(midi_indices, audio_indices):
        diff = cost_matrix[m_idx, a_idx]
        if diff <= window:
            matches.append(diff)

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