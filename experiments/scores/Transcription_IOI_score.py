import numpy as np
import pretty_midi
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_CENS
from synctoolbox.dtw.core import compute_warping_path
from scipy.optimize import linear_sum_assignment
import librosa
from scipy.ndimage import gaussian_filter1d


def get_ioi_notes_from_midi(midi_path):
    """
    Extracts [onset, pitch, IOI] from a MIDI file.
    IOI is the time since the last onset of the SAME pitch.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Group by pitch first to calculate isolated IOIs
    pitch_dict = {}
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                if note.pitch not in pitch_dict:
                    pitch_dict[note.pitch] = []
                pitch_dict[note.pitch].append(note.start)
                
    notes = []
    for pitch, onsets in pitch_dict.items():
        # Sort chronologically just in case
        onsets = sorted(onsets)
        for i, onset in enumerate(onsets):
            # Calculate Inter-Onset Interval
            # If it's the first time we see this pitch, IOI is 0
            ioi = onset - onsets[i-1] if i > 0 else 0.0
            notes.append([onset, pitch, ioi])
            
    # Sort the final array chronologically
    notes = sorted(notes, key=lambda x: x[0])
    return np.array(notes)

def Transcription_IOI_score(audio_transcribed_midi_path, midi_path, tau=0.05):
    #Computes a 2D alignment score between two MIDI files using Pitch and Time.
    
    ref_notes = get_ioi_notes_from_midi(midi_path)
    est_notes = get_ioi_notes_from_midi(audio_transcribed_midi_path)

    if len(ref_notes) == 0 or len(est_notes) == 0:
        return 0.0

    ref_onsets, ref_pitches, ref_iois = ref_notes[:, 0], ref_notes[:, 1], ref_notes[:, 2]
    est_onsets, est_pitches, est_iois = est_notes[:, 0], est_notes[:, 1], est_notes[:, 2]

    # Build the 3D Cost Matrix
    onset_diffs = np.abs(ref_onsets[:, np.newaxis] - est_onsets)
    ioi_diffs = np.abs(ref_iois[:, np.newaxis] - est_iois)
    
    # Total error is the absolute time shift PLUS the rhythmic distortion
    W_onset = 0.5
    W_ioi = 0.5
    total_time_diffs = (W_onset * onset_diffs) + (W_ioi * ioi_diffs)
    
    pitch_matches = ref_pitches[:, np.newaxis] == est_pitches
    LARGE_PENALTY = 1e6
    cost_matrix = np.where(pitch_matches, total_time_diffs, LARGE_PENALTY)

    # Hungarian Algorithm
    ref_indices, trans_indices = linear_sum_assignment(cost_matrix)

    # Filter Pitch Mismatches
    valid_costs = []
    for r_idx, t_idx in zip(ref_indices, trans_indices):
        cost = cost_matrix[r_idx, t_idx]
        
        #to avoid cases where the hungarian algorithm forced a match between non-matching pitches
        if cost < (LARGE_PENALTY / 2): 
            valid_costs.append(cost)

    if len(valid_costs) == 0:
        return 0.0

    # Continuous Scoring
    avg_time_errors = np.array(valid_costs)

    #this turns each note into 1 or 0, but with a smooth transition for notes that are close to the threshold, instead of a hard cutoff.
    timing_weights = np.exp(-avg_time_errors / tau) 
    
    #therefore, the sum is a smooth counter of how many notes were correctly matched in terms of both pitch and timing, 
    #with a gradual penalty for those that are close but not perfect matches.
    continuous_match_count = np.sum(timing_weights)

    precision = continuous_match_count / len(est_notes)
    recall = continuous_match_count / len(ref_notes)
    
    if precision + recall == 0:
        return 0.0
        
    final_score = (2 * precision * recall) / (precision + recall)

    return final_score
