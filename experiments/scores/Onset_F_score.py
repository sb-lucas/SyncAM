import numpy as np
import pretty_midi
import mir_eval


def Onset_F_score(audio_onsets, midi_onsets):
    """
    Calculates the F-score between audio onsets and MIDI onsets.
    
    Parameters:
        audio_onsets (np.ndarray): Array of audio onsets.
        midi_onsets (np.ndarray): Array of MIDI onsets.
    """
    # Calculate F-score using mir_eval
    # This handles the matching within the specified tolerance 'window'
    f_measure, precision, recall = mir_eval.onset.f_measure(
        midi_onsets, 
        audio_onsets, 
        window=0.05  # 50 ms tolerance for matching onsets
    )
    
    return f_measure