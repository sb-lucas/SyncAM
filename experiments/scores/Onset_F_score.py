import numpy as np
import pretty_midi
import mir_eval
import madmom
from feature_extractors.extract_onset import get_onsets_from_midi, get_madmom_onsets


def Onset_F_score(audio_file, midi_file):
    """
    Calculates the F-score between audio onsets and MIDI onsets.
    
    Parameters:
        audio_file (str): Path to audio.
        midi_file (str): Path to MIDI.
    """
    # 1. Extract onsets
    audio_onsets = get_madmom_onsets(audio_file)
    midi_onsets = get_onsets_from_midi(midi_file)
    
    # 2. Calculate F-score using mir_eval
    # This handles the matching within the specified tolerance 'window'
    f_measure, precision, recall = mir_eval.onset.f_measure(
        midi_onsets, 
        audio_onsets, 
        window=0.03  # 30 ms tolerance for matching onsets
    )
    
    return f_measure