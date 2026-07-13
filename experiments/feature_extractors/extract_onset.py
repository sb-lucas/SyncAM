import numpy as np
import pretty_midi
import mir_eval
import madmom

def get_onsets_from_midi(midi_file, combine=0.03):
    
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Using a set to automatically handle duplicates across tracks
    onsets = set()
    for instrument in midi_data.instruments:
        # Skip drum tracks if you only want melodic alignment
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            onsets.add(note.start)
            
    onsets_sorted = sorted(list(onsets))
    
    # Merging close onsets (Vectorized-ish approach)
    if not onsets_sorted:
        return np.array([])
        
    merged = [onsets_sorted[0]]
    for onset in onsets_sorted[1:]:
        if onset - merged[-1] > combine:
            merged.append(onset)
            
    return np.array(merged)


def get_madmom_onsets(audio_file, method="cnn", onset_kwargs=None, peak_kwargs=None):
    """
    Extract onset times using madmom's onset processors (CNN, RNN, or Spectral).

    Parameters:
        audio_file (str): Path to the audio file.
        method (str): One of ['cnn', 'rnn', 'spectral'].
        onset_kwargs (dict, optional): Extra parameters for the chosen onset processor.
        peak_kwargs (dict, optional): Extra parameters for OnsetPeakPickingProcessor.

    Returns:
        np.ndarray: Array of detected onset times (in seconds).
    """
    onset_kwargs = onset_kwargs or {}
    peak_kwargs = peak_kwargs or {}

    # --- Default peak picking parameters ---
    default_peak_kwargs = {
        "fps": 100,
        "threshold": 0.4,
        "pre_max": 0.05,
        "post_max": 0.05,
        "combine": 0.03
    }

    # Merge: user kwargs override defaults
    peak_kwargs = {**default_peak_kwargs, **peak_kwargs}

    # --- Choose onset processor ---
    if method.lower() == "cnn":
        onset_proc = madmom.features.onsets.CNNOnsetProcessor()
    elif method.lower() == "rnn":
        onset_proc = madmom.features.onsets.RNNOnsetProcessor()
    elif method.lower() == "spectral":
        onset_proc = madmom.features.onsets.SpectralOnsetProcessor(**onset_kwargs)
    else:
        raise ValueError("method must be one of ['cnn', 'rnn', 'spectral']")

    # --- Compute activation (novelty) ---
    activation = onset_proc(audio_file)

    # --- Peak picking ---
    peak_proc = madmom.features.onsets.OnsetPeakPickingProcessor(**peak_kwargs)
    onsets = peak_proc(activation)

    return onsets
