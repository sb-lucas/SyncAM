'''Onset detection methods for audio and MIDI files.'''

import librosa
import numpy as np
import matplotlib.pyplot as plt
import mir_eval
import pretty_midi
import os
import random
import librosa.display
import pandas as pd
from tempfile import TemporaryDirectory
import soundfile as sf
try:
    import madmom
except ImportError:
    print("madmom library not found. Some onset detection methods may not be available.")
    madmom = None  # madmom is optional
    
def get_onsets_from_midi(midi_file, combine=0.03):
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Collect all note onsets (in seconds)
    onsets_ground_truth = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            if note.start not in onsets_ground_truth:
                onsets_ground_truth.append(float(note.start))   # note.start is the onset time in seconds

    # Sort onsets
    onsets_ground_truth = sorted(onsets_ground_truth)


    # Merge onsets that are within `combine` seconds
    merged_onsets = []
    if onsets_ground_truth:
        last_onset = onsets_ground_truth[0]
        merged_onsets.append(last_onset)
        for onset in onsets_ground_truth[1:]:
            # if the current onset is sufficiently far from the last added onset, add it to the list
            if onset - last_onset > combine:
                merged_onsets.append(onset)
                last_onset = onset
            # else: ignore this onset (it’s within the combine window)

    return merged_onsets

def get_onsets_from_midi_by_note(midi_file, combine=0.03):
    """
    Extract onset times from MIDI file, keeping track of individual notes.
    
    Args:
        midi_file: Path to MIDI file or PrettyMIDI object
        combine: Time tolerance for merging onsets (in seconds)
    
    Returns:
        list: List of dictionaries, each containing:
              {'onset': float, 'pitch': int, 'velocity': int, 'duration': float, 'instrument': int}
    """
    # Load MIDI file if it's a path
    if isinstance(midi_file, str):
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    else:
        midi_data = midi_file

    # Collect all notes with their properties
    notes_data = []
    for instrument_idx, instrument in enumerate(midi_data.instruments):
        for note in instrument.notes:
            notes_data.append({
                'onset': float(note.start),
                'pitch': int(note.pitch),
                'velocity': int(note.velocity),
                'duration': float(note.end - note.start),
                'instrument': instrument_idx,
                'program': instrument.program
            })

    # Sort notes by onset time
    notes_data.sort(key=lambda x: x['onset'])
    
    # Optional: merge notes that are very close in time (same as combine parameter)
    if combine > 0:
        merged_notes = []
        if notes_data:
            merged_notes.append(notes_data[0])
            for note in notes_data[1:]:
                # Check if this note should be merged with the last one
                last_note = merged_notes[-1]
                if (note['onset'] - last_note['onset'] <= combine and 
                    note['pitch'] == last_note['pitch']):
                    # Merge with existing note (keep the one with higher velocity)
                    if note['velocity'] > last_note['velocity']:
                        merged_notes[-1] = note
                else:
                    merged_notes.append(note)
        notes_data = merged_notes

    return notes_data

def get_onsets_from_audio_librosa(audio_file='exemplo.wav', hop_length=512):

   #carrega audio
   y, sr = librosa.load(audio_file)

   #extrai espectrograma
   D = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=hop_length*4))

   #-------------------------Novelty function extraction-------------------------#

   #pre-processing. logarithmic compression to account for the logarithmic sensation 
   # of sound intensity and to balance out the dynamic range of the signal
   gamma = 1000  # typical value
   D = np.log1p(gamma * D)  # log(1 + γ·|X|)

   #novelty function a partir de spectral flux
   novelty = librosa.onset.onset_strength(y=None, sr=sr, S=D, lag=1, max_size=1, 
                                          ref=None, detrend=False, center=True, 
                                          feature=None, aggregate=None)

   #post-processing to enhance the peak structure of the novelty function
   M = 3  # averaging window (in frames)
   kernel = np.ones(2*M + 1) / (2*M + 1)
   mu = np.convolve(novelty, kernel, mode='same')
   novelty = np.maximum(0, novelty - mu)

   #-------------------------Peak Picking Part---------------------------#
   #gets energy envelope to get better onset positions
   #energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
   energy = None

   #detecta onsets
   onsets = librosa.onset.onset_detect(onset_envelope=novelty, sr=sr, units='time', backtrack=True, hop_length=hop_length, energy=energy)

   return onsets
    

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

def filter_midi(transcribed_midi, original_midi, time_tolerance=0.05, consider_offset=False, offset_tolerance=0.05):
    """
    Filter transcribed MIDI notes to keep only those that match original MIDI notes within a time tolerance.

    Args:
        transcribed_midi: pretty_midi.PrettyMIDI object of the transcribed MIDI
        original_midi: pretty_midi.PrettyMIDI object of the original MIDI
        time_tolerance: Time tolerance in seconds for matching note onsets
        consider_offset: If True, require the note end (offset) to match within
                         `offset_tolerance` as well. Default: False (we care only about onsets).
        offset_tolerance: Time tolerance in seconds for matching note offsets (end times)
    Returns:
        filtered_midi: pretty_midi.PrettyMIDI object with filtered notes
    """
    filtered_midi = pretty_midi.PrettyMIDI()
    for instrument in transcribed_midi.instruments:
        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum)
        for note in instrument.notes:
            # Check if there is a matching note in the original MIDI within the time tolerance
            # Both pitch and onset timing must match; optionally also check the offset/duration
            note_matched = False
            for orig_instrument in original_midi.instruments:
                if note_matched:
                    break
                for orig_note in orig_instrument.notes:
                    if note.pitch != orig_note.pitch:
                        continue
                    if abs(note.start - orig_note.start) > time_tolerance:
                        continue
                    # If we reach here, onset and pitch match within tolerance
                    if consider_offset:
                        # require offset to match as well
                        if abs(note.end - orig_note.end) <= offset_tolerance:
                            new_instrument.notes.append(note)
                            note_matched = True
                            break
                        else:
                            # mismatch on offset -> do not accept this match
                            continue
                    else:
                        new_instrument.notes.append(note)
                        note_matched = True
                        break
        filtered_midi.instruments.append(new_instrument)
    return filtered_midi

def get_onsets_transcription(audio_file, transcribed_midi_file, original_midi_file=None, time_tolerance=0.05, by_note=False):
    """
    Get onset times from transcribed MIDI file for a given audio file and filter them
    based on the original MIDI file if provided.
    Args:
        audio_file: Path to the audio file
        transcribed_midi_file: Path to the transcribed MIDI file
        original_midi_file: Path to the original MIDI file (optional)
        time_tolerance: Time tolerance in seconds for filtering onsets (default: 0.05)
        by_note: If True, return detailed note information; if False, return combined onsets
    Returns:
        onsets: List of onset times (in seconds) if by_note=False
                List of note dictionaries if by_note=True
    """
    # Load
    transcribed_midi = pretty_midi.PrettyMIDI(transcribed_midi_file)
    if original_midi_file:
        original_midi = pretty_midi.PrettyMIDI(original_midi_file)
        # Filter transcribed MIDI based on original MIDI
        filtered_midi = filter_midi(transcribed_midi, original_midi, time_tolerance=time_tolerance)
    else:
        filtered_midi = transcribed_midi
    
    # Extract onsets from filtered MIDI
    if by_note:
        onsets = get_onsets_from_midi_by_note(filtered_midi, combine=0.03)
    else:
        onsets = get_onsets_from_midi(filtered_midi)
    return onsets