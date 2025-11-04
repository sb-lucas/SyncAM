'''Onset detection methods for audio and MIDI files.'''

import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import mir_eval
import pretty_midi
import os
import random
import librosa.display
import pandas as pd
from tempfile import TemporaryDirectory
import pretty_midi
import madmom
import soundfile as sf

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

def get_onsets_transcription(audio_file):
    #midi = transcribre(audio_file)
    #onsets = get_midi_onsets()
    
    
    onsets = [] # Placeholder

    return onsets