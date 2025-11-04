'''Python script to measure onset detection error for different instruments using various methods.'''


import librosa
import madmom
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
import soundfile as sf
from midi_helpers import trim_random_midi_segment, render_midi_to_audio, INSTRUMENTS, synthesize_dataset
from onset_detection import get_onsets_from_midi, get_onsets_from_audio_librosa, get_madmom_onsets, get_onsets_transcription
from metric import mean_absolute_onset_error
import argparse

def main(method_name, **kwargs):

    errors = [0] * len(INSTRUMENTS)
    midi_folder = '../maestro_midi'

    #--------------------------------Evaluate onset detection-------------------------#
    for i, midi_file in enumerate(os.listdir(midi_folder)):
        print(f"Processing {i+1}/{len(os.listdir(midi_folder))}: {midi_file}")

        midi_onsets = get_onsets_from_midi(f'{midi_folder}/{midi_file}', combine=0.03)

        for j, instrument in enumerate(INSTRUMENTS.keys()):
            instrument_id = instrument.replace(" ", "_").lower()
            audio_file = f"../sintetizados/{instrument_id}/{midi_file.replace('.midi', '.wav')}"

            onsets = get_onsets(audio_file=audio_file, **kwargs)

            mean_abs_error = mean_absolute_onset_error(midi_onsets, onsets, window=0.05)
            errors[j] += mean_abs_error

    errors = [e / len(os.listdir(midi_folder)) for e in errors]

    #---------------------------Save results to CSV---------------------------#
    instrument_names = list(INSTRUMENTS.keys())
    df_row = pd.DataFrame([errors], columns=instrument_names, index=[method_name])

    csv_path = "onset_detection_results.csv"

    if os.path.exists(csv_path):
        # Append mode, no header
        df_row.to_csv(csv_path, mode='a', header=False)
    else:
        # First time: include header
        df_row.to_csv(csv_path, mode='w', header=True)

if __name__ == "__main__":
    #specify onset detection method
    parser = argparse.ArgumentParser(description="Estimate measure error for different instruments.")
    parser.add_argument('--method', type=str, default='cnn', choices=['cnn', 'rnn', 'spectral', 'librosa', 'transcription'],
                        help='Onset detection method to use: "cnn" for Madmom CNN, "librosa" for Librosa onset detection.')

    args = parser.parse_args()
     
    if args.method == 'librosa':
        get_onsets = get_onsets_from_audio_librosa
        kwargs = {'hop_length': 512}
    elif args.method == 'cnn':
        get_onsets = get_madmom_onsets
        kwargs = {
            "method": "cnn",
        }
    elif args.method == 'rnn':
        get_onsets = get_madmom_onsets
        kwargs = {
            "method": "rnn",
        }
    elif args.method == 'spectral':
        get_onsets = get_madmom_onsets
        kwargs = {
            "method": "spectral",
            "onset_kwargs": dict(onset_method='superflux', fps=100),
        }
    elif args.method == 'transcription':
        get_onsets = get_onsets_transcription
        kwargs = {}

    main(args.method, **kwargs)


    