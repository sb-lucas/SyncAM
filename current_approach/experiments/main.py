import pretty_midi
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import random
from evaluator import Evaluator
import ot  # Python Optimal Transport library

from midi_helpers import render_midi_to_audio
#from onset_detection import get_onsets_from_midi, get_madmom_onsets

from synctoolbox.dtw.core import compute_warping_path
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_CENS
from synctoolbox.feature.utils import estimate_tuning

def extract_audio_chroma(audio_path, sr=22050, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr)

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    chroma = chroma.T  # (frames, 12)

    # Normalize each frame
    chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1e-6)

    return chroma

def extract_midi_chroma(midi_path, fs=22050/512):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    piano_roll = midi_data.get_piano_roll(fs=fs)

    chroma = np.zeros((12, piano_roll.shape[1]))

    for note in range(128):
        chroma[note % 12] += piano_roll[note]

    chroma = chroma.T  # (frames, 12)

    chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1e-6)

    return chroma

def OT_score(audio_path, midi_path, reg_m=1.0):
    global contador
    contador += 1
    print(f"Calculating OT score for pair {contador}")

    audio_features = extract_audio_chroma(audio_path)
    midi_features = extract_midi_chroma(midi_path)

    C = ot.dist(midi_features, audio_features, metric='sqeuclidean')

    n, m = len(midi_features), len(audio_features)

    p = np.ones(n) / n
    q = np.ones(m) / m

    distance = ot.unbalanced.sinkhorn_unbalanced2(
        p, q, C,
        reg=0.1,
        reg_m=reg_m
    )

    max_dist = 2 * reg_m

    similarity = 1 - (distance / max_dist)

    return np.clip(similarity, 0, 1)

def DTW_score(audio_path, midi_path):
    global contador
    contador += 1
    print(f"Calculating DTW score for pair {contador}")

    audio_features = extract_audio_chroma(audio_path)
    midi_features = extract_midi_chroma(midi_path)

    C = 1 - np.dot(audio_features, midi_features.T)

    D, _, wp = compute_warping_path(C=C)

    raw_distance = D[-1, -1]

    L = wp.shape[1]

    normalized_dist = raw_distance / L

    return 1 - np.clip(normalized_dist, 0, 1)

def main():
    #SÃO 2200 runs pro smd. o score_calculator roda 2200 vezes.
    global contador
    contador = 0
    get_score = DTW_score  # Change to DTW_score to use the DTW-based metric

    eval = Evaluator('/home/labic/bombril/raia/smd', get_score)
    metrics = eval.evaluate()   
    metrics.to_csv(f'{get_score.__name__}_results.csv', index=False)

if __name__ == "__main__":
    main()