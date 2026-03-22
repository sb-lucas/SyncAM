import numpy as np
import librosa
import pretty_midi

from synctoolbox.dtw.core import compute_warping_path
from scipy.spatial.distance import cdist

# -----------------------------
# DTW alignment + score
# -----------------------------
def Calculate_DTW_CENS(cens_audio, cens_midi):

    # shape: (12, T) → transpose to (T, 12)
    X = cens_audio.T
    Y = cens_midi.T

    # cosine distance matrix
    C = cdist(X, Y, metric="cosine")

    # DTW
    D, E, wp = compute_warping_path(C)

    # total cost is bottom-right of accumulated cost matrix
    total_cost = D[-1, -1]

    # normalize by path length
    path_length = wp.shape[1]  # wp has shape (2, L)
    avg_cost = total_cost / path_length # in [0,2]

    # convert to similarity
    avg_sim = 1 - avg_cost      # now in [-1, 1]
    score = (avg_sim + 1) / 2   # map to [0, 1]

    # clamp to [0,1] -> just to be safe
    score = np.clip(score, 0.0, 1.0)

    return score

# -----------------------------
# Full pipeline
# -----------------------------
def DTW_CENS_Score(audio_path, midi_path):
    root_dir = audio_path.split('/')[0:-2]
    root_dir = '/'.join(root_dir)

    audio_folder_name = audio_path.split('/')[-2] + '_cens'
    midi_folder_name = midi_path.split('/')[-2] + '_cens'

    midi_file_name = os.path.basename(midi_path).replace('.mid', '.npy').replace('.midi', '.npy')
    audio_file_name = os.path.basename(audio_path).replace('.wav', '.npy')

    audio_cens_path = os.path.join(root_dir, audio_folder_name, audio_file_name)
    midi_cens_path = os.path.join(root_dir, midi_folder_name, midi_file_name)

    cens_audio = np.load(audio_cens_path)
    cens_midi = np.load(midi_cens_path)

    score = Calculate_DTW_CENS(cens_audio, cens_midi)

    return score

def 