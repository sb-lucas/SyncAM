import numpy as np
import librosa
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_CENS
from synctoolbox.dtw.core import compute_warping_path
import pretty_midi

def extract_cens_audio(audio_path, sr=22050, hop_length=2048):
    y, sr = librosa.load(audio_path, sr=sr)

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    cens = librosa.feature.chroma_cens(
        C=chroma,
        hop_length=hop_length
    )

    return cens

def DTW_Standard_CENS_score(audio_path, midi_path):
    # ==========================================
    # 1. AUDIO FEATURE EXTRACTION
    # ==========================================
    f_cens_audio = extract_cens_audio(audio_path)
    
    # ==========================================
    # 2. MIDI FEATURE EXTRACTION
    # ==========================================
    #Load synthesized corrupt midi features
    root_folder = '/'.join(audio_path.split('/')[:-2])
    midi_file = midi_path.split('/')[-1].split('.')[0]
    midi_path = f'{root_folder}/corrupt_midi_audio/{midi_file}.wav'
    f_cens_midi = extract_cens_audio(midi_path)

    # ==========================================
    # 3. COST MATRIX & DTW
    # ==========================================
    # We use the inner product (cosine similarity) converted to distance.
    # C[n, m] = 1 - <audio_feature[n], midi_feature[m]>
    C = 1 - np.dot(f_cens_audio.T, f_cens_midi)
    
    # 5. Compute Warping Path
    # Standard DTW weights [1, 1, 1] for horizontal, vertical, and diagonal steps
    step_sizes = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int64)
    step_weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    
    D, E, wp = compute_warping_path(C, step_sizes=step_sizes, step_weights=step_weights)
    
    # The alignment score is the value at the last cell of the accumulated cost matrix
    # Normalizing by path length helps compare scores across different file lengths
    raw_score = D[-1, -1]
    normalized_score = raw_score / wp.shape[1]

    alignment_confidence = 1 - (normalized_score / 2.0)
    # Ensure it stays strictly within [0, 1] due to potential float precision noise
    alignment_confidence = max(0.0, min(1.0, alignment_confidence))
    
    return alignment_confidence
