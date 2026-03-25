import numpy as np
import librosa
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_CENS
from synctoolbox.dtw.core import compute_warping_path
import pretty_midi


def DTW_Standard_CENS_score(audio_path, midi_path):

    # ==========================================
    # 1. AUDIO FEATURE EXTRACTION
    # ==========================================
    feature_rate = 10

    # Load audio
    audio, fs = librosa.load(audio_path, sr=22050)
    
    # Extract 120-dimensional pitch features 
    f_pitch_audio = audio_to_pitch_features(audio, Fs=fs, feature_rate=feature_rate)
    
    # Compute CENS directly from the Pitch features
    f_cens_audio, CENS_feature_rate = pitch_to_CENS(f_pitch_audio, input_feature_rate=feature_rate)
    
    # ==========================================
    # 2. MIDI FEATURE EXTRACTION
    # ==========================================
    #Load synthesized corrupt midi features
    root_folder = '/'.join(audio_path.split('/')[:-2])
    midi_file = midi_path.split('/')[-1].split('.')[0]
    midi_path = f'{root_folder}/corrupt_midi_audio/{midi_file}.wav'
    midi_audio, fs = librosa.load(midi_path, sr=22050)
    
    f_pitch_midi = audio_to_pitch_features(midi_audio, Fs=22050, feature_rate=feature_rate)
    f_cens_midi, CENS_feature_rate = pitch_to_CENS(f_pitch_midi, input_feature_rate=feature_rate)

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
