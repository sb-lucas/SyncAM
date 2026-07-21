import numpy as np
import librosa
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from scipy.spatial.distance import cdist

def DTW_MrMs_CENS_score(audio_features, midi_features):
    """
    Uses Synctoolbox's multi-scale DTW to find the alignment path with no memory 
    footprint, then calculates the mean cosine distance along that path for the score.
    """

    audio_cens = audio_features
    midi_cens = midi_features

    # 1. Calculate your feature rate (frames per second)
    # If sample_rate=22050 and hop_length=64, your rate is 22050 / 64 = 344.53
    feature_rate = 22050 / 64    
    
    # 2. Run Memory-Restricted Multi-Scale DTW
    # It takes the features directly. No dense cost matrices are ever built.
    wp = sync_via_mrmsdtw(
        f_chroma1=audio_cens, 
        f_chroma2=midi_cens, 
        input_feature_rate=feature_rate
    )
    
    # 3. Extract features exactly along the warping path
    # wp is shape (2, T) where T is the total length of the alignment path.
    # This pulls out the matched frames side-by-side.
    wp_int = wp.astype(int)

    aligned_audio = audio_cens[:, wp_int[0, :]]
    aligned_midi  = midi_cens[:, wp_int[1, :]]
    
    # 4. Calculate Cosine Distance ONLY along the path
    # This is a 1D calculation, meaning it takes virtually zero memory.
    
    # Dot product along the feature axis
    dot_product = np.sum(aligned_audio * aligned_midi, axis=0)
    
    # Magnitudes for normalization
    norm_audio = np.linalg.norm(aligned_audio, axis=0)
    norm_midi = np.linalg.norm(aligned_midi, axis=0)
    
    # Calculate Cosine Distance (with epsilon to prevent division by zero on silence)
    epsilon = 1e-10
    cosine_distances = 1.0 - (dot_product / (norm_audio * norm_midi + epsilon))
    
    # 5. Normalize by path length and convert to your [0, 1] score
    mean_distance = np.mean(cosine_distances) # This is distance / path_length
    
    raw_score = 1.0 - mean_distance

    assert raw_score <= 1 and raw_score >= 0, f"Warning: raw score out of bounds: {raw_score}"
    print(f"Raw score (before clamping): {raw_score}")

    final_score = np.clip(raw_score, 0.0, 1.0) # Ensure the final score is in [0, 1]
    
    return final_score