import numpy as np
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw

def MrMsDTW_score(audio_features, midi_features, feature_rate=22050/64):
    """
    Uses Synctoolbox's multi-scale DTW to find the alignment path with no memory 
    footprint, then calculates the mean cosine distance along that path for the score.
    """

    # Run Memory-Restricted Multi-Scale DTW
    # It takes the features directly. No dense cost matrices are ever built.
    wp = sync_via_mrmsdtw(
        f_chroma1=audio_features, 
        f_chroma2=midi_features, 
        input_feature_rate=feature_rate
    )
    
    # Extract features exactly along the warping path
    # wp is shape (2, T) where T is the total length of the alignment path.
    # This pulls out the matched frames side-by-side.
    wp_int = wp.astype(int)

    aligned_audio = audio_features[:, wp_int[0, :]]
    aligned_midi  = midi_features[:, wp_int[1, :]]
    
    # Calculate Cosine Similarity ONLY along the path
    dot_product = np.sum(aligned_audio * aligned_midi, axis=0)
    norm_audio = np.linalg.norm(aligned_audio, axis=0)
    norm_midi = np.linalg.norm(aligned_midi, axis=0)

    epsilon = 1e-10
    cosine_similarity = (dot_product / (norm_audio * norm_midi + epsilon))
    
    # Normalize by path length and convert to your [0, 1] score
    raw_score = np.mean(cosine_similarity) # This is similarity / path_length
    
    assert raw_score <= 1 and raw_score >= 0, f"Warning: raw score out of bounds: {raw_score}"

    final_score = float(np.clip(raw_score, 0.0, 1.0)) # Ensure the final score is in [0, 1]
    
    return final_score