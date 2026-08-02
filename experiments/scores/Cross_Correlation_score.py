import numpy as np
#import torch

def Cross_Correlation_score(audio_features, midi_features):
    """
    Computes the cross-correlation between two sets of features.

    Args:
        audio_features: (F, T1)
        midi_features: (F, T2)

    Returns:
        score in [0, 1]
    """
    S1 = audio_features
    S2 = midi_features

    #make sure they have the same number of frequency bins
    assert S1.shape[0] == S2.shape[0] 

    # --- match length (truncate to shortest) ---
    T = min(S1.shape[1], S2.shape[1])
    S1 = S1[:, :T]
    S2 = S2[:, :T]

    def normalize(S):
        return S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-8)

    S1 = normalize(S1)
    S2 = normalize(S2)

    # --- cosine similarity across all frames ---
    corr = np.mean(np.sum(S1 * S2, axis=0))

    assert corr <= 1 and corr >= 0, f"Warning: need to rescale: {corr}"

    return float(np.clip(corr, 0.0, 1.0))