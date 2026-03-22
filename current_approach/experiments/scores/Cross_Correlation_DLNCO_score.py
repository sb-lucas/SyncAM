import numpy as np

def normalized_cross_correlation(X, Y):
    """
    X: (12, T1)
    Y: (12, T2)
    """

    # L2 normalize along chroma dimension (important)
    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-8)
    Y = Y / (np.linalg.norm(Y, axis=0, keepdims=True) + 1e-8)

    T1 = X.shape[1]
    T2 = Y.shape[1]

    max_corr = -np.inf

    # lag: shift Y relative to X
    for lag in range(-T2 + 1, T1):
        
        if lag < 0:
            x_start = 0
            y_start = -lag
        else:
            x_start = lag
            y_start = 0

        length = min(T1 - x_start, T2 - y_start)
        if length <= 0:
            continue

        X_seg = X[:, x_start:x_start+length]
        Y_seg = Y[:, y_start:y_start+length]

        # cosine similarity per frame, then mean
        sim = np.sum(X_seg * Y_seg, axis=0)  # (length,)
        corr = np.mean(sim)

        if corr > max_corr:
            max_corr = corr

    return max_corr

def DLNCO_cross_correlation_score(dlnco_audio, dlnco_midi):

    # compute max correlation
    max_corr = normalized_cross_correlation(dlnco_audio, dlnco_midi)

    # max_corr is cosine similarity ∈ [-1, 1]
    score = (max_corr + 1.0) / 2.0

    return np.clip(score, 0.0, 1.0)