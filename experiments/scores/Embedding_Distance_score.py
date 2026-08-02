import numpy as np

def Embedding_Distance_score(audio_embeddings, midi_embeddings):
    """
    Computes cosine similarity between two embedding vectors 
    and rescales from [-1, 1] to [0, 1].
    """
    # 1. Ensure vectors are flat 1D arrays (handles shapes like (1, D) or (D, 1))
    a = np.asarray(audio_embeddings).squeeze()
    m = np.asarray(midi_embeddings).squeeze()

    # 2. Compute Cosine Similarity
    dot_product = np.dot(a, m)
    norm_a = np.linalg.norm(a)
    norm_m = np.linalg.norm(m)

    epsilon = 1e-8
    similarity = dot_product / (norm_a * norm_m + epsilon)

    # 3. Rescale from [-1, 1] to [0, 1]
    rescaled_score = (similarity + 1.0) / 2.0

    # 4. Clip safely to absorb floating-point precision rounding
    return float(np.clip(rescaled_score, 0.0, 1.0))