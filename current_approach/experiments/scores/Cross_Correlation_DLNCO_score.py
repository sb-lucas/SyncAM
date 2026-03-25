import numpy as np

def cross_correlation_dlnco_score(audio_path, midi_path):
    """
    Alignment score WITHOUT allowing time shifts.
    Penalizes onset misalignment.

    Args:
        S1: (F, T1)
        S2: (F, T2)

    Returns:
        score in [0, 1]
    """

    #Load DLNCO features
    root_folder = '/'.join(audio_path.split('/')[:-2])
    audio_file = audio_path.split('/')[-1].split('.')[0]
    midi_file = midi_path.split('/')[-1].split('.')[0]
    S1 = np.load(f'{root_folder}/synthesized_midi_dlnco/{audio_file}.npy')
    S2 = np.load(f'{root_folder}/corrupt_midi_dlnco/{midi_file}.npy')

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

    assert corr <= 1 and corr >= 0, f"Warning: precisa reescalar: {corr}"

    return float(np.clip(corr, 0.0, 1.0))