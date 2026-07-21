import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import soundfile as sf

def Embedding_Distance_Audio_MIDI_score(audio_features, midi_features):

    #Load CLAMP Embeddings
    audio_embeddings = audio_features
    midi_embeddings = midi_features

    #convert to tensors
    audio_embeddings = torch.tensor(audio_embeddings)
    midi_embeddings = torch.tensor(midi_embeddings)

    # 3. Calculate Cosine Similarity
    similarity = F.cosine_similarity(audio_embeddings, midi_embeddings, dim=0).item()

    #convert to [0,1] range
    similarity = (similarity + 1) / 2

    return similarity