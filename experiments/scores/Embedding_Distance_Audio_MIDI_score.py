import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import soundfile as sf

def Embedding_Distance_Audio_MIDI_score(audio_path, midi_path):

    #Load CLAMP Embeddings
    root_folder = '/'.join(audio_path.split('/')[:-2])
    audio_file = audio_path.split('/')[-1].split('.')[0]
    midi_file = midi_path.split('/')[-1].split('.')[0]

    audio_embeddings = np.load(f'{root_folder}/synthesized_midi_clamp/{audio_file}.npy')
    midi_embeddings = np.load(f'{root_folder}/corrupt_midi_clamp/{midi_file}.npy')

    #convert to tensors
    audio_embeddings = torch.tensor(audio_embeddings)
    midi_embeddings = torch.tensor(midi_embeddings)

    # 3. Calculate Cosine Similarity
    similarity = F.cosine_similarity(audio_embeddings, midi_embeddings, dim=0).item()

    #convert to [0,1] range
    similarity = (similarity + 1) / 2

    return similarity