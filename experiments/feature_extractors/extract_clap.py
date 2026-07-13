import pandas as pd
import numpy as np
import os
import torch
import librosa as rosa
from transformers import AutoProcessor, ClapModel

class HuggingFaceClap:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained("/home/ldsbrandao/exp/hf_models/laion/clap-htsat-unfused", local_files_only=True).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("/home/ldsbrandao/exp/hf_models/laion/clap-htsat-unfused", local_files_only=True)

    def get_audio_tokens(self, audio_path):
        audio_data, _ = rosa.load(audio_path, sr=48000)  # shape (T,)
        audio_data = audio_data.reshape(1, -1)  # (1, T)

        input_text = [""]
        inputs = self.processor(text=input_text, audio=audio_data, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        #local tokens, of shape [L,D], where L is the number of tokens and D is the dimension of the embeddings
        audio_tokens = outputs.audio_model_output.last_hidden_state  # [1, 1024, 2, 32]
        audio_tokens = audio_tokens.squeeze(0)  # [1024, 2, 32]
        audio_tokens = audio_tokens.permute(1, 2, 0)  # [2, 32, 1024]
        emb_size = audio_tokens.shape[-1]  # 1024
        local_audio_tokens = audio_tokens.reshape(-1, emb_size)  # [64, 1024]

        global_audio_tokens = outputs.audio_embeds

        return global_audio_tokens, local_audio_tokens

    def get_text_tokens(self, input_text):
        #generate fake audio data for testing
        audio_data = np.random.rand(1, 48000)

        input_text = [input_text]

        inputs = self.processor(text=input_text, audio=audio_data, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        global_text_tokens = outputs.text_embeds

        #local tokens, of shape [L,D], where L is the number of tokens and D is the dimension of the embeddings
        local_text_tokens = outputs.text_model_output.last_hidden_state 

        return global_text_tokens, local_text_tokens

def main():
    #-----------------------------Load Paths----------------------------------#
    root_dir = "/home/ldsbrandao/exp/smd"
    corrupt_midi_audio_dir = os.path.join(root_dir, "corrupt_midi_audio")
    synthesized_midi_dir = os.path.join(root_dir, "synthesized_midi")

    corrupt_midi_clap_dir = os.path.join(root_dir, "corrupt_midi_clap")
    synthesized_midi_clap_dir = os.path.join(root_dir, "synthesized_midi_clap")

    os.makedirs(corrupt_midi_clap_dir, exist_ok=True)
    os.makedirs(synthesized_midi_clap_dir, exist_ok=True)

    #----------------------Load the models once--------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    model = HuggingFaceClap()

    #----------------------Extract CLAP of synthesized audio from original MIDIs----------------#
    tam_original = len(os.listdir(synthesized_midi_dir))
    audio_files = os.listdir(synthesized_midi_dir)
    for i, file in enumerate(audio_files):
        emb, _ = model.get_audio_tokens(os.path.join(synthesized_midi_dir, file))
        file_name = file.split('.')[0] + ".npy"
        np.save(os.path.join(synthesized_midi_clap_dir, file_name), emb.squeeze().detach().cpu().numpy())
        print(f"{i+1}/{tam_original}")
        
    #----------------------Extract CLAP of synthesized audio from corrupt MIDIs----------------#
    tam_corrupt = len(os.listdir(corrupt_midi_audio_dir))
    audio_files = os.listdir(corrupt_midi_audio_dir)
    for i, file in enumerate(audio_files):
        emb, _ = model.get_audio_tokens(os.path.join(corrupt_midi_audio_dir, file))
        file_name = file.split('.')[0] + ".npy"
        np.save(os.path.join(corrupt_midi_clap_dir, file_name), emb.squeeze().detach().cpu().numpy())
        print(f"{i+1}/{tam_corrupt}")

if __name__ == "__main__":
    print("começo")
    main()

