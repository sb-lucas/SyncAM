import pandas as pd
import numpy as np
import os
import librosa as rosa
from transformers import AutoProcessor, ClapModel
from dotenv import load_dotenv
import torch
import argparse

class HuggingFaceClap:
    def __init__(self, hf_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(f"{hf_path}/laion/clap-htsat-unfused", local_files_only=True).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(f"{hf_path}/laion/clap-htsat-unfused", local_files_only=True)

    def get_audio_tokens(self, audio_path):
        # librosa returns a 1D array of shape (T,)
        audio_data, sr = rosa.load(audio_path, sr=48000)
        
        # 10 seconds * 48000 samples/sec = 480,000 samples per chunk
        chunk_size = 10 * sr 
        
        all_embeddings = []
        
        # Slice the audio into non-overlapping 10-second blocks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            
            # The processor accepts a list of 1D numpy arrays.
            # It automatically pads the final chunk if it is shorter than 10s.
            inputs = self.processor(
                text=[""], 
                audios=[chunk], 
                return_tensors="pt", 
                padding=True, 
                sampling_rate=sr
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            emb = outputs.audio_embeds

            # Move to CPU immediately to free up GPU VRAM
            all_embeddings.append(emb.cpu())

        #take the mean of all embeddings to get a single embedding for the entire audio
        # Resulting shape will be (1, 512)
        final_embeddings = torch.stack(all_embeddings).mean(dim=0)
        
        print(f"Extracted CLAP embedding for {os.path.basename(audio_path)}: {final_embeddings.shape}")
        
        return final_embeddings


def main():
    # ----------------------------- Argparse Setup ---------------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the dataset to use"
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Extract only from the original data."
    )
    args = parser.parse_args()
    # ----------------------------- Environment & Config ---------------------------------- #
    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    #if args.original:
        #datasets = ['maestro', 'smd', 'maps_synth', 'maps_ENSTDkAm', 'maps_ENSTDkCl', 'musicnet']
    #else:
        #datasets = ['maestro', 'smd', 'maps_synth']

    datasets = [args.dataset_name]

    #----------------------Load the models once--------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    HF_PATH = os.environ["HF_PATH"]
    model = HuggingFaceClap(hf_path=HF_PATH)

    #main loop
    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        original_audio_dir = os.path.join(root_dir, "audios")
        original_audio_clap_dir = os.path.join(root_dir, "audios_clap")

        if args.original:
            synthesized_midi_dir = os.path.join(root_dir, "synthesized_original_midi")
            corrupt_midi_clap_dir = os.path.join(root_dir, "original_midi_clap")
        else:
            synthesized_midi_dir = os.path.join(root_dir, "synthesized_midi")
            corrupt_midi_clap_dir = os.path.join(root_dir, "corrupt_midi_clap")
                

        os.makedirs(corrupt_midi_clap_dir, exist_ok=True)
        os.makedirs(original_audio_clap_dir, exist_ok=True)

        #----------------------Extract CLAP from original audio ----------------#
        audio_files = os.listdir(original_audio_dir)
        tam_original = len(audio_files)
        for i, file in enumerate(audio_files):
            file_name = os.path.splitext(file)[0] + ".npy"
            save_path = os.path.join(original_audio_clap_dir, file_name)
            
            if os.path.exists(save_path):
                print(f"CLAP embedding already exists for {file_name}, skipping.")
                continue

            emb = model.get_audio_tokens(os.path.join(original_audio_dir, file))
            np.save(save_path, emb.squeeze().detach().cpu().numpy())
            print(f"{i+1}/{tam_original}")
            
        #----------------------Extract CLAP of synthesized audio from corrupt MIDIs----------------#
        audio_files = os.listdir(synthesized_midi_dir)
        tam_corrupt = len(audio_files)
        for i, file in enumerate(audio_files):
            file_name = os.path.splitext(file)[0] + ".npy"
            save_path = os.path.join(corrupt_midi_clap_dir, file_name)
            
            if os.path.exists(save_path):
                print(f"CLAP embedding already exists for {file_name}, skipping.")
                continue

            emb = model.get_audio_tokens(os.path.join(synthesized_midi_dir, file))
            np.save(save_path, emb.squeeze().detach().cpu().numpy())
            print(f"{i+1}/{tam_corrupt}")

if __name__ == "__main__":
    print("começo")
    main()

