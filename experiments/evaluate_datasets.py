import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from scores.Cosine_Similarity_score import Cosine_Similarity_score #CS-SPEC, CS-CENS, CS-DLNCO
from scores.MrMsDTW_score import MrMsDTW_score #MrMsDTW-CENS, MrMsDTW-DLNCO
from scores.Embedding_Distance_score import Embedding_Distance_score #CLAMP3, CLAP
from scores.Onset_F_score import Onset_F_score #Onset-F1
from scores.Onset_Alignment_Error_score import Onset_Alignment_Error_score #Onset-MAE
from scores.Transcription_score import Transcription_score #AMT-BP, AMT-BD, AMT-TK
from scores.Transcription_IOI_score import Transcription_IOI_score #AMT-IOI


def get_method_config(method, root_dir):
    """Returns feature folders, score function, and file extensions for a given method."""
    if method == 'CS-CENS':
        return (
            os.path.join(root_dir, "audios_cens"),
            os.path.join(root_dir, "original_midi_cens"),
            Cosine_Similarity_score,
            '.npy'
        )
    elif method == 'CS-DLNCO':
        return (
            os.path.join(root_dir, "audios_dlnco"),
            os.path.join(root_dir, "original_midi_dlnco"),
            Cosine_Similarity_score,
            '.npy'
        )
    elif method == 'CS-SPEC':
        return (
            os.path.join(root_dir, "audios_spectrogram"),
            os.path.join(root_dir, "original_midi_spectrogram"),
            Cosine_Similarity_score,
            '.npy'
        )
    elif method == 'MrMsDTW-CENS':
        return (
            os.path.join(root_dir, "audios_cens"),
            os.path.join(root_dir, "original_midi_cens"),
            MrMsDTW_score,
            '.npy'
        )
    elif method == 'MrMsDTW-DLNCO':
        return (
            os.path.join(root_dir, "audios_dlnco"),
            os.path.join(root_dir, "original_midi_dlnco"),
            MrMsDTW_score,
            '.npy'
        )
    elif method == 'CLAMP3':
        return (
            os.path.join(root_dir, "audios_clamp"),
            os.path.join(root_dir, "original_midi_clamp"),
            Embedding_Distance_score,
            '.npy'
        )
    elif method == 'CLAP':
        return (
            os.path.join(root_dir, "audios_clap"),
            os.path.join(root_dir, "original_midi_clap"),
            Embedding_Distance_score,
            '.npy'
        )
    elif method == 'Onset-F1':
        return (
            os.path.join(root_dir, "audios_onsets"),
            os.path.join(root_dir, "original_midi_onsets"),
            Onset_F_score,
            '.npy'
        )
    elif method == 'Onset-MAE':
        return (
            os.path.join(root_dir, "audios_onsets"),
            os.path.join(root_dir, "original_midi_onsets"),
            Onset_Alignment_Error_score,
            '.npy'
        )
    elif method == 'AMT-BP':
        return (
            os.path.join(root_dir, "audios_transcribed_bp"),
            os.path.join(root_dir, "midis"),
            Transcription_score,
            '.mid',
        )
    elif method == 'AMT-BD':
        return (
            os.path.join(root_dir, "audios_transcribed_bd"),
            os.path.join(root_dir, "midis"),
            Transcription_score,
            '.mid',
        )
    elif method == 'AMT-TK':
        return (
            os.path.join(root_dir, "audios_transcribed_tk"),
            os.path.join(root_dir, "midis"),
            Transcription_score,
            '.mid',
        )
    elif method == 'AMT-IOI-BP':
        return (
            os.path.join(root_dir, "audios_transcribed_bp"),
            os.path.join(root_dir, "midis"),
            Transcription_IOI_score,
            '.mid',
        )
    elif method == 'AMT-IOI-BD':
        return (
            os.path.join(root_dir, "audios_transcribed_bd"),
            os.path.join(root_dir, "midis"),
            Transcription_IOI_score,
            '.mid',
        )
    elif method == 'AMT-IOI-TK':
        return (
            os.path.join(root_dir, "audios_transcribed_tk"),
            os.path.join(root_dir, "midis"),
            Transcription_IOI_score,
            '.mid',
        )
    else:
        raise ValueError(f"Unknown method configuration: {method}")

def main():
    load_dotenv()
    
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    EXPERIMENTS_FOLDER = os.environ["EXPERIMENTS_FOLDER"]

    results_folder = os.path.join(EXPERIMENTS_FOLDER, 'results')
    os.makedirs(results_folder, exist_ok=True)
    csv_path = os.path.join(results_folder, "table_evaluate_datasets.csv")

    datasets = ['smd', 'maestro', 'maps_synth', 'maps_ENSTDkAm', 'maps_ENSTDkCl', 'musicnet']
    #methods = ['CS-CENS', 'CS-DLNCO', 'CS-SPEC', 'MrMsDTW-CENS', 'MrMsDTW-DLNCO', 'CLAMP3', 'CLAP', 'Onset-F1', 'Onset-MAE', 'AMT-BP', 'AMT-BD', 'AMT-TK', 'AMT-IOI-BP', 'AMT-IOI-BD', 'AMT-IOI-TK']
    methods = ['CS-CENS', 'CS-SPEC', 'MrMsDTW-CENS', 'AMT-IOI-BP', 'AMT-IOI-BD', 'AMT-IOI-TK']

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, int(cpu_count * 0.2))
    print(f"Running on {max_workers} CPU cores.")

    if os.path.exists(csv_path):
        print(f"CSV file {csv_path} already exists. Loading existing results.")
        df_existing = pd.read_csv(csv_path, index_col='Dataset')
        results = df_existing.to_dict(orient='index')
    else:
        results = {dataset: {method: "-1" for method in methods} for dataset in datasets}
    
    for dataset in datasets:
        print(f"\nProcessing: {dataset}")
        root_dir = os.path.join(DATASETS_FOLDER, dataset)

        if dataset not in results:
            results[dataset] = {method: "-1" for method in methods}

        # Iterate through each method and compute scores
        for method in methods:
            if str(results[dataset].get(method, "-1")) != "-1":
                print(f"Score for {method} in {dataset} already computed, skipping.")
                continue

            audios_dir, midis_dir, score_fn, file_ext = get_method_config(method, root_dir) 

            scores = []
            audio_files = os.listdir(audios_dir)
            
            for i, audio_file in enumerate(audio_files):
                print(f"[{i+1}/{len(audio_files)}] Processing {method} for {audio_file}")

                basename = os.path.splitext(audio_file)[0]
                audio_path = os.path.join(audios_dir, basename + file_ext)
                midi_path = os.path.join(midis_dir, basename + file_ext)

                if file_ext == '.npy':
                    audio_feat = np.load(audio_path)
                    midi_feat = np.load(midi_path)
                elif file_ext == '.mid':
                    audio_feat = audio_path
                    midi_feat = midi_path
                    if dataset == 'maestro':
                        midi_feat = midi_path + 'i'

                score = score_fn(audio_feat, midi_feat)
                scores.append(score)

            # Compute mean and standard deviation across all pairs
            mean_score = np.mean(scores)
            std_score = np.std(scores, ddof=1)  # Sample standard deviation

            # Save formatted string "mean ± std" to the dictionary
            results[dataset][method] = f"{mean_score} ± {std_score}"
            
            # Save checkpoint
            df = pd.DataFrame.from_dict(results, orient='index')
            df.index.name = 'Dataset'
            df.to_csv(csv_path)
            print('Saved checkpoint.')

if __name__ == "__main__":
    main()