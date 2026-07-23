import soundfile as sf
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from dotenv import load_dotenv

from scores.Cross_Correlation_CENS_score import Cross_Correlation_CENS_score #CC-CENS
from scores.Cross_Correlation_DLNCO_score import Cross_Correlation_DLNCO_score #CC-DLNCO
from scores.Cross_Correlation_Spectrogram_score import Cross_Correlation_Spectrogram_score #CC-SPEC

from scores.DTW_MrMs_CENS_score import DTW_MrMs_CENS_score #MrMsDTW-CENS
from scores.DTW_MrMs_DLNCO_score import DTW_MrMs_DLNCO_score #MrMsDTW-DLNCO

from scores.Embedding_Distance_Audio_Audio_score import Embedding_Distance_Audio_Audio_score #CLAP
from scores.Embedding_Distance_Audio_MIDI_score import Embedding_Distance_Audio_MIDI_score #CLAMP3

from scores.Onset_F_score import Onset_F_score #Onset-F1
from scores.Onset_Alignment_Error_score import Onset_Alignment_Error_score#Onset-MAE

def get_method_config(method, root_dir):
    """Returns feature folders, score function, and file extensions for a given method."""
    if method == 'CC-CENS':
        return (
            os.path.join(root_dir, "audios_cens"),
            os.path.join(root_dir, "corrupt_midi_cens"),
            Cross_Correlation_CENS_score,
            '.npy'
        )
    elif method == 'CC-DLNCO':
        return (
            os.path.join(root_dir, "audios_dlnco"),
            os.path.join(root_dir, "corrupt_midi_dlnco"),
            Cross_Correlation_DLNCO_score,
            '.npy'
        )
    elif method == 'CC-SPEC':
        return (
            os.path.join(root_dir, "audios_spectrogram"),
            os.path.join(root_dir, "corrupt_midi_spectrogram"),
            Cross_Correlation_Spectrogram_score,
            '.npy'
        )
    elif method == 'MrMsDTW-CENS':
        return (
            os.path.join(root_dir, "audios_cens"),
            os.path.join(root_dir, "corrupt_midi_cens"),
            DTW_MrMs_CENS_score,
            '.npy'
        )
    elif method == 'MrMsDTW-DLNCO':
        return (
            os.path.join(root_dir, "audios_dlnco"),
            os.path.join(root_dir, "corrupt_midi_dlnco"),
            DTW_MrMs_DLNCO_score,
            '.npy'
        )
    elif method == 'CLAMP3':
        return (
            os.path.join(root_dir, "audios_clamp"),
            os.path.join(root_dir, "corrupt_midi_clamp"),
            Embedding_Distance_Audio_MIDI_score,
            '.npy'
        )
    elif method == 'CLAP':
        return (
            os.path.join(root_dir, "audios_clap"),
            os.path.join(root_dir, "corrupt_midi_clap"),
            Embedding_Distance_Audio_Audio_score,
            '.npy'
        )
    elif method == 'Onset-F1':
        return (
            os.path.join(root_dir, "audios_onsets"),
            os.path.join(root_dir, "corrupt_midi_onsets"),
            Onset_F_score,
            '.npy'
        )
    elif method == 'Onset-MAE':
        return (
            os.path.join(root_dir, "audios_onsets"),
            os.path.join(root_dir, "corrupt_midi_onsets"),
            Onset_Alignment_Error_score,
            '.npy'
        )
    elif method == 'AMT-BP':
        return (
            os.path.join(root_dir, "audios_transcribed_bp"),
            os.path.join(root_dir, "corrupt_midi"),
            Onset_Alignment_Error_score,
            '.mid',
        )
    else:
        raise ValueError(f"Unknown method configuration: {method}")

def evaluate_audio_group(args):
    """
    Worker function executed in parallel.
    Loads the audio feature ONCE, then iterates through all associated MIDIs.
    """
    audio_path, midi_tasks, method_name, score_fn = args
    results = []

    # 1. Load Audio Feature ONCE
    # mmap_mode='r' is a huge speedup for numpy! 
    # It reads the file virtually without loading the whole thing to RAM at once if not needed.
    if method_name.startswith('AMT'):
        pass #for later definition
    else:
        audio_feature = np.load(audio_path, mmap_mode='r')

    # 2. Iterate through all corrupted MIDIs for this audio
    for idx, midi_path in midi_tasks:
        if method_name.startswith('AMT'):
            pass #for later definition
        else:
            midi_feature = np.load(midi_path, mmap_mode='r')
            score = score_fn(audio_feature, midi_feature)
            
        results.append((idx, float(score)))

    return results

def main():
    load_dotenv()
    
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    EXPERIMENTS_FOLDER = os.environ["EXPERIMENTS_FOLDER"]

    results_folder = os.path.join(EXPERIMENTS_FOLDER, 'results')
    os.makedirs(results_folder, exist_ok=True)

    #datasets = ['smd', 'maestro', 'maps/ENSTDkAm', 'maps/ENSTDkCl', 'musicnet']
    datasets = ['smd']
    #methods = ['CC-CENS', 'CC-DLNCO', 'CC-SPEC', 'MrMsDTW-CENS', 'MrMsDTW-DLNCO', 'CLAMP3', 'CLAP', 'Onset-F1', 'Onset-MAE', 'AMT-BP' 'outros amt']
    #methods = ['CC-CENS', 'CC-DLNCO', 'CC-SPEC', 'MrMsDTW-CENS', 'MrMsDTW-DLNCO', 'CLAMP3', 'Onset-F1']
    methods = ['CC-SPEC']

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, int(cpu_count * 0.8))
    print(f"Running on {max_workers} CPU cores.")

    for dataset in datasets:
        print(f"\nProcessing: {dataset}")
        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        corrupt_midi_dir = os.path.join(root_dir, "corrupt_midi")

        # Sanitize dataset string for clean file paths (e.g., 'maps/ENSTDkAm' -> 'maps_ENSTDkAm')
        dataset_clean_name = dataset.replace('/', '_')
        results_path = os.path.join(results_folder, f"{dataset_clean_name}_results.csv")

        # 1. Initialize or load DataFrame
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
        else:
            df = pd.DataFrame(columns=["audio", "midi", "corruption_type", "corruption_amount"])
            midi_files = sorted([f for f in os.listdir(corrupt_midi_dir) if f.endswith(('.mid', '.midi'))])
            records = []

            for midi_file in midi_files:
                parts = midi_file.split('_')
                song_name = '_'.join(parts[:-2])
                corruption_type = parts[-2]
                corruption_amount = int(parts[-1].split('.')[0])

                records.append({
                    "audio": song_name,
                    "corrupt_midi": os.path.splitext(midi_file)[0],
                    "corruption_type": corruption_type,
                    "corruption_amount": corruption_amount
                })

            df = pd.DataFrame(records)
            df.to_csv(results_path, index=False)
            
        # 2. Iterate through each method and compute scores

        #aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        #PEGANDO SÓ UM PEDACINHO PRA VER 
        df = df.head(50)  # Limit to first 10 rows for testing
        #aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii

        for method in methods:
            # Initialize method column if missing
            if method not in df.columns:
                df[method] = np.nan

            # Filter rows that haven't been computed yet (resumability)
            uncomputed_mask = df[method].isna()
            uncomputed_df = df.loc[uncomputed_mask]
            uncomputed_indices = uncomputed_df.index.tolist()
            grouped = uncomputed_df.groupby('audio')

            if len(uncomputed_indices) == 0:
                print(f" -> Method [{method}]: All rows already computed. Skipping.")
                continue

            print(f" -> Method [{method}]: Computing {len(uncomputed_indices)} pending pairs...")
            audio_folder, midi_folder, score_fn, ext = get_method_config(method, root_dir)
            
            tasks = []
            for audio_name, group in grouped:
                audio_path = os.path.join(audio_folder, f"{audio_name}{ext}")
                
                # List of (dataframe_index, midi_path) tuples
                midi_tasks = [
                    (idx, os.path.join(midi_folder, f"{row['corrupt_midi']}{ext}"))
                    for idx, row in group.iterrows()
                ]
                
                tasks.append((audio_path, midi_tasks, method, score_fn))

            # Run scoring in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(evaluate_audio_group, task) for task in tasks]

                total_completed = 0
                for future in as_completed(futures):
                    group_results = future.result()
                    
                    # Update dataframe with all results from this audio group
                    for idx, score in group_results:
                        df.loc[idx, method] = score
                        total_completed += 1

                    # Save progress periodically (e.g., every 1000 pairs)
                    if total_completed % 500 == 0 or total_completed == len(uncomputed_indices):
                        print(f"    Progress [{method}]: {total_completed}/{len(uncomputed_indices)} completed.")
                        df.to_csv(results_path, index=False)

    print("\nAll experiment runs completed successfully!")

if __name__ == "__main__":
    main()

