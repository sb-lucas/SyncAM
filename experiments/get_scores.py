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

def process_pair(args):
    synthesized_midi, corrupt_midi, synthesized_dir, corrupt_dir, score_calculator = args

    synthesized_midi_path = os.path.join(synthesized_dir, synthesized_midi)
    corrupt_midi_path = os.path.join(corrupt_dir, corrupt_midi)

    song_name = synthesized_midi.split('.')[0]
    parts = corrupt_midi.split('_')

    corruption_amount = int(parts[-1].split('.')[0]) / 100
    corruption_type = parts[-2]

    score = score_calculator(synthesized_midi_path, corrupt_midi_path)

    return {
        "audio": song_name,
        "midi": corrupt_midi,
        "corruption_type": corruption_type,
        "corruption_amount": corruption_amount,
        "score": score,
        "label": int(corruption_amount > 0)
    }

def calculate_scores(self):
    #calculate scores for all audio-midi pairs in all corruption types and saves in a df for later
    #calculation of metrics

    #build list of tasks to process in parallel
    tasks = []
    for synthesized_midi in os.listdir(self.synthesized_midi_folder):
        song_name = synthesized_midi.split('.')[0]

        corrupt_midis = [
            midi for midi in os.listdir(self.corrupt_midi_folder)
            if midi.startswith(song_name)
        ]

        for corrupt_midi in corrupt_midis:
            tasks.append((
                synthesized_midi,
                corrupt_midi,
                self.synthesized_midi_folder,
                self.corrupt_midi_folder,
                self.score_calculator
            ))

    #define the number of workers for parallel processing, using 70% of available CPU cores
    max_workers = max(1, int(os.cpu_count() * 0.1))
    #max_workers = 1
    print(f"Processing {len(tasks)} tasks with {max_workers} workers...")

    #process tasks in parallel and collect results
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(self.process_pair, task) for task in tasks]

        for i, f in enumerate(as_completed(futures)):
            try:
                rows.append(f.result())
                print(f"{i+1}/{len(tasks)}")
            except Exception as e:
                print(f"FAILED {i+1}: {e}")

    df = pd.DataFrame(rows)

    return df

def evaluate_pair(args):
    """
    Worker function executed in parallel.
    Loads feature arrays (or paths) and computes the alignment score for one pair.
    """
    idx, audio_path, midi_path, method_name, score_fn = args

    # AMT score functions operating directly on file/MIDI paths
    if method_name.startswith('AMT'):
        score = score_fn(audio_path, midi_path)
    else:
        # Feature-based functions operating on pre-extracted numpy arrays
        audio_feature = np.load(audio_path, allow_pickle=True)
        midi_feature = np.load(midi_path, allow_pickle=True)
        score = score_fn(audio_feature, midi_feature)

    return idx, float(score)

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

def main():
    load_dotenv()
    
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    EXPERIMENTS_FOLDER = os.environ["EXPERIMENTS_FOLDER"]

    results_folder = os.path.join(EXPERIMENTS_FOLDER, 'results')
    os.makedirs(results_folder, exist_ok=True)

    datasets = ['maestro', 'maps/ENSTDkAm', 'maps/ENSTDkCl', 'musicnet', 'smd']
    #methods = ['CC-CENS', 'CC-DLNCO', 'CC-SPEC', 'MrMsDTW-CENS', 'MrMsDTW-DLNCO', 'CLAMP3', 'CLAP', 'Onset-F1', 'Onset-MAE', 'AMT-BP' 'outros amt']
    
    methods = ['CC-CENS', 'CC-DLNCO', 'CC-SPEC', 'MrMsDTW-CENS', 'MrMsDTW-DLNCO', 'CLAMP3', 'Onset-F1']
#    methods = ['CC-DLNCO', 'CC-SPEC', 'MrMsDTW-DLNCO', 'CLAMP3', 'Onset-F1']
    #datasets = ['musicnet', 'smd']

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
            uncomputed_indices = df[uncomputed_mask].index.tolist()

            if len(uncomputed_indices) == 0:
                print(f" -> Method [{method}]: All rows already computed. Skipping.")
                continue

            print(f" -> Method [{method}]: Computing {len(uncomputed_indices)} pending pairs...")

            audio_folder, midi_folder, score_fn, ext = get_method_config(method, root_dir)

            tasks = []
            for idx in uncomputed_indices:
                row = df.loc[idx]
                audio_path = os.path.join(audio_folder, f"{row['audio']}{ext}")
                midi_path = os.path.join(midi_folder, f"{row['corrupt_midi']}{ext}")
                tasks.append((idx, audio_path, midi_path, method, score_fn))

            # Run scoring in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(evaluate_pair, task) for task in tasks]

                for i, future in enumerate(as_completed(futures), start=1):
                    idx, score = future.result()
                    df.loc[idx, method] = score

                        # Save progress after every 500 computations or at the end
                    if i % 500 == 0 or i == len(tasks):
                        print(f"    Progress [{method}]: {i}/{len(tasks)} completed.")
                        df.to_csv(results_path, index=False)

    print("\nAll experiment runs completed successfully!")

if __name__ == "__main__":
    main()