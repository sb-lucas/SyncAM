import soundfile as sf
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import roc_auc_score

class Evaluator():
    def __init__(self, root_folder, score_calculator):
        #we expect a root_folder with the dataset
        #in it, there should be 4 subfolders: 
        # 'audio' -> folder with original audio
        # 'midi' -> folder with original midi files
        # 'synthesized_midi' -> folder with audio synthesized from the original midi files
        # 'corrupt_midi' -> folder with corrupted midi files

        #score_calculator is a function that takes in two files (audio and midi) and returns a score of how similar they are.
        #the output should be a number between 0 and 1, where 1 means they are very similar and 0 means they are very different.

        self.root_folder = root_folder
        self.midi_folder = os.path.join(root_folder, 'midi')
        self.audio_folder = os.path.join(root_folder, 'audio')
        self.corrupt_midi_folder = os.path.join(root_folder, 'corrupt_midi')
        self.synthesized_midi_folder = os.path.join(root_folder, 'synthesized_midi')

        self.score_calculator = score_calculator

    def evaluate(self):
        df = self.__calculate_scores__()
        metrics = self.__calculate_metrics__(df)

        return metrics
    
    @staticmethod
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

    def __calculate_scores__(self):
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
    
    def __calculate_metrics__(self, df):
        #uses calculate scores and aggregates them in different groups to return metrics in a df

        #monotonicity
        monotonicity = df.groupby("corruption_type").apply(
            lambda x: spearmanr(x["corruption_amount"], x["score"]).statistic
        )

        #sensitivity
        sensitivity = df.groupby(["audio", "corruption_type"]).apply(
            lambda x: np.polyfit(x["corruption_amount"], x["score"], 1)[0]
        )
        sensitivity_mean = sensitivity.groupby("corruption_type").mean()

        #stability
        stability_levels = df.groupby(["corruption_type", "corruption_amount"]).apply(
            lambda x: x["score"].var()
        )
        stability = stability_levels.groupby("corruption_type").mean()

        #separability
        separability = df.groupby("corruption_type").apply(
            lambda x: roc_auc_score(x["label"], -x["score"])
        )

        #aggregate metrics into a DataFrame
        df_metrics = pd.concat(
            [monotonicity, sensitivity_mean, stability, separability],
            axis=1
        )
        df_metrics.columns = [
            "monotonicity",
            "sensitivity",
            "stability",
            "separability"
        ]
        df_metrics = df_metrics.reindex(["addRemove", "onset", "pitch", "all"])


        return df_metrics