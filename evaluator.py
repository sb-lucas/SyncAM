import pretty_midi
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import pandas as pd
import os

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

    def __calculate_scores__(self):
        #calculate scores for all audio-midi pairs in all corruption types and saves in a df for later
        #calculation of metrics

        rows = []
        for synthesized_midi in os.listdir(self.synthesized_midi_folder):

            synthesized_midi_path = os.path.join(self.synthesized_midi_folder, synthesized_midi)
            song_name = synthesized_midi.split('.')[0]

            corrupt_midis = [
                midi for midi in os.listdir(self.corrupt_midi_folder)
                if midi.startswith(song_name)
            ]

            for corrupt_midi in corrupt_midis:

                corrupt_midi_path = os.path.join(self.corrupt_midi_folder, corrupt_midi)

                parts = corrupt_midi.split('_')

                corruption_amount = int(parts[-1].split('.')[0])/100
                corruption_type = parts[-2]

                score = self.score_calculator(synthesized_midi_path, corrupt_midi_path)

                rows.append({
                    "audio": song_name,
                    "midi": corrupt_midi,
                    "corruption_type": corruption_type,
                    "corruption_amount": corruption_amount,
                    "score": score,
                    "label": int(corruption_amount > 0)
                })

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