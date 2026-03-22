#from scores import DTW_CENS_Score
from evaluator import Evaluator
import os

def random_score(synthesized_midi_path, corrupt_midi_path):
    import random
    return random.random()

def main():
    #SÃO 2200 runs pro smd. o score_calculator roda 2200 vezes.
    #get_score = DTW_CENS_Score  # Change to DTW_score to use the DTW-based metric
    get_score = random_score

    eval = Evaluator('/home/ldsbrandao/exp/smd', get_score)
    metrics = eval.evaluate()   
    metrics.to_csv(f'results/{get_score.__name__}_results.csv', index=True)

if __name__ == "__main__":
    main()