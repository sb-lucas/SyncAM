# from scores.Cross_Correlation_CENS_score import cross_correlation_cens_score
# from scores.Cross_Correlation_Spectrogram_score import cross_correlation_spectrogram_score
# from scores.Cross_Correlation_DLNCO_score import cross_correlation_dlnco_score

# from scores.DTW_MrMs_DLNCO_score import DTW_MrMs_DLNCO_score
# from scores.DTW_MrMs_CENS_score import DTW_MrMs_CENS_score
from scores.DTW_Standard_CENS_score import DTW_Standard_CENS_score

#from scores.Onset_F_score import Onset_F_score
#from scores.Onset_Alignment_Error_score import Onset_Alignment_Error_score

# from scores.Embedding_Distance_Audio_Audio_score import Embedding_Distance_Audio_Audio_score
# from scores.Embedding_Distance_Audio_MIDI_score import Embedding_Distance_Audio_MIDI_score

from evaluator import Evaluator
import os

def random_score(synthesized_midi_path, corrupt_midi_path):
    import random
    return random.random()

def main():
    #SÃO 2200 runs pro smd. o score_calculator roda 2200 vezes.

    #get_score = cross_correlation_spectrogram_score
    #get_score = cross_correlation_cens_score
    #get_score = cross_correlation_dlnco_score

    #get_score = DTW_MrMs_DLNCO_score
    #get_score = DTW_MrMs_CENS_score
    get_score = DTW_Standard_CENS_score

    #get_score = Onset_F_score
    #get_score = Onset_Alignment_Error_score

    #get_score = Embedding_Distance_Audio_MIDI_score
    #get_score = Embedding_Distance_Audio_Audio_score

    #get_score = random_score

    eval = Evaluator('/home/ldsbrandao/exp/smd', get_score)
    metrics = eval.evaluate()   
    metrics.to_csv(f'results/{get_score.__name__}_results.csv', index=True)

if __name__ == "__main__":
    main()