'''Python script to measure onset detection error for different instruments using various methods.'''


import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import mir_eval
import pretty_midi
import os
import random
import librosa.display
import pandas as pd
from tempfile import TemporaryDirectory
import pretty_midi
import soundfile as sf
import warnings
from midi_helpers import trim_random_midi_segment, render_midi_to_audio, INSTRUMENTS, synthesize_dataset
from onset_detection import get_onsets_from_midi, get_onsets_from_audio_librosa, get_madmom_onsets, get_onsets_transcription, get_onsets_from_midi_by_note
from metric import mean_absolute_onset_error, note_based_onset_error
import argparse
try:
    import madmom
except ImportError:
    print("madmom library not found. Some onset detection methods may not be available.")
    madmom = None  # madmom is optional
    
def main(method_name, by_note=False, **kwargs):
    errors = [0] * len(INSTRUMENTS)
    total_processed = [0] * len(INSTRUMENTS)  # Para calcular a média correta
    
    # Se calculando por nota, também armazenar detalhes individuais
    if by_note:
        note_errors_details = [[] for _ in range(len(INSTRUMENTS))]
        note_stats = [[] for _ in range(len(INSTRUMENTS))]
    
    # Pastas dos dados
    samples_folder = './samples'
    samples_wavs_folder = './samples_wavs'
    transcribed_midis_folder = './transcribed_midis'

    # Define a função de detecção de onsets com base no método
    if method_name == 'librosa':
        get_onsets = get_onsets_from_audio_librosa
    elif method_name in ['cnn', 'rnn', 'spectral']:
        get_onsets = get_madmom_onsets
    # Para 'transcription' não precisamos definir get_onsets

    # Lista todos os arquivos MIDI originais
    midi_files = [f for f in os.listdir(samples_folder) if f.endswith('.midi')]
    
    print(f"Encontrados {len(midi_files)} arquivos MIDI originais")
    print(f"Modo de cálculo: {'Por nota individual' if by_note else 'Por onsets agrupados'}")

    #--------------------------------Evaluate onset detection-------------------------#
    for i, midi_file in enumerate(midi_files):
        print(f"Processando {i+1}/{len(midi_files)}: {midi_file}")

        # Carrega onsets do MIDI original
        original_midi_path = os.path.join(samples_folder, midi_file)
        
        if by_note:
            midi_onsets = get_onsets_from_midi_by_note(original_midi_path, combine=0.03)
        else:
            midi_onsets = get_onsets_from_midi(original_midi_path, combine=0.03)

        for j, instrument in enumerate(INSTRUMENTS.keys()):
            instrument_id = instrument.replace(" ", "_").lower()
            
            # Arquivo de áudio sintetizado
            audio_filename = midi_file.replace('.midi', '.wav')
            audio_file = os.path.join(samples_wavs_folder, instrument_id, audio_filename)
            
            # Arquivo MIDI transcrito
            transcribed_filename = midi_file.replace('.midi', '_basic_pitch.mid')
            transcribed_midi_file = os.path.join(transcribed_midis_folder, instrument_id, transcribed_filename)
            
            # Verifica se os arquivos existem
            if not os.path.exists(audio_file):
                warnings.warn(f"Arquivo de áudio não encontrado: {audio_file}")
                continue
                
            if not os.path.exists(transcribed_midi_file):
                warnings.warn(f"Arquivo MIDI transcrito não encontrado: {transcribed_midi_file}")
                continue

            try:
                # Usa o método específico para obter onsets
                if method_name == 'transcription':
                    onsets = get_onsets_transcription(
                        audio_file=audio_file, 
                        transcribed_midi_file=transcribed_midi_file,
                        original_midi_file=original_midi_path,
                        by_note=by_note,
                        **kwargs
                    )
                else:
                    onsets = get_onsets(audio_file=audio_file, **kwargs)
                    # Para outros métodos, converter para formato de nota se necessário
                    if by_note:
                        # Converter lista de tempos para formato de nota (sem pitch info)
                        onsets = [{'onset': float(t), 'pitch': 60, 'velocity': 80, 'duration': 0.1, 'instrument': 0} 
                                for t in onsets]

                # Calcular erro
                if by_note:
                    error_result = note_based_onset_error(midi_onsets, onsets, window=0.05)
                    if error_result['mean_error'] is not None:
                        errors[j] += error_result['mean_error']
                        total_processed[j] += 1
                        note_errors_details[j].extend(error_result['note_errors'])
                        note_stats[j].append({
                            'file': midi_file,
                            'matched_count': error_result['matched_count'],
                            'total_reference': error_result['total_reference'],
                            'total_estimated': error_result['total_estimated'],
                            'mean_error': error_result['mean_error']
                        })
                else:
                    mean_abs_error = mean_absolute_onset_error(midi_onsets, onsets, window=0.05)
                    if mean_abs_error is not None:
                        errors[j] += mean_abs_error
                        total_processed[j] += 1
                
            except Exception as e:
                warnings.warn(f"Erro ao processar {midi_file} - {instrument}: {str(e)}")
                continue

    # Calcula médias apenas para os instrumentos que foram processados
    final_errors = []
    instrument_names = list(INSTRUMENTS.keys())
    
    for j in range(len(INSTRUMENTS)):
        if total_processed[j] > 0:
            final_errors.append(errors[j] / total_processed[j])
            print(f"{instrument_names[j]}: {total_processed[j]} arquivos processados, erro médio: {final_errors[-1]:.4f}")
            
            if by_note and note_errors_details[j]:
                all_note_errors = note_errors_details[j]
                print(f"  Detalhes por nota: {len(all_note_errors)} notas processadas")
                print(f"  Erro médio por nota: {np.mean(all_note_errors):.4f} ± {np.std(all_note_errors):.4f}")
                print(f"  Mínimo: {np.min(all_note_errors):.4f}, Máximo: {np.max(all_note_errors):.4f}")
                
        else:
            final_errors.append(float('nan'))  # NaN para instrumentos não processados
            print(f"{instrument_names[j]}: Nenhum arquivo processado")

    #---------------------------Save results to CSV---------------------------#
    method_suffix = f"_{method_name}_by_note" if by_note else f"_{method_name}_grouped"
    df_row = pd.DataFrame([final_errors], columns=instrument_names, index=[method_suffix])

    csv_path = "onset_detection_results.csv"

    if os.path.exists(csv_path):
        # Append mode, no header
        df_row.to_csv(csv_path, mode='a', header=False)
    else:
        # First time: include header
        df_row.to_csv(csv_path, mode='w', header=True)
        
    print(f"Resultados salvos em {csv_path}")
    print(f"Resumo: {sum(total_processed)} arquivos processados no total")
    
    # Salvar detalhes por nota se aplicável
    if by_note:
        detailed_csv_path = f"note_details_{method_name}.csv"
        detailed_data = []
        
        for j, instrument in enumerate(instrument_names):
            for stat in note_stats[j]:
                detailed_data.append({
                    'instrument': instrument,
                    'file': stat['file'],
                    'matched_count': stat['matched_count'],
                    'total_reference': stat['total_reference'],
                    'total_estimated': stat['total_estimated'],
                    'mean_error': stat['mean_error']
                })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(detailed_csv_path, index=False)
            print(f"Detalhes por nota salvos em {detailed_csv_path}")

if __name__ == "__main__":
    #specify onset detection method
    parser = argparse.ArgumentParser(description="Estimate measure error for different instruments.")
    parser.add_argument('--method', type=str, default='cnn', choices=['cnn', 'rnn', 'spectral', 'librosa', 'transcription'],
                        help='Onset detection method to use: "cnn" for Madmom CNN, "librosa" for Librosa onset detection.')
    parser.add_argument('--by-note', action='store_true', 
                        help='Calculate error per individual note instead of grouped onsets')

    args = parser.parse_args()
     
    if args.method == 'librosa':
        get_onsets = get_onsets_from_audio_librosa
        kwargs = {'hop_length': 512}
    elif args.method == 'cnn':
        get_onsets = get_madmom_onsets
        kwargs = {
            "method": "cnn",
        }
    elif args.method == 'rnn':
        get_onsets = get_madmom_onsets
        kwargs = {
            "method": "rnn",
        }
    elif args.method == 'spectral':
        get_onsets = get_madmom_onsets
        kwargs = {
            "method": "spectral",
            "onset_kwargs": dict(onset_method='superflux', fps=100),
        }
    elif args.method == 'transcription':
        # Para o método de transcrição, não precisamos definir get_onsets aqui
        # pois será tratado especialmente na função main
        kwargs = {'time_tolerance': 0.05}

    main(args.method, by_note=args.by_note, **kwargs)


    