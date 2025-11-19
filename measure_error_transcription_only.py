'''Python script to measure onset detection error for different instruments using transcription method only.'''

import os
import warnings
import pandas as pd
import pretty_midi
import numpy as np
from midi_helpers import INSTRUMENTS
from onset_detection_simple import get_onsets_from_midi, get_onsets_transcription, get_onsets_from_midi_by_note
from metric import mean_absolute_onset_error, note_based_onset_error
import argparse

def main(by_note=False):
    errors = [0] * len(INSTRUMENTS)
    total_processed = [0] * len(INSTRUMENTS)  # Para calcular a média correta
    
    # Se calculando por nota, armazenar detalhes individuais
    if by_note:
        note_errors_details = [[] for _ in range(len(INSTRUMENTS))]
        note_stats = [[] for _ in range(len(INSTRUMENTS))]
        all_note_details = []  # Para CSV detalhado
    
    # Pastas dos dados
    samples_folder = './samples'
    samples_wavs_folder = './samples_wavs'
    transcribed_midis_folder = './transcribed_midis'

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
                # Usa o método de transcrição para obter onsets
                onsets = get_onsets_transcription(
                    audio_file=audio_file, 
                    transcribed_midi_file=transcribed_midi_file,
                    original_midi_file=original_midi_path,
                    time_tolerance=0.05,
                    by_note=by_note
                )

                # Calcular erro
                if by_note:
                    error_result = note_based_onset_error(midi_onsets, onsets, window=0.05)
                    if error_result['mean_error'] is not None:
                        errors[j] += error_result['mean_error']
                        total_processed[j] += 1
                        note_errors_details[j].extend(error_result['note_errors'])
                        
                        # Salvar estatísticas para este arquivo
                        note_stats[j].append({
                            'file': midi_file,
                            'matched_count': error_result['matched_count'],
                            'total_reference': error_result['total_reference'],
                            'total_estimated': error_result['total_estimated'],
                            'mean_error': error_result['mean_error']
                        })
                        
                        # Adicionar detalhes de cada nota para CSV detalhado
                        for k, note_error in enumerate(error_result['note_errors']):
                            matched_pair = error_result['matched_pairs'][k] if k < len(error_result['matched_pairs']) else (None, None, note_error)
                            all_note_details.append({
                                'file': midi_file,
                                'instrument': instrument,
                                'note_error': note_error,
                                'estimated_note_idx': matched_pair[0],
                                'reference_note_idx': matched_pair[1]
                            })
                        
                        print(f"  {instrument}: {error_result['matched_count']}/{error_result['total_reference']} notas matched, erro médio = {error_result['mean_error']:.4f}")
                else:
                    mean_abs_error = mean_absolute_onset_error(midi_onsets, onsets, window=0.05)
                    if mean_abs_error is not None:
                        errors[j] += mean_abs_error
                        total_processed[j] += 1
                        print(f"  {instrument}: erro = {mean_abs_error:.4f}")
                
            except Exception as e:
                warnings.warn(f"Erro ao processar {midi_file} - {instrument}: {str(e)}")
                continue

    # Calcula médias apenas para os instrumentos que foram processados
    final_errors = []
    instrument_names = list(INSTRUMENTS.keys())
    
    print("\n" + "="*50)
    print("RESULTADOS FINAIS:")
    print("="*50)
    
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
    method_suffix = 'transcription_by_note' if by_note else 'transcription_grouped'
    df_row = pd.DataFrame([final_errors], columns=instrument_names, index=[method_suffix])

    csv_path = f"onset_detection_results_{method_suffix}.csv"

    if os.path.exists(csv_path):
        # Append mode, no header
        df_row.to_csv(csv_path, mode='a', header=False)
    else:
        # First time: include header
        df_row.to_csv(csv_path, mode='w', header=True)
        
    print(f"\nResultados salvos em {csv_path}")
    print(f"Resumo: {sum(total_processed)} arquivos processados no total")
    
    # Salvar detalhes por nota se aplicável
    if by_note:
        # CSV com estatísticas por arquivo e instrumento
        detailed_csv_path = "note_stats_transcription.csv"
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
            print(f"Estatísticas por arquivo salvos em {detailed_csv_path}")
        
        # CSV com todos os detalhes de notas individuais
        if all_note_details:
            all_notes_csv_path = "all_note_errors_transcription.csv"
            all_notes_df = pd.DataFrame(all_note_details)
            all_notes_df.to_csv(all_notes_csv_path, index=False)
            print(f"Detalhes de todas as notas salvos em {all_notes_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure onset detection error using transcription method.")
    parser.add_argument('--by-note', action='store_true', 
                        help='Calculate error per individual note instead of grouped onsets')
    
    args = parser.parse_args()
    main(by_note=args.by_note)