'''Python script to measure onset detection error for different instruments using transcription method only.'''

import os
import warnings
import pandas as pd
import pretty_midi
import numpy as np
from onset_detection_simple import get_onsets_from_midi, get_onsets_transcription, get_onsets_from_midi_by_note
from metric import mean_absolute_onset_error, note_based_onset_error
import argparse

def discover_timbres(samples_wavs_folder: str) -> list:
    """Descobrir timbres disponíveis baseado na estrutura de pastas."""
    timbres = []
    if not os.path.exists(samples_wavs_folder):
        print(f"Aviso: Pasta de áudios não encontrada: {samples_wavs_folder}")
        return []
    
    for item in sorted(os.listdir(samples_wavs_folder)):
        item_path = os.path.join(samples_wavs_folder, item)
        if os.path.isdir(item_path):
            # Verificar se há arquivos .wav nesta pasta
            wav_files = [f for f in os.listdir(item_path) if f.endswith('.wav')]
            if wav_files:
                timbres.append(item)
                print(f"Timbre encontrado: {item} ({len(wav_files)} arquivos)")
    
    return timbres

def main(by_note: bool,
         samples_folder: str,
         samples_wavs_folder: str,
         transcribed_midis_folder: str,
         output_prefix: str,
         model_name: str):
    """Main function to measure onset detection error using transcription method only."""
    
    # Descobrir timbres automaticamente
    timbres = discover_timbres(samples_wavs_folder)
    
    if not timbres:
        print("Erro: Nenhum timbre encontrado na pasta de áudios")
        return
    
    print(f"Timbres detectados: {timbres}")
    
    errors = [0] * len(timbres)
    squared_errors = [0] * len(timbres)  # Para MSE
    total_processed = [0] * len(timbres)  # Para calcular a média correta
    
    # Se calculando por nota, armazenar detalhes individuais
    if by_note:
        note_errors_details = [[] for _ in range(len(timbres))]
        note_squared_errors_details = [[] for _ in range(len(timbres))]  # Para erros quadráticos
        note_stats = [[] for _ in range(len(timbres))]
        all_note_details = []  # Para CSV detalhado

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

        for j, timbre in enumerate(timbres):
            # Arquivo de áudio sintetizado
            audio_filename = midi_file.replace('.midi', '.wav')
            audio_file = os.path.join(samples_wavs_folder, timbre, audio_filename)
            
            # Arquivo MIDI transcrito
            transcribed_filename = midi_file.replace('.midi', f'_{model_name}.mid')
            transcribed_midi_file = os.path.join(transcribed_midis_folder, timbre, transcribed_filename)
            
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
                        # Calcular erro quadrático médio
                        squared_error = np.mean([e**2 for e in error_result['note_errors']])
                        squared_errors[j] += squared_error
                        total_processed[j] += 1
                        note_errors_details[j].extend(error_result['note_errors'])
                        note_squared_errors_details[j].extend([e**2 for e in error_result['note_errors']])
                        
                        # Salvar estatísticas para este arquivo
                        note_stats[j].append({
                            'file': midi_file,
                            'matched_count': error_result['matched_count'],
                            'total_reference': error_result['total_reference'],
                            'total_estimated': error_result['total_estimated'],
                            'mean_error': error_result['mean_error'],
                            'mean_squared_error': squared_error
                        })
                        
                        # Adicionar detalhes de cada nota para CSV detalhado
                        for k, note_error in enumerate(error_result['note_errors']):
                            matched_pair = error_result['matched_pairs'][k] if k < len(error_result['matched_pairs']) else (None, None, note_error)
                            all_note_details.append({
                                'file': midi_file,
                                'timbre': timbre,
                                'note_error': note_error,
                                'estimated_note_idx': matched_pair[0],
                                'reference_note_idx': matched_pair[1]
                            })
                        
                        print(f"  {timbre}: {error_result['matched_count']}/{error_result['total_reference']} notas matched, erro médio = {error_result['mean_error']:.4f}")
                else:
                    mean_abs_error = mean_absolute_onset_error(midi_onsets, onsets, window=0.05)
                    if mean_abs_error is not None:
                        errors[j] += mean_abs_error
                        # Calcular erro quadrático (assumindo mesma janela de matching)
                        ref = np.array(midi_onsets)
                        est = np.array(onsets)
                        used_ref = np.zeros(len(ref), dtype=bool)
                        individual_errors = []
                        for e in est:
                            diffs = np.abs(ref - e)
                            diffs[used_ref] = np.inf
                            idx = np.argmin(diffs)
                            if diffs[idx] <= 0.05:
                                individual_errors.append(diffs[idx])
                                used_ref[idx] = True
                        if individual_errors:
                            mse = np.mean([e**2 for e in individual_errors])
                            squared_errors[j] += mse
                        total_processed[j] += 1
                        print(f"  {timbre}: erro = {mean_abs_error:.4f}")
                
            except Exception as e:
                warnings.warn(f"Erro ao processar {midi_file} - {timbre}: {str(e)}")
                continue

    # Calcula médias apenas para os timbres que foram processados
    final_errors = []
    final_squared_errors = []
    timbre_names = timbres
    
    print("\n" + "="*50)
    print("RESULTADOS FINAIS:")
    print("="*50)
    
    for j in range(len(timbres)):
        if total_processed[j] > 0:
            mae = errors[j] / total_processed[j]
            mse = squared_errors[j] / total_processed[j]
            final_errors.append(mae)
            final_squared_errors.append(mse)
            print(f"{timbre_names[j]}: {total_processed[j]} arquivos processados")
            print(f"  MAE (erro absoluto médio): {mae:.4f}")
            print(f"  MSE (erro quadrático médio): {mse:.6f}")
            print(f"  RMSE (raiz do erro quadrático médio): {np.sqrt(mse):.4f}")
            
            if by_note and note_errors_details[j]:
                all_note_errors = note_errors_details[j]
                all_squared_errors = note_squared_errors_details[j]
                print(f"  Detalhes por nota: {len(all_note_errors)} notas processadas")
                print(f"  MAE por nota: {np.mean(all_note_errors):.4f} ± {np.std(all_note_errors):.4f}")
                print(f"  MSE por nota: {np.mean(all_squared_errors):.6f}")
                print(f"  Mínimo: {np.min(all_note_errors):.4f}, Máximo: {np.max(all_note_errors):.4f}")
                
        else:
            final_errors.append(float('nan'))  # NaN para timbres não processados
            final_squared_errors.append(float('nan'))
            print(f"{timbre_names[j]}: Nenhum arquivo processado")

    #---------------------------Save results to CSV---------------------------#
    method_suffix = 'transcription_by_note' if by_note else 'transcription_grouped'
    
    # Salvar MAE (Mean Absolute Error)
    df_mae = pd.DataFrame([final_errors], columns=timbre_names, index=[method_suffix])
    csv_mae_path = f"{output_prefix}_{method_suffix}_mae.csv"
    if os.path.exists(csv_mae_path):
        df_mae.to_csv(csv_mae_path, mode='a', header=False)
    else:
        df_mae.to_csv(csv_mae_path, mode='w', header=True)
    
    # Salvar MSE (Mean Squared Error)
    df_mse = pd.DataFrame([final_squared_errors], columns=timbre_names, index=[method_suffix])
    csv_mse_path = f"{output_prefix}_{method_suffix}_mse.csv"
    if os.path.exists(csv_mse_path):
        df_mse.to_csv(csv_mse_path, mode='a', header=False)
    else:
        df_mse.to_csv(csv_mse_path, mode='w', header=True)
        
    print(f"\nResultados MAE salvos em {csv_mae_path}")
    print(f"Resultados MSE salvos em {csv_mse_path}")
    print(f"Resumo: {sum(total_processed)} arquivos processados no total")
    
    # Salvar detalhes por nota se aplicável
    if by_note:
        # CSV com estatísticas por arquivo e timbre
        detailed_csv_path = f"{output_prefix}_note_stats_transcription.csv"
        detailed_data = []
        
        for j, timbre in enumerate(timbre_names):
            for stat in note_stats[j]:
                detailed_data.append({
                    'timbre': timbre,
                    'file': stat['file'],
                    'matched_count': stat['matched_count'],
                    'total_reference': stat['total_reference'],
                    'total_estimated': stat['total_estimated'],
                    'mean_error': stat['mean_error'],
                    'mean_squared_error': stat['mean_squared_error']
                })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(detailed_csv_path, index=False)
            print(f"Estatísticas por arquivo salvos em {detailed_csv_path}")
        
        # CSV com todos os detalhes de notas individuais
        if all_note_details:
            all_notes_csv_path = f"{output_prefix}_all_note_errors_transcription.csv"
            all_notes_df = pd.DataFrame(all_note_details)
            all_notes_df.to_csv(all_notes_csv_path, index=False)
            print(f"Detalhes de todas as notas salvos em {all_notes_csv_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure onset detection error using transcription method.")
    parser.add_argument('--by-note', action='store_true', 
                        help='Calculate error per individual note instead of grouped onsets')
    parser.add_argument('--samples-folder', default='./samples',
                        help='Folder containing original MIDI files (default: ./samples)')
    parser.add_argument('--samples-wavs-folder', default='./samples_wavs',
                        help='Folder containing synthesized WAV files (default: ./samples_wavs)')
    parser.add_argument('--transcribed-midis-folder', default='./transcribed_midis',
                        help='Folder containing transcribed MIDI files (default: ./transcribed_midis)')
    parser.add_argument('--output-prefix', default='onset_detection_results',
                        help='Prefix for output CSV files (default: onset_detection_results)')
    parser.add_argument('--model-name', default='basic_pitch',
                        help='Name of the transcription model used (default: basic_pitch)')
    
    args = parser.parse_args()
    main(by_note=args.by_note, 
            samples_folder=args.samples_folder,
            samples_wavs_folder=args.samples_wavs_folder,
            transcribed_midis_folder=args.transcribed_midis_folder,
            output_prefix=args.output_prefix,
            model_name=args.model_name)