'''Python script to measure onset detection error using transcription method.

Compares original MIDI files with transcribed MIDI files to calculate onset errors.
Supports both grouped onset calculation and per-note error calculation.
'''

import os
import warnings
import pandas as pd
import pretty_midi
import numpy as np
from onset_detection_simple import get_onsets_from_midi, get_onsets_transcription, get_onsets_from_midi_by_note
from metric import mean_absolute_onset_error, note_based_onset_error
import argparse

def get_all_midi_pairs(samples_folder: str, transcribed_midis_folder: str, model_name: str):
    """Encontrar pares de MIDIs originais e transcritos.
    
    Args:
        samples_folder: Pasta com MIDIs originais
        transcribed_midis_folder: Pasta com MIDIs transcritos (busca recursiva)
        model_name: Nome do modelo usado na transcrição (sufixo dos arquivos)
        
    Returns:
        list: Lista de tuplas (original_path, transcribed_path, relative_name)
    """
    pairs = []
    
    # Listar todos os MIDIs originais
    if not os.path.exists(samples_folder):
        print(f"Erro: Pasta de MIDIs originais não encontrada: {samples_folder}")
        return []
    
    original_midis = []
    for root, dirs, files in os.walk(samples_folder):
        for f in files:
            if f.endswith('.midi') or f.endswith('.mid'):
                # Armazenar caminho relativo à pasta samples_folder
                rel_path = os.path.relpath(os.path.join(root, f), samples_folder)
                original_midis.append(rel_path)
    
    if not original_midis:
        print(f"Erro: Nenhum MIDI original encontrado em {samples_folder}")
        return []
    
    print(f"Encontrados {len(original_midis)} MIDIs originais")
    
    # Buscar MIDIs transcritos recursivamente
    for original_midi in sorted(original_midis):
        original_path = os.path.join(samples_folder, original_midi)
        
        # Nome base sem extensão
        base_name = os.path.splitext(original_midi)[0]
        
        # Procurar o MIDI transcrito correspondente (busca recursiva)
        # Procurar o MIDI transcrito correspondente (busca recursiva)
        # Tentar ambas extensões .mid e .midi
        transcribed_path = None
        
        for ext in ['.mid', '.midi']:
            transcribed_name = f"{base_name}_{model_name}{ext}"
            
            # Buscar recursivamente
            for root, dirs, files in os.walk(transcribed_midis_folder):
                if transcribed_name in files:
                    transcribed_path = os.path.join(root, transcribed_name)
                    break
            
            if transcribed_path:
                break
        
        if transcribed_path and os.path.exists(transcribed_path):
            pairs.append((original_path, transcribed_path, original_midi))
        else:
            warnings.warn(f"MIDI transcrito não encontrado para: {original_midi}")
    
    print(f"Encontrados {len(pairs)} pares de MIDIs (original + transcrito)")
    return pairs

def main(by_note: bool,
         samples_folder: str,
         transcribed_midis_folder: str,
         output_prefix: str,
         model_name: str):
    """Main function to measure onset detection error using transcription method."""
    
    # Encontrar pares de MIDIs
    midi_pairs = get_all_midi_pairs(samples_folder, transcribed_midis_folder, model_name)
    
    if not midi_pairs:
        print("Erro: Nenhum par de MIDI encontrado")
        return
    
    # Estatísticas
    total_error = 0.0
    total_squared_error = 0.0
    total_processed = 0
    
    # Para modo por nota
    if by_note:
        all_note_errors = []
        all_squared_errors = []
        note_stats = []
        all_note_details = []
    
    print(f"Modo de cálculo: {'Por nota individual' if by_note else 'Por onsets agrupados'}")
    print(f"{'='*60}")

    # Processar cada par de MIDIs
    for i, (original_path, transcribed_path, name) in enumerate(midi_pairs):
        print(f"[{i+1}/{len(midi_pairs)}] Processando: {name}")

        try:
            # Carregar onsets do MIDI original
            if by_note:
                midi_onsets = get_onsets_from_midi_by_note(original_path, combine=0.03)
            else:
                midi_onsets = get_onsets_from_midi(original_path, combine=0.03)
            
            # Carregar onsets do MIDI transcrito (sem filtro de áudio)
            transcribed_midi = pretty_midi.PrettyMIDI(transcribed_path)
            if by_note:
                onsets = get_onsets_from_midi_by_note(transcribed_midi, combine=0.03)
            else:
                onsets = get_onsets_from_midi(transcribed_midi, combine=0.03)

            # Calcular erro
            if by_note:
                error_result = note_based_onset_error(midi_onsets, onsets, window=0.05)
                if error_result['mean_error'] is not None:
                    total_error += error_result['mean_error']
                    squared_error = np.mean([e**2 for e in error_result['note_errors']])
                    total_squared_error += squared_error
                    total_processed += 1
                    
                    all_note_errors.extend(error_result['note_errors'])
                    all_squared_errors.extend([e**2 for e in error_result['note_errors']])
                    
                    note_stats.append({
                        'file': name,
                        'matched_count': error_result['matched_count'],
                        'total_reference': error_result['total_reference'],
                        'total_estimated': error_result['total_estimated'],
                        'mean_error': error_result['mean_error'],
                        'mean_squared_error': squared_error
                    })
                    
                    for k, note_error in enumerate(error_result['note_errors']):
                        matched_pair = error_result['matched_pairs'][k] if k < len(error_result['matched_pairs']) else (None, None, note_error)
                        all_note_details.append({
                            'file': name,
                            'note_error': note_error,
                            'estimated_note_idx': matched_pair[0],
                            'reference_note_idx': matched_pair[1]
                        })
                    
                    print(f"  ✓ {error_result['matched_count']}/{error_result['total_reference']} notas matched, erro médio = {error_result['mean_error']:.4f}")
            else:
                mean_abs_error = mean_absolute_onset_error(midi_onsets, onsets, window=0.05)
                if mean_abs_error is not None:
                    total_error += mean_abs_error
                    
                    # Calcular erro quadrático
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
                        total_squared_error += mse
                    
                    total_processed += 1
                    print(f"  ✓ erro = {mean_abs_error:.4f}")
            
        except Exception as e:
            warnings.warn(f"Erro ao processar {name}: {str(e)}")
            continue

    # Calcular médias finais
    print("\n" + "="*60)
    print("RESULTADOS FINAIS:")
    print("="*60)
    
    if total_processed > 0:
        mae = total_error / total_processed
        mse = total_squared_error / total_processed
        rmse = np.sqrt(mse)
        
        print(f"Total de arquivos processados: {total_processed}")
        print(f"MAE (erro absoluto médio): {mae:.4f}")
        print(f"MSE (erro quadrático médio): {mse:.6f}")
        print(f"RMSE (raiz do erro quadrático médio): {rmse:.4f}")
        
        if by_note and all_note_errors:
            print(f"\nDetalhes por nota: {len(all_note_errors)} notas processadas")
            print(f"MAE por nota: {np.mean(all_note_errors):.4f} ± {np.std(all_note_errors):.4f}")
            print(f"MSE por nota: {np.mean(all_squared_errors):.6f}")
            print(f"Mínimo: {np.min(all_note_errors):.4f}, Máximo: {np.max(all_note_errors):.4f}")
    else:
        print("Nenhum arquivo processado com sucesso")
        return

    # Salvar resultados
    method_suffix = 'transcription_by_note' if by_note else 'transcription_grouped'
    
    # Salvar MAE
    df_mae = pd.DataFrame([{'method': method_suffix, 'mae': mae, 'files_processed': total_processed}])
    csv_mae_path = f"{output_prefix}_{method_suffix}_mae.csv"
    df_mae.to_csv(csv_mae_path, index=False)
    
    # Salvar MSE
    df_mse = pd.DataFrame([{'method': method_suffix, 'mse': mse, 'rmse': rmse, 'files_processed': total_processed}])
    csv_mse_path = f"{output_prefix}_{method_suffix}_mse.csv"
    df_mse.to_csv(csv_mse_path, index=False)
        
    print(f"\nResultados MAE salvos em {csv_mae_path}")
    print(f"Resultados MSE salvos em {csv_mse_path}")
    
    # Salvar detalhes por nota se aplicável
    if by_note:
        if note_stats:
            detailed_csv_path = f"{output_prefix}_note_stats_transcription.csv"
            detailed_df = pd.DataFrame(note_stats)
            detailed_df.to_csv(detailed_csv_path, index=False)
            print(f"Estatísticas por arquivo salvos em {detailed_csv_path}")
        
        if all_note_details:
            all_notes_csv_path = f"{output_prefix}_all_note_errors_transcription.csv"
            all_notes_df = pd.DataFrame(all_note_details)
            all_notes_df.to_csv(all_notes_csv_path, index=False)
            print(f"Detalhes de todas as notas salvos em {all_notes_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure onset detection error using transcription method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Cálculo por onsets agrupados
  python measure_error_transcription_only.py --samples-folder ./samples \
    --transcribed-midis-folder ./transcribed_midis --model-name basic_pitch
  
  # Cálculo por nota individual
  python measure_error_transcription_only.py --by-note --samples-folder ./samples \
    --transcribed-midis-folder ./transcribed_midis --model-name bytedance
        """)
    parser.add_argument('--by-note', action='store_true', 
                        help='Calculate error per individual note instead of grouped onsets')
    parser.add_argument('--samples-folder', default='./samples',
                        help='Folder containing original MIDI files (default: ./samples)')
    parser.add_argument('--transcribed-midis-folder', default='./transcribed_midis',
                        help='Folder containing transcribed MIDI files - recursive search (default: ./transcribed_midis)')
    parser.add_argument('--output-prefix', default='onset_detection_results',
                        help='Prefix for output CSV files (default: onset_detection_results)')
    parser.add_argument('--model-name', default='basic_pitch',
                        help='Name of the transcription model used (default: basic_pitch)')
    
    args = parser.parse_args()
    main(by_note=args.by_note, 
         samples_folder=args.samples_folder,
         transcribed_midis_folder=args.transcribed_midis_folder,
         output_prefix=args.output_prefix,
         model_name=args.model_name)