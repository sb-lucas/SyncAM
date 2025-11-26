#!/usr/bin/env python3
"""
Script para comparar MIDIs transcritos com MIDIs originais usando onset_detection.py
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

try:
    import pretty_midi
    from onset_detection import (
        get_onsets_from_midi, 
        get_onsets_from_midi_by_note,
        get_onsets_transcription
    )
    from metric import mean_absolute_onset_error, note_based_onset_error
    print("✅ Bibliotecas importadas com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar bibliotecas: {e}")
    sys.exit(1)

def find_matching_original_midi(transcribed_midi_path, original_midi_dir):
    """
    Encontrar arquivo MIDI original correspondente ao transcrito
    
    Args:
        transcribed_midi_path: Caminho para o MIDI transcrito
        original_midi_dir: Diretório contendo MIDIs originais
        
    Returns:
        str: Caminho para o MIDI original ou None
    """
    transcribed_name = Path(transcribed_midi_path).stem
    
    # Remover sufixos comuns (_filtered, _basic_pitch, etc.)
    base_name = transcribed_name
    for suffix in ['_filtered', '_basic_pitch', '_transcribed']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
    
    # Tentar diferentes extensões
    possible_extensions = ['.mid', '.midi', '.MID', '.MIDI']
    original_midi_path = Path(original_midi_dir)
    
    for ext in possible_extensions:
        midi_path = original_midi_path / f"{base_name}{ext}"
        if midi_path.exists():
            return str(midi_path)
    
    # Tentar busca recursiva
    for ext in possible_extensions:
        for midi_file in original_midi_path.rglob(f"{base_name}{ext}"):
            return str(midi_file)
    
    return None

def compare_midi_files(transcribed_midi_path, original_midi_path, by_note=False, time_tolerance=0.05):
    """
    Comparar dois arquivos MIDI e calcular métricas de erro
    
    Args:
        transcribed_midi_path: Caminho para MIDI transcrito
        original_midi_path: Caminho para MIDI original
        by_note: Se True, calcular erro por nota individual
        time_tolerance: Tolerância temporal para matching
        
    Returns:
        dict: Dicionário com métricas de erro
    """
    try:
        # Carregar MIDIs
        transcribed_midi = pretty_midi.PrettyMIDI(transcribed_midi_path)
        original_midi = pretty_midi.PrettyMIDI(original_midi_path)
        
        # Extrair onsets
        if by_note:
            original_onsets = get_onsets_from_midi_by_note(original_midi, combine=0.03)
            transcribed_onsets = get_onsets_from_midi_by_note(transcribed_midi, combine=0.03)
            
            # Calcular erro por nota
            error_result = note_based_onset_error(
                original_onsets, 
                transcribed_onsets, 
                window=time_tolerance
            )
            
            # Calcular métricas adicionais
            matched_count = error_result['matched_count']
            total_reference = error_result['total_reference']
            total_estimated = error_result['total_estimated']
            
            precision = matched_count / total_estimated if total_estimated > 0 else 0.0
            recall = matched_count / total_reference if total_reference > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            false_positives = total_estimated - matched_count
            false_negatives = total_reference - matched_count
            false_positive_rate = false_positives / total_estimated if total_estimated > 0 else 0.0
            false_negative_rate = false_negatives / total_reference if total_reference > 0 else 0.0
            
            return {
                'mean_error': error_result['mean_error'],
                'matched_count': matched_count,
                'total_reference': total_reference,
                'total_estimated': total_estimated,
                'note_errors': error_result['note_errors'],
                'matched_pairs': error_result.get('matched_pairs', []),
                # Métricas adicionais
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate
            }
        else:
            original_onsets = get_onsets_from_midi(original_midi, combine=0.03)
            transcribed_onsets = get_onsets_from_midi(transcribed_midi, combine=0.03)
            
            # Calcular erro agrupado
            mean_error = mean_absolute_onset_error(
                original_onsets,
                transcribed_onsets,
                window=time_tolerance
            )
            
            # Calcular métricas adicionais
            total_reference = len(original_onsets)
            total_estimated = len(transcribed_onsets)
            matched_count = len(original_onsets) if mean_error is not None else 0
            
            precision = matched_count / total_estimated if total_estimated > 0 else 0.0
            recall = matched_count / total_reference if total_reference > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            false_positives = total_estimated - matched_count
            false_negatives = total_reference - matched_count
            false_positive_rate = false_positives / total_estimated if total_estimated > 0 else 0.0
            false_negative_rate = false_negatives / total_reference if total_reference > 0 else 0.0
            
            return {
                'mean_error': mean_error,
                'total_reference': total_reference,
                'total_estimated': total_estimated,
                'matched_count': matched_count,
                # Métricas adicionais
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate
            }
            
    except Exception as e:
        warnings.warn(f"Erro ao comparar {transcribed_midi_path}: {e}")
        return None

def process_directory(transcribed_dir, original_midi_dir, by_note=False, time_tolerance=0.05, output_csv=None):
    """
    Processar todos os MIDIs transcritos em um diretório
    
    Args:
        transcribed_dir: Diretório contendo MIDIs transcritos
        original_midi_dir: Diretório contendo MIDIs originais
        by_note: Se True, calcular erro por nota individual
        time_tolerance: Tolerância temporal para matching
        output_csv: Caminho para salvar CSV com resultados (opcional)
    """
    transcribed_path = Path(transcribed_dir)
    original_midi_path = Path(original_midi_dir)
    
    if not transcribed_path.exists():
        print(f"❌ Diretório não encontrado: {transcribed_dir}")
        return
    
    if not original_midi_path.exists():
        print(f"❌ Diretório de MIDIs originais não encontrado: {original_midi_dir}")
        return
    
    # Encontrar todos os arquivos MIDI transcritos
    midi_files = list(transcribed_path.rglob("*.mid")) + list(transcribed_path.rglob("*.midi"))
    
    if not midi_files:
        print(f"⚠️ Nenhum arquivo MIDI encontrado em {transcribed_dir}")
        return
    
    print(f"📁 Encontrados {len(midi_files)} arquivos MIDI transcritos")
    print(f"📊 Modo: {'Por nota individual' if by_note else 'Por onsets agrupados'}")
    print("=" * 60)
    
    results = []
    total_processed = 0
    total_errors = []
    
    for i, transcribed_midi_file in enumerate(sorted(midi_files), 1):
        # Encontrar MIDI original correspondente
        original_midi_file = find_matching_original_midi(transcribed_midi_file, original_midi_path)
        
        if original_midi_file is None:
            print(f"[{i}/{len(midi_files)}] ⚠️ MIDI original não encontrado para: {transcribed_midi_file.name}")
            continue
        
        print(f"[{i}/{len(midi_files)}] Processando: {transcribed_midi_file.name}")
        
        # Comparar MIDIs
        error_result = compare_midi_files(
            str(transcribed_midi_file),
            original_midi_file,
            by_note=by_note,
            time_tolerance=time_tolerance
        )
        
        if error_result is None:
            print(f"  ❌ Erro ao processar")
            continue
        
        # Calcular caminho relativo para identificação
        try:
            rel_path = transcribed_midi_file.relative_to(transcribed_path)
        except ValueError:
            rel_path = transcribed_midi_file.name
        
        # Armazenar resultados
        result_row = {
            'file': transcribed_midi_file.name,
            'path': str(rel_path),
            'original_file': Path(original_midi_file).name,
            'mean_error': error_result['mean_error'],
            'matched_count': error_result['matched_count'],
            'total_reference': error_result['total_reference'],
            'total_estimated': error_result['total_estimated']
        }
        
        if by_note and 'note_errors' in error_result:
            result_row['num_note_errors'] = len(error_result['note_errors'])
            if error_result['note_errors']:
                result_row['min_error'] = np.min(error_result['note_errors'])
                result_row['max_error'] = np.max(error_result['note_errors'])
                result_row['std_error'] = np.std(error_result['note_errors'])
        
        # Adicionar métricas adicionais
        result_row['precision'] = error_result.get('precision', 0.0)
        result_row['recall'] = error_result.get('recall', 0.0)
        result_row['f1_score'] = error_result.get('f1_score', 0.0)
        result_row['false_positives'] = error_result.get('false_positives', 0)
        result_row['false_negatives'] = error_result.get('false_negatives', 0)
        result_row['false_positive_rate'] = error_result.get('false_positive_rate', 0.0)
        result_row['false_negative_rate'] = error_result.get('false_negative_rate', 0.0)
        
        results.append(result_row)
        
        # Estatísticas
        if error_result['mean_error'] is not None:
            total_errors.append(error_result['mean_error'])
            total_processed += 1
            print(f"  ✅ Erro médio: {error_result['mean_error']:.4f}s")
            print(f"     Matched: {error_result['matched_count']}/{error_result['total_reference']} (ref) vs {error_result['total_estimated']} (est)")
            print(f"     Precision: {error_result.get('precision', 0):.2%} | Recall: {error_result.get('recall', 0):.2%} | F1: {error_result.get('f1_score', 0):.2%}")
            print(f"     Falsos Positivos: {error_result.get('false_positives', 0)} | Falsos Negativos: {error_result.get('false_negatives', 0)}")
        else:
            print(f"  ⚠️ Sem matches suficientes")
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"Total processado: {total_processed}/{len(midi_files)}")
    
    if total_errors:
        print(f"Erro médio geral: {np.mean(total_errors):.4f}s ± {np.std(total_errors):.4f}s")
        print(f"Erro mínimo: {np.min(total_errors):.4f}s")
        print(f"Erro máximo: {np.max(total_errors):.4f}s")
    
    # Salvar CSV se solicitado
    if output_csv and results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Resultados salvos em: {output_csv}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Comparar MIDIs transcritos com MIDIs originais"
    )
    parser.add_argument(
        'transcribed_dir',
        type=str,
        help='Diretório contendo MIDIs transcritos (ex: data/transcribed/audio_original ou data/transcribed/sintetizados/FluidR3_GM/acoustic_grand_piano)'
    )
    parser.add_argument(
        '--original_midi_dir',
        type=str,
        default='data/midi',
        help='Diretório contendo MIDIs originais (padrão: data/midi)'
    )
    parser.add_argument(
        '--by-note',
        action='store_true',
        help='Calcular erro por nota individual (em vez de onsets agrupados)'
    )
    parser.add_argument(
        '--time-tolerance',
        type=float,
        default=0.05,
        help='Tolerância temporal para matching em segundos (padrão: 0.05)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Caminho para salvar CSV com resultados (opcional)'
    )
    
    args = parser.parse_args()
    
    # Resolver caminhos absolutos se necessário
    base_dir = Path(__file__).parent
    transcribed_dir = base_dir / args.transcribed_dir if not Path(args.transcribed_dir).is_absolute() else Path(args.transcribed_dir)
    original_midi_dir = base_dir / args.original_midi_dir if not Path(args.original_midi_dir).is_absolute() else Path(args.original_midi_dir)
    
    print("=" * 60)
    print("COMPARAÇÃO DE MIDIs TRANSCRITOS vs ORIGINAIS")
    print("=" * 60)
    print(f"Diretório transcritos: {transcribed_dir}")
    print(f"Diretório originais: {original_midi_dir}")
    print(f"Modo: {'Por nota' if args.by_note else 'Por onsets agrupados'}")
    print(f"Tolerância temporal: {args.time_tolerance}s")
    print("=" * 60)
    print()
    
    # Processar
    results = process_directory(
        str(transcribed_dir),
        str(original_midi_dir),
        by_note=args.by_note,
        time_tolerance=args.time_tolerance,
        output_csv=args.output_csv
    )
    
    if results:
        print(f"\n✅ Processamento concluído! {len(results)} arquivos processados.")
    else:
        print("\n⚠️ Nenhum resultado obtido.")

if __name__ == "__main__":
    main()

