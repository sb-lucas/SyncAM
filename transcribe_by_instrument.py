#!/usr/bin/env python3
"""
Script simplificado para transcrever um instrumento por vez usando Basic Pitch
"""

import os
import sys
from pathlib import Path

try:
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH
    print("Basic Pitch importado com sucesso!")
except ImportError as e:
    print(f"Erro ao importar Basic Pitch: {e}")
    sys.exit(1)

def transcribe_instrument(instrument_name, max_files=None):
    """
    Transcrever todos os arquivos de um instrumento específico
    
    Args:
        instrument_name (str): Nome do instrumento (pasta)
        max_files (int): Máximo de arquivos a processar (None = todos)
    """
    samples_dir = f"samples_wavs/{instrument_name}"
    output_dir = f"transcribed_midis/{instrument_name}"
    
    if not os.path.exists(samples_dir):
        print(f"Diretório não encontrado: {samples_dir}")
        return
    
    # Criar diretório de saída
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Listar arquivos WAV
    wav_files = [f for f in os.listdir(samples_dir) if f.endswith('.wav')]
    
    if max_files:
        wav_files = wav_files[:max_files]
    
    print(f"\n{instrument_name}: {len(wav_files)} arquivos")
    print("=" * 60)
    
    for i, wav_file in enumerate(wav_files, 1):
        # Verificar se já foi processado
        expected_midi = os.path.join(output_dir, f"{wav_file[:-4]}_basic_pitch.mid")
        if os.path.exists(expected_midi):
            print(f"[{i}/{len(wav_files)}] PULANDO: {wav_file} (já existe)")
            continue
        
        print(f"[{i}/{len(wav_files)}] PROCESSANDO: {wav_file}")
        
        try:
            audio_path = os.path.join(samples_dir, wav_file)
            
            # Transcrever
            predict_and_save(
                audio_path_list=[audio_path],
                output_directory=output_dir,
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH
            )
            
            # Verificar se foi criado
            if os.path.exists(expected_midi):
                file_size = os.path.getsize(expected_midi)
                print(f"[{i}/{len(wav_files)}] SUCESSO: {os.path.basename(expected_midi)} ({file_size} bytes)")
            else:
                print(f"[{i}/{len(wav_files)}] ERRO: Arquivo não foi criado")
                
        except Exception as e:
            print(f"[{i}/{len(wav_files)}] ERRO: {e}")

def main():
    instruments = [
        'acoustic_bass', 'acoustic_grand_piano', 'acoustic_nylon_guitar',
        'bright_acoustic_piano', 'cello', 'chorused_piano', 'clavinet',
        'electric_clean_guitar', 'electric_grand_piano', 'harpsichord',
        'honky-tonk_piano', 'overdriven_guitar', 'rhodes_piano',
        'trombone', 'trumpet', 'violin'
    ]
    
    # Processar um instrumento específico ou todos
    if len(sys.argv) > 1:
        instrument = sys.argv[1]
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
        if instrument in instruments:
            transcribe_instrument(instrument, max_files)
        else:
            print(f"Instrumento inválido. Opções: {', '.join(instruments)}")
    else:
        # Processar todos os instrumentos, um por vez
        for instrument in instruments:
            try:
                transcribe_instrument(instrument)
            except KeyboardInterrupt:
                print(f"\nInterrompido durante processamento de {instrument}")
                break
            except Exception as e:
                print(f"Erro ao processar {instrument}: {e}")
                continue

if __name__ == "__main__":
    main()