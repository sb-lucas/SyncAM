#!/usr/bin/env python3
"""
Script de teste para verificar se o basic-pitch está funcionando corretamente
"""

import os
import sys

try:
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH
    print("Basic Pitch importado com sucesso!")
except ImportError as e:
    print(f"Erro ao importar Basic Pitch: {e}")
    sys.exit(1)

# Testar com um arquivo
test_file = "samples_wavs/acoustic_grand_piano/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
output_dir = "test_output"

if os.path.exists(test_file):
    print(f"Testando transcrição de: {test_file}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        predict_and_save(
            audio_path_list=[test_file],
            output_directory=output_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )
        print("Teste concluído com sucesso!")
        
        # Verificar se o arquivo foi criado
        expected_file = os.path.join(output_dir, "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_basic_pitch.mid")
        if os.path.exists(expected_file):
            print(f"MIDI criado: {expected_file}")
        else:
            print("MIDI não foi criado")
            
    except Exception as e:
        print(f"Erro no teste: {e}")
else:
    print(f"Arquivo de teste não encontrado: {test_file}")