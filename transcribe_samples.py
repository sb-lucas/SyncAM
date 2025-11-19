#!/usr/bin/env python3
"""
Script para transcrever todos os áudios sintéticos da pasta samples_wavs usando Basic Pitch
e salvar os MIDIs transcritos organizados por tipo de instrumento.
"""

import os
import glob
import sys
from pathlib import Path
from datetime import datetime

try:
    import basic_pitch
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH
    print("Basic Pitch importado com sucesso!")
except ImportError as e:
    print(f"Erro ao importar Basic Pitch: {e}")
    print("Verifique se o basic-pitch está instalado corretamente.")
    sys.exit(1)

def get_all_wav_files(samples_dir):
    """
    Obter todos os arquivos .wav organizados por instrumento
    
    Args:
        samples_dir (str): Caminho para a pasta samples_wavs
        
    Returns:
        dict: {instrumento: [lista_de_arquivos_wav]}
    """
    instruments = {}
    
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {samples_dir}")
    
    # Listar todas as pastas de instrumentos
    for instrument_dir in sorted(os.listdir(samples_dir)):
        instrument_path = os.path.join(samples_dir, instrument_dir)
        
        if os.path.isdir(instrument_path):
            # Encontrar todos os arquivos .wav nesta pasta
            wav_pattern = os.path.join(instrument_path, "*.wav")
            wav_files = glob.glob(wav_pattern)
            
            if wav_files:
                instruments[instrument_dir] = sorted(wav_files)
                print(f"Encontrados {len(wav_files)} arquivos .wav para {instrument_dir}")
            else:
                print(f"Nenhum arquivo .wav encontrado para {instrument_dir}")
    
    return instruments

def create_output_structure(base_output_dir):
    """
    Criar estrutura de diretórios para os MIDIs transcritos
    
    Args:
        base_output_dir (str): Diretório base de saída
    """
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Diretório de saída criado: {base_output_dir}")

def transcribe_audio_file(audio_path, output_dir):
    """
    Transcrever um único arquivo de áudio usando Basic Pitch
    
    Args:
        audio_path (str): Caminho para o arquivo de áudio
        output_dir (str): Diretório onde salvar o MIDI transcrito
        
    Returns:
        bool: True se a transcrição foi bem-sucedida, False caso contrário
    """
    try:
        print(f"Transcrevendo: {os.path.basename(audio_path)}")
        
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        expected_midi = os.path.join(output_dir, f"{audio_basename}_basic_pitch.mid")
        print(expected_midi, os.path.exists(expected_midi))
        if os.path.exists(expected_midi):
            print(f"MIDI já existe, pulando: {expected_midi}")
            return True
        # Usar predict_and_save que é mais simples e robusta
        predict_and_save(
            audio_path_list=[audio_path],
            output_directory=output_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )
        
        # Verificar se o arquivo foi criado        
        if os.path.exists(expected_midi):
            print(f"MIDI salvo: {expected_midi}")
            return True
        else:
            print(f"MIDI não foi criado: {expected_midi}")
            return False
        
    except Exception as e:
        print(f"Erro ao transcrever {audio_path}: {str(e)}")
        return False

def main():
    """Função principal"""
    from multiprocessing import Pool
    # Configurações
    samples_dir = "samples_wavs"
    output_base_dir = "transcribed_midis"
    workers = 4  # Número de workers paralelos
    
    print("Iniciando transcrição em lote com Basic Pitch")
    print(f"Diretório de entrada: {samples_dir}")
    print(f"Diretório de saída: {output_base_dir}")
    
    # Criar estrutura de saída
    create_output_structure(output_base_dir)
    
    # Obter todos os arquivos WAV organizados por instrumento
    try:
        instruments = get_all_wav_files(samples_dir)
    except FileNotFoundError as e:
        print(str(e))
        return
    
    if not instruments:
        print("Nenhum instrumento com arquivos .wav encontrado!")
        return
    
    # Estatísticas
    total_files = sum(len(files) for files in instruments.values())
    processed_files = 0
    successful_transcriptions = 0
    failed_transcriptions = 0
    
    print(f"Total de {total_files} arquivos para transcrever em {len(instruments)} instrumentos")
    
    if workers > 1:
        pool = Pool(processes=workers)
        results = []
        # Preparar lista de tarefas para processamento paralelo
        tasks = []
        for instrument, wav_files in instruments.items():
            instrument_output_dir = os.path.join(output_base_dir, instrument)
            Path(instrument_output_dir).mkdir(parents=True, exist_ok=True)
            
            for wav_file in wav_files:
                wav_basename = os.path.splitext(os.path.basename(wav_file))[0]
                expected_midi = os.path.join(instrument_output_dir, f"{wav_basename}_basic_pitch.mid")
                
                # Só adicionar à lista se não foi transcrito ainda
                if not os.path.exists(expected_midi):
                    tasks.append((wav_file, instrument_output_dir))
                else:
                    successful_transcriptions += 1
                    processed_files += 1
            
        # Executar processamento paralelo
        for wav_file, output_dir in tasks:
            result = pool.apply_async(transcribe_audio_file, (wav_file, output_dir))
            results.append((wav_file, result))
        
        # Aguardar todos os resultados
        for wav_file, result in results:
            try:
                success = result.get(timeout=300)  # timeout de 5 minutos por arquivo
                processed_files += 1
                if success:
                    successful_transcriptions += 1
                else:
                    failed_transcriptions += 1
                
                overall_progress = (processed_files / total_files) * 100
                print(f"Concluído: {os.path.basename(wav_file)} - Progresso: {processed_files}/{total_files} ({overall_progress:.1f}%)")
            except Exception as e:
                print(f"Erro no processamento de {wav_file}: {e}")
                failed_transcriptions += 1
                processed_files += 1
        
        pool.close()
        pool.join()
    else:
        # Processar cada instrumento
        for instrument, wav_files in instruments.items():
            print(f"\n{'='*60}")
            print(f"Processando instrumento: {instrument}")
            print(f"Arquivos a processar: {len(wav_files)}")
            print(f"{'='*60}")
            
            # Criar diretório para este instrumento
            instrument_output_dir = os.path.join(output_base_dir, instrument)
            Path(instrument_output_dir).mkdir(parents=True, exist_ok=True)
            
            # Processar cada arquivo WAV
            for i, wav_file in enumerate(wav_files, 1):
                # Verificar se já foi transcrito (basic-pitch usa sufixo _basic_pitch.mid)
                wav_basename = os.path.splitext(os.path.basename(wav_file))[0]
                expected_midi = os.path.join(instrument_output_dir, f"{wav_basename}_basic_pitch.mid")
                
                # Verificar se já foi transcrito
                if os.path.exists(expected_midi):
                    print(f"[{i}/{len(wav_files)}] Pulando (já existe): {os.path.basename(expected_midi)}")
                    processed_files += 1
                    successful_transcriptions += 1
                    continue
                
                # Transcrever
                print(f"[{i}/{len(wav_files)}] Processando: {os.path.basename(wav_file)}")
                success = transcribe_audio_file(wav_file, instrument_output_dir)
                
                processed_files += 1
                if success:
                    successful_transcriptions += 1
                else:
                    failed_transcriptions += 1
                
                # Exibir progresso geral
                overall_progress = (processed_files / total_files) * 100
                print(f"Progresso geral: {processed_files}/{total_files} ({overall_progress:.1f}%)")
        
    # Relatório final
    print(f"\n{'='*60}")
    print("RELATÓRIO FINAL")
    print(f"{'='*60}")
    print(f"Total de arquivos processados: {processed_files}")
    print(f"Transcrições bem-sucedidas: {successful_transcriptions}")
    print(f"Transcrições com falha: {failed_transcriptions}")
    print(f"Taxa de sucesso: {(successful_transcriptions/processed_files)*100:.1f}%" if processed_files > 0 else "N/A")
    print(f"Arquivos salvos em: {os.path.abspath(output_base_dir)}")

if __name__ == "__main__":
    main()