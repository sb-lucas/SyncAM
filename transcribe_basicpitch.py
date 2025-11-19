#!/usr/bin/env python3
"""
Script para transcrever todos os áudios sintéticos da pasta samples_wavs usando Basic Pitch
e salvar os MIDIs transcritos organizados por tipo de instrumento.
"""

import os
import glob
import sys
from pathlib import Path
import logging
from datetime import datetime

try:
    import basic_pitch
    from basic_pitch.inference import predict_and_save, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH, build_icassp_2022_model_path, FilenameSuffix
    print("Basic Pitch importado com sucesso!")
except ImportError as e:
    print(f"Erro ao importar Basic Pitch: {e}")
    print("Verifique se o basic-pitch está instalado corretamente.")
    sys.exit(1)

def setup_logging():
    """Configurar logging para acompanhar o progresso"""
    log_filename = f"transcription_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_all_wav_files(wavs_dir):
    """
    Obter todos os arquivos .wav organizados por instrumento
    
    Args:
        wavs_dir (str): Caminho para a pasta samples_wavs
        
    Returns:
        dict: {instrumento: [lista_de_arquivos_wav]}
    """
    instruments = {}
    
    if not os.path.exists(wavs_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {wavs_dir}")
    
    # Listar todas as pastas de instrumentos
    for instrument_dir in sorted(os.listdir(wavs_dir)):
        instrument_path = os.path.join(wavs_dir, instrument_dir)
        
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

def transcribe_audio_file(audio_path, output_dir, logger, model):
    """
    Transcrever um único arquivo de áudio usando Basic Pitch
    
    Args:
        audio_path (str): Caminho para o arquivo de áudio
        output_dir (str): Diretório onde salvar o MIDI transcrito
        logger: Logger para registrar o progresso
        model: Modelo Basic Pitch pré-carregado
        
    Returns:
        bool: True se a transcrição foi bem-sucedida, False caso contrário
    """
    try:
        logger.info(f"Transcrevendo: {os.path.basename(audio_path)}")
        
        # Definir audio_basename no início da função
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        expected_midi = os.path.join(output_dir, f"{audio_basename}_basic_pitch.mid")
        
        if os.path.exists(expected_midi):
            logger.info(f"Arquivo já transcrito, pulando: {audio_path}")
            return True
            
        # Usar predict_and_save que é mais simples e robusta
        predict_and_save(
            audio_path_list=[audio_path],
            output_directory=output_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=model
        )
        
        # Verificar se o arquivo foi criado
        if os.path.exists(expected_midi):
            logger.info(f"MIDI salvo: {expected_midi}")
            return True
        else:
            logger.error(f"MIDI não foi criado: {expected_midi}")
            return False
        
    except Exception as e:
        logger.error(f"Erro ao transcrever {audio_path}: {str(e)}")
        return False

def main():
    """Função principal"""
    logger = setup_logging()
    import argparse
    parser = argparse.ArgumentParser(description="Transcrever áudios sintéticos usando Basic Pitch")
    parser.add_argument("wavs_dir", type=str, help="Diretório contendo arquivos WAV sintéticos")
    parser.add_argument("--output_dir", type=str, default="transcribed_midis", help="Diretório para salvar os MIDIs transcritos")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo para execução do modelo (cpu, cuda, ou cuda:0)")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID da GPU para usar quando device=cuda (padrão: 0)")
    parser.add_argument("--batch_size", type=int, default=1, help="Número de arquivos para processar por lote (experimental)")
    args = parser.parse_args()
    wavs_dir = args.wavs_dir
    output_base_dir = args.output_dir
    device = args.device
    gpu_id = args.gpu_id
    batch_size = args.batch_size
    # # Configurações
    # wavs_dir = "samples_wavs"
    # output_base_dir = "transcribed_midis"
    
    logger.info("Iniciando transcrição em lote com Basic Pitch")
    logger.info(f"Diretório de entrada: {wavs_dir}")
    logger.info(f"Diretório de saída: {output_base_dir}")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Tamanho do lote: {batch_size}")
    
    # Configurar dispositivo TensorFlow
    logger.info("Configurando dispositivos...")
    
    # Configurar GPU específica se solicitado
    if device.startswith("cuda"):
        # Configurar variável de ambiente para GPU específica
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Configurando GPU {gpu_id} como única GPU visível")
        
        try:
            import tensorflow as tf
            # Suprimir warnings do TensorFlow
            tf.get_logger().setLevel('ERROR')
            
            # Verificar se CUDA/GPU está realmente disponível
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                logger.info(f"✅ GPU(s) detectada(s) pelo TensorFlow: {len(physical_devices)}")
                
                # Configurar crescimento de memória para evitar OOM
                for gpu_device in physical_devices:
                    try:
                        tf.config.experimental.set_memory_growth(gpu_device, True)
                        logger.info(f"Crescimento de memória configurado para {gpu_device}")
                    except RuntimeError as e:
                        logger.warning(f"Erro ao configurar crescimento de memória: {e}")
                        
                device_status = "cuda"
            else:
                logger.warning("⚠️ Nenhuma GPU detectada pelo TensorFlow")
                logger.warning("🔍 Diagnóstico:")
                logger.warning(f"   • TensorFlow version: {tf.__version__}")
                logger.warning(f"   • CUDA built: {tf.test.is_built_with_cuda()}")
                logger.warning("   • Possíveis causas:")
                logger.warning("     - Drivers CUDA/cuDNN incompatíveis com TensorFlow")
                logger.warning("     - TensorFlow CPU-only instalado")
                logger.warning("     - GPUs ocupadas por outros processos")
                logger.warning("")
                logger.warning("🔄 Continuando com CPU (pode ser mais lento)...")
                
                # Forçar CPU se GPU não disponível
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                device_status = "cpu_fallback"
                
        except ImportError:
            logger.error("❌ TensorFlow não encontrado!")
            device_status = "cpu_fallback"
        except Exception as e:
            logger.error(f"❌ Erro ao configurar GPU: {e}")
            device_status = "cpu_fallback"
            
    elif device == "cpu":
        # Forçar uso apenas de CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("🔧 Forçando uso exclusivo de CPU")
        device_status = "cpu"
        
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            logger.info("✅ CPU configurada para uso!")
        except ImportError:
            logger.error("❌ TensorFlow não encontrado!")
            
    else:
        logger.warning(f"⚠️ Dispositivo '{device}' não reconhecido, usando configuração padrão")
        device_status = "default"
    
    logger.info(f"📊 Status final do dispositivo: {device_status}")
    
    # Adicionar informações sobre performance esperada
    if device_status in ["cpu", "cpu_fallback"]:
        logger.info("💡 Dica: Para melhorar a performance:")
        logger.info("   • Use --batch_size maior (ex: --batch_size 3)")
        logger.info("   • Considere instalar tensorflow-gpu se você tem GPUs NVIDIA")
        logger.info("   • Verifique se não há conflitos de drivers CUDA")
    
    # Criar estrutura de saída
    create_output_structure(output_base_dir)
    
    # Carregar modelo uma única vez para reutilizar
    logger.info("Carregando modelo Basic Pitch...")
    model = None
    
    # Lista de formatos para tentar, em ordem de preferência (ONNX primeiro)
    model_formats = [
        (FilenameSuffix.onnx, "ONNX (Mais rápido)"),
        (FilenameSuffix.tflite, "TensorFlow Lite"),
        (FilenameSuffix.tf, "TensorFlow"),
    ]
    
    for format_suffix, format_name in model_formats:
        try:
            model_path = build_icassp_2022_model_path(format_suffix)
            logger.info(f"Tentando carregar modelo {format_name}: {model_path}")
            
            if os.path.exists(model_path):
                model = Model(model_path)
                logger.info(f"✅ Modelo {format_name} carregado com sucesso!")
                break
            else:
                logger.warning(f"❌ Arquivo de modelo {format_name} não encontrado: {model_path}")
                
        except Exception as e:
            logger.warning(f"❌ Erro ao carregar modelo {format_name}: {e}")
            continue
    
    if model is None:
        # Fallback para o caminho padrão
        try:
            logger.info("Tentando caminho padrão como último recurso...")
            model = Model(ICASSP_2022_MODEL_PATH)
            logger.info("✅ Modelo padrão carregado com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro crítico: Não foi possível carregar nenhum modelo: {e}")
            logger.error("Verifique se o basic-pitch foi instalado corretamente")
            logger.error("Tente reinstalar: pip install basic-pitch")
            return
    
    # Obter todos os arquivos WAV organizados por instrumento
    try:
        instruments = get_all_wav_files(wavs_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if not instruments:
        logger.warning("Nenhum instrumento com arquivos .wav encontrado!")
        return
    
    # Estatísticas
    total_files = sum(len(files) for files in instruments.values())
    processed_files = 0
    successful_transcriptions = 0
    failed_transcriptions = 0
    
    logger.info(f"Total de {total_files} arquivos para transcrever em {len(instruments)} instrumentos")
    
    # Processar cada instrumento
    for instrument, wav_files in instruments.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando instrumento: {instrument}")
        logger.info(f"Arquivos a processar: {len(wav_files)}")
        logger.info(f"{'='*60}")
        
        # Criar diretório para este instrumento
        instrument_output_dir = os.path.join(output_base_dir, instrument)
        Path(instrument_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Processar cada arquivo WAV
        i = 0
        while i < len(wav_files):
            wav_file = wav_files[i]
            # Verificar se já foi transcrito (basic-pitch usa sufixo _basic_pitch.mid)
            wav_basename = os.path.splitext(os.path.basename(wav_file))[0]
            expected_midi = os.path.join(instrument_output_dir, f"{wav_basename}_basic_pitch.mid")
            
            # Verificar se já foi transcrito
            if os.path.exists(expected_midi):
                logger.info(f"[{i+1}/{len(wav_files)}] Pulando (já existe): {os.path.basename(expected_midi)}")
                processed_files += 1
                successful_transcriptions += 1
                i += 1
                continue
            
            # Processar em lotes se batch_size > 1
            if batch_size > 1:
                # Coletar arquivos para o lote
                batch_files = []
                batch_indices = []
                
                for j in range(i, min(i + batch_size, len(wav_files))):
                    batch_wav = wav_files[j]
                    batch_basename = os.path.splitext(os.path.basename(batch_wav))[0]
                    batch_expected = os.path.join(instrument_output_dir, f"{batch_basename}_basic_pitch.mid")
                    
                    if not os.path.exists(batch_expected):
                        batch_files.append(batch_wav)
                        batch_indices.append(j+1)
                
                if batch_files:
                    logger.info(f"Processando lote de {len(batch_files)} arquivos: {[os.path.basename(f) for f in batch_files[:3]]}{'...' if len(batch_files) > 3 else ''}")
                    
                    # Processar lote
                    try:
                        predict_and_save(
                            audio_path_list=batch_files,
                            output_directory=instrument_output_dir,
                            save_midi=True,
                            sonify_midi=False,
                            save_model_outputs=False,
                            save_notes=False,
                            model_or_model_path=model
                        )
                        
                        # Verificar sucessos
                        batch_successes = 0
                        for batch_wav in batch_files:
                            batch_basename = os.path.splitext(os.path.basename(batch_wav))[0]
                            batch_expected = os.path.join(instrument_output_dir, f"{batch_basename}_basic_pitch.mid")
                            if os.path.exists(batch_expected):
                                batch_successes += 1
                                logger.debug(f"MIDI salvo: {batch_expected}")
                            else:
                                logger.error(f"MIDI não foi criado: {batch_expected}")
                        
                        successful_transcriptions += batch_successes
                        failed_transcriptions += len(batch_files) - batch_successes
                        logger.info(f"Lote concluído: {batch_successes}/{len(batch_files)} sucessos")
                        
                    except Exception as e:
                        logger.error(f"Erro no processamento do lote: {e}")
                        failed_transcriptions += len(batch_files)
                    
                    processed_files += len(batch_files)
                
                # Avançar pelo tamanho do lote
                i += batch_size
                continue
            
            # Processamento individual (batch_size == 1)
            logger.info(f"[{i+1}/{len(wav_files)}] Processando: {os.path.basename(wav_file)}")
            success = transcribe_audio_file(wav_file, instrument_output_dir, logger, model)
            
            processed_files += 1
            if success:
                successful_transcriptions += 1
            else:
                failed_transcriptions += 1
            
            # Avançar para o próximo arquivo
            i += 1
            
            # Exibir progresso geral
            overall_progress = (processed_files / total_files) * 100
            logger.info(f"Progresso geral: {processed_files}/{total_files} ({overall_progress:.1f}%)")
    
    # Relatório final
    logger.info(f"\n{'='*60}")
    logger.info("RELATÓRIO FINAL")
    logger.info(f"{'='*60}")
    logger.info(f"Total de arquivos processados: {processed_files}")
    logger.info(f"Transcrições bem-sucedidas: {successful_transcriptions}")
    logger.info(f"Transcrições com falha: {failed_transcriptions}")
    logger.info(f"Taxa de sucesso: {(successful_transcriptions/processed_files)*100:.1f}%" if processed_files > 0 else "N/A")
    logger.info(f"Arquivos salvos em: {os.path.abspath(output_base_dir)}")

if __name__ == "__main__":
    main()