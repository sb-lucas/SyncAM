#!/usr/bin/env python3
"""
Script para transcrever áudios usando Basic Pitch.

Modo normal (sem --recursive):
    Transcreve apenas arquivos de áudio na pasta especificada (sem subpastas).
    
    Exemplo:
        python transcribe_basicpitch.py /caminho/audio --output_dir midis_transcritos

Modo recursivo (com --recursive):
    Busca recursivamente por arquivos de áudio em todas as subpastas,
    mantendo a estrutura de diretórios na saída.
    
    Exemplo:
        python transcribe_basicpitch.py /caminho/audio --output_dir midis_transcritos --recursive

Formatos suportados: .wav, .mp3, .flac, .m4a, .ogg, .aac, .wma
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
            # logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

<<<<<<< HEAD
def get_all_wav_files(wavs_dir, base_dir=None):
    """
    Obter todos os arquivos de áudio organizados por caminho relativo (recursivamente)
    
    Args:
        wavs_dir (str): Caminho para a pasta base a processar
        base_dir (str): Diretório base para manter estrutura relativa (opcional)
        
    Returns:
        dict: {caminho_relativo: [lista_de_arquivos_audio]}
=======
def get_all_audio_files(audio_dir, recursive=False):
    """
    Obter todos os arquivos de áudio (vários formatos)
    
    Args:
        audio_dir (str): Caminho para a pasta com arquivos de áudio
        recursive (bool): Se True, busca recursivamente em subpastas
        
    Returns:
        list: Lista de tuplas (caminho_absoluto, caminho_relativo)
>>>>>>> b6800e367d27a1dac8049db0d43a39d60b7adee4
    """
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {audio_dir}")
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    audio_files = []
    
<<<<<<< HEAD
    # Se base_dir não fornecido, usar wavs_dir como base
    if base_dir is None:
        base_dir = wavs_dir
    
    wavs_dir_path = Path(wavs_dir)
    base_dir_path = Path(base_dir)
    
    # Buscar recursivamente todos os arquivos de áudio
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    for audio_file in sorted(wavs_dir_path.rglob("*")):
        if audio_file.is_file():
            # Verificar se é um arquivo de áudio
            if any(audio_file.suffix.lower() == ext.lower() for ext in audio_extensions):
                # Calcular caminho relativo ao diretório base
                try:
                    rel_path = audio_file.parent.relative_to(base_dir_path)
                except ValueError:
                    # Se não é relativo ao base_dir, usar relativo ao wavs_dir
                    rel_path = audio_file.parent.relative_to(wavs_dir_path)
                
                # Usar caminho relativo como chave (ou "." se for o diretório raiz)
                if str(rel_path) == ".":
                    key = wavs_dir_path.name
                else:
                    key = str(rel_path)
                
                if key not in instruments:
                    instruments[key] = []
                instruments[key].append(str(audio_file))
    
    # Ordenar arquivos em cada grupo
    for key in instruments:
        instruments[key] = sorted(instruments[key])
        print(f"Encontrados {len(instruments[key])} arquivos de áudio em '{key}'")
=======
    if recursive:
        # Busca recursiva
        for root, dirs, files in os.walk(audio_dir):
            for file in sorted(files):
                if Path(file).suffix.lower() in audio_extensions:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, audio_dir)
                    audio_files.append((abs_path, rel_path))
    else:
        # Apenas pasta raiz
        for file in sorted(os.listdir(audio_dir)):
            file_path = os.path.join(audio_dir, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in audio_extensions:
                audio_files.append((file_path, file))
>>>>>>> b6800e367d27a1dac8049db0d43a39d60b7adee4
    
    return audio_files

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
    parser = argparse.ArgumentParser(
        description="Transcrever áudios usando Basic Pitch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Transcrever apenas arquivos da pasta raiz
  python transcribe_basicpitch.py /caminho/audio --output_dir midis
  
  # Transcrever recursivamente mantendo estrutura
  python transcribe_basicpitch.py /caminho/audio --output_dir midis --recursive
  
  # Usar GPU
  python transcribe_basicpitch.py /caminho/audio --device cuda --gpu_id 0 --recursive
        """)
    parser.add_argument("audio_dir", type=str, help="Diretório contendo arquivos de áudio")
    parser.add_argument("--output_dir", type=str, default="transcribed_midis", help="Diretório para salvar os MIDIs transcritos")
    parser.add_argument("--base_dir", type=str, default=None, help="Diretório base para manter estrutura relativa (ex: data/)")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo para execução do modelo (cpu, cuda, ou cuda:0)")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID da GPU para usar quando device=cuda (padrão: 0)")
    parser.add_argument("--recursive", action="store_true", help="Buscar recursivamente em subpastas")
    args = parser.parse_args()
    audio_dir = args.audio_dir
    output_base_dir = args.output_dir
    base_dir = args.base_dir if args.base_dir else wavs_dir
    device = args.device
    gpu_id = args.gpu_id
    recursive = args.recursive
    logger.info("Iniciando transcrição em lote com Basic Pitch")
    logger.info(f"Diretório de entrada: {audio_dir}")
    logger.info(f"Diretório de saída: {output_base_dir}")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Modo recursivo: {recursive}")
    
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
    
<<<<<<< HEAD
    # Obter todos os arquivos WAV organizados por caminho relativo
    try:
        instruments = get_all_wav_files(wavs_dir, base_dir=base_dir)
=======
    # Obter todos os arquivos de áudio
    try:
        audio_files = get_all_audio_files(audio_dir, recursive=recursive)
>>>>>>> b6800e367d27a1dac8049db0d43a39d60b7adee4
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if not audio_files:
        logger.warning("Nenhum arquivo de áudio encontrado!")
        return
    
    # Estatísticas
    total_files = len(audio_files)
    processed_files = 0
    successful_transcriptions = 0
    failed_transcriptions = 0
    
    logger.info(f"Total de {total_files} arquivos para transcrever")
    logger.info(f"{'='*60}")
    
    # Processar cada arquivo
    for abs_path, rel_path in audio_files:
        processed_files += 1
        
<<<<<<< HEAD
        # Criar diretório para este instrumento
        # A chave já é o caminho relativo, então podemos usar diretamente
        instrument_output_dir = os.path.join(output_base_dir, instrument)
        Path(instrument_output_dir).mkdir(parents=True, exist_ok=True)
=======
        # Determinar diretório de saída (preserva estrutura se recursivo)
        if recursive:
            rel_dir = os.path.dirname(rel_path)
            output_dir = os.path.join(output_base_dir, rel_dir) if rel_dir else output_base_dir
        else:
            output_dir = output_base_dir
>>>>>>> b6800e367d27a1dac8049db0d43a39d60b7adee4
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Verificar se já foi transcrito
        audio_basename = os.path.splitext(os.path.basename(abs_path))[0]
        expected_midi = os.path.join(output_dir, f"{audio_basename}_basic_pitch.mid")
        
        if os.path.exists(expected_midi):
            logger.info(f"[{processed_files}/{total_files}] ✓ Já existe: {rel_path}")
            successful_transcriptions += 1
            continue
        
        # Transcrever
        logger.info(f"[{processed_files}/{total_files}] Processando: {rel_path}")
        success = transcribe_audio_file(abs_path, output_dir, logger, model)
        
        if success:
            successful_transcriptions += 1
            logger.info(f"✓ MIDI salvo: {os.path.join(output_dir, f'{audio_basename}_basic_pitch.mid')}")
        else:
            failed_transcriptions += 1
            logger.error(f"✗ Falha ao transcrever: {rel_path}")
        
        # Progresso
        progress = (processed_files / total_files) * 100
        logger.info(f"Progresso: {processed_files}/{total_files} ({progress:.1f}%)")
        logger.info(f"{'-'*60}")
    
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