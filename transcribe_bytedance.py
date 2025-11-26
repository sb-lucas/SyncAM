#!/usr/bin/env python3
"""
Script para transcrever áudios usando o framework piano_transcription (Bytedance).

Suporta múltiplos formatos de áudio (wav, mp3, flac, m4a, ogg) e modos:
  - Modo normal: transcreve apenas áudios na raiz do diretório de entrada
  - Modo recursivo: transcreve todos os áudios mantendo estrutura de pastas

Entrada:
  audio_dir/arquivo1.wav
  audio_dir/arquivo2.mp3
  audio_dir/subpasta/arquivo3.flac  (apenas com --recursive)

Saída (modo normal):
  output_dir/arquivo1_bytedance.mid
  output_dir/arquivo2_bytedance.mid

Saída (modo recursivo):
  output_dir/arquivo1_bytedance.mid
  output_dir/arquivo2_bytedance.mid
  output_dir/subpasta/arquivo3_bytedance.mid

Requisitos:
  - PyTorch instalado
  - Dependências do piano_transcription instaladas
  - Checkpoint PyTorch compatível

Uso:
  # Modo normal (apenas raiz)
  python transcribe_bytedance.py AUDIO_DIR \
      --checkpoint piano_transcription/checkpoint.pth \
      --output_dir midis_output \
      --device cuda --gpu_id 0

  # Modo recursivo (todas as subpastas)
  python transcribe_bytedance.py AUDIO_DIR \
      --checkpoint piano_transcription/checkpoint.pth \
      --output_dir midis_output \
      --recursive \
      --device cuda --gpu_id 0
"""

import os
import sys
import glob
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import librosa

# Adicionar diretórios do framework ao PYTHONPATH para import relativo funcionar
ROOT_DIR = Path(__file__).parent.resolve()
PT_DIR = ROOT_DIR / 'piano_transcription' / 'pytorch'
UTILS_DIR = ROOT_DIR / 'piano_transcription' / 'utils'
if str(PT_DIR) not in sys.path:
    sys.path.insert(0, str(PT_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

try:
    from inference import PianoTranscription  # usa sys.path interno para utils
    import config  # de utils/config.py
except Exception as e:
    print(f"Erro ao importar framework de piano_transcription: {e}")
    print("Verifique se as dependências foram instaladas e o caminho está correto.")
    sys.exit(1)


def setup_logging():
    log_filename = f"bytedance_transcription_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("bytedance")


def get_all_audio_files(audio_dir: str, recursive: bool = False) -> list:
    """
    Coleta arquivos de áudio suportados.
    
    Args:
        audio_dir: Diretório raiz para buscar áudios
        recursive: Se True, busca recursivamente em subpastas
        
    Returns:
        Lista de tuplas (caminho_absoluto, caminho_relativo)
    """
    supported_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    audio_files = []
    
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {audio_dir}")
    
    audio_dir = os.path.abspath(audio_dir)
    
    if recursive:
        # Busca recursiva em todas as subpastas
        for root, dirs, files in os.walk(audio_dir):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, audio_dir)
                    audio_files.append((abs_path, rel_path))
    else:
        # Busca apenas na raiz
        for file in sorted(os.listdir(audio_dir)):
            file_path = os.path.join(audio_dir, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    audio_files.append((file_path, file))
    
    return audio_files


def load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    """Carrega áudio mono no sample rate alvo retornando numpy float32."""
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio


def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(
        description="Transcrever áudios usando PianoTranscription (Bytedance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Transcrever apenas arquivos na raiz
  python transcribe_bytedance.py audio_folder --checkpoint checkpoint.pth

  # Transcrever recursivamente mantendo estrutura de pastas
  python transcribe_bytedance.py audio_folder --checkpoint checkpoint.pth --recursive

  # Usar GPU específica
  python transcribe_bytedance.py audio_folder --checkpoint checkpoint.pth --device cuda --gpu_id 1

Formatos suportados: wav, mp3, flac, m4a, ogg, aac, wma
        """
    )
    parser.add_argument("audio_dir", type=str, help="Diretório contendo arquivos de áudio")
    parser.add_argument("--checkpoint", required=True, type=str, help="Caminho para checkpoint .pth/.pt")
    parser.add_argument("--output_dir", type=str, default="bytedance_midis", help="Diretório para salvar os MIDIs gerados")
    parser.add_argument("--recursive", action="store_true", help="Buscar áudios recursivamente em subpastas")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo (cpu|cuda)")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID da GPU quando --device=cuda")
    parser.add_argument("--model_type", type=str, default="Note_pedal", help="Classe de modelo (Note_pedal)")
    parser.add_argument("--segment_seconds", type=float, default=10.0, help="Duração dos segmentos em segundos")
    parser.add_argument("--post_processor", type=str, default="regression", choices=["regression", "onsets_frames"], help="Tipo de pós-processamento")
    args = parser.parse_args()

    audio_dir = args.audio_dir
    checkpoint = args.checkpoint
    output_dir = args.output_dir

    if not os.path.exists(audio_dir):
        logger.error(f"Diretório de entrada não encontrado: {audio_dir}")
        return 1
    if not os.path.isfile(checkpoint):
        logger.error(f"Checkpoint não encontrado: {checkpoint}")
        return 1
    os.makedirs(output_dir, exist_ok=True)

    # Dispositivo
    if args.device.startswith('cuda'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Usando GPU id {args.gpu_id}")
        else:
            logger.warning("CUDA solicitada mas não disponível. Usando CPU.")
            device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = torch.device('cpu')
        logger.info("Usando CPU")

    segment_samples = int(args.segment_seconds * config.sample_rate)
    mode_str = "recursivo" if args.recursive else "raiz apenas"
    logger.info(
        "Configuração de execução:\n"
        f"  Entrada: {audio_dir}\n"
        f"  Saída: {output_dir}\n"
        f"  Modo: {mode_str}\n"
        f"  Modelo: {args.model_type}\n"
        f"  Checkpoint: {checkpoint}\n"
        f"  Sample rate: {config.sample_rate}\n"
        f"  Segment samples: {segment_samples}\n"
        f"  Pós-processador: {args.post_processor}"
    )

    # Instancia transcritor
    try:
        transcriber = PianoTranscription(
            model_type=args.model_type,
            checkpoint_path=checkpoint,
            segment_samples=segment_samples,
            device=device,
            post_processor_type=args.post_processor
        )
    except Exception as e:
        logger.error(f"Falha ao inicializar modelo: {e}")
        return 1

    # Coleta arquivos
    try:
        audio_files = get_all_audio_files(audio_dir, recursive=args.recursive)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    if not audio_files:
        logger.warning("Nenhum arquivo de áudio encontrado")
        return 0

    total_files = len(audio_files)
    processed = 0
    success = 0
    failed = 0

    logger.info(f"Total de {total_files} arquivos de áudio encontrados")
    logger.info("="*60)

    for abs_path, rel_path in audio_files:
        processed += 1
        
        # Determinar caminho de saída mantendo estrutura de pastas
        basename = os.path.splitext(os.path.basename(rel_path))[0]
        rel_dir = os.path.dirname(rel_path)
        
        if rel_dir:
            # Áudio está em subpasta - manter estrutura
            output_subdir = os.path.join(output_dir, rel_dir)
            Path(output_subdir).mkdir(parents=True, exist_ok=True)
            target_midi = os.path.join(output_subdir, f"{basename}_bytedance.mid")
        else:
            # Áudio está na raiz
            target_midi = os.path.join(output_dir, f"{basename}_bytedance.mid")
        
        if os.path.exists(target_midi):
            logger.info(f"[{processed}/{total_files}] Pulando (já existe): {rel_path}")
            success += 1
            continue
        
        logger.info(f"[{processed}/{total_files}] Transcrevendo: {rel_path}")
        
        try:
            audio = load_audio_mono(abs_path, config.sample_rate)
            _ = transcriber.transcribe(audio, target_midi)
            
            if os.path.exists(target_midi):
                logger.info(f"  ✓ MIDI salvo: {os.path.relpath(target_midi, output_dir)}")
                success += 1
            else:
                logger.error(f"  ✗ MIDI não criado")
                failed += 1
        except Exception as e:
            logger.error(f"  ✗ Erro: {e}")
            failed += 1
        
        overall = (processed / total_files) * 100
        logger.info(f"  Progresso: {processed}/{total_files} ({overall:.1f}%)")

    logger.info("="*60)
    logger.info("RELATÓRIO FINAL")
    logger.info(f"Arquivos processados: {processed}")
    logger.info(f"Sucessos: {success}")
    logger.info(f"Falhas: {failed}")
    if processed:
        logger.info(f"Taxa de sucesso: {success/processed*100:.1f}%")
    logger.info(f"MIDIs salvos em: {os.path.abspath(output_dir)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
