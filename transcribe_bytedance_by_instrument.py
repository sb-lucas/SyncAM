#!/usr/bin/env python3
"""
Script para transcrever áudios organizados por instrumento usando o framework
de piano transcription (Bytedance) presente em `piano_transcription/`.

Alinhado ao estilo de parâmetros do `transcribe_basicpitch.py`.

Entrada esperada:
  wavs_dir/INSTRUMENTO_A/*.wav
  wavs_dir/INSTRUMENTO_B/*.wav
  ...

Saída:
  output_dir/INSTRUMENTO_A/<arquivo>_bytedance.mid
  output_dir/INSTRUMENTO_B/<arquivo>_bytedance.mid

Requisitos:
  - PyTorch instalado
  - Dependências do diretório `piano_transcription` instaladas
  - Checkpoint PyTorch compatível contendo chave 'model' (state_dict) conforme
    utilizado em `piano_transcription/pytorch/inference.py`.

Uso:
  python transcribe_bytedance.py WAVS_DIR \
      --checkpoint /path/para/checkpoint.pth \
      --output_dir midis_bytedance \
      --device cuda --gpu_id 0 \
      --model_type Note_pedal \
      --segment_seconds 10

Observações:
  - `--batch_size` está presente para alinhamento mas não é utilizado (o
    processo atual faz forward segmento a segmento com batch interno = 1).
  - Para acelerar, garanta que o checkpoint esteja no formato correto e use GPU.
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


def get_all_wav_files(wavs_dir: str) -> Dict[str, list]:
    """Coleta arquivos .wav em subpastas imediatas (tratadas como instrumentos)."""
    instruments = {}
    if not os.path.exists(wavs_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {wavs_dir}")
    for instrument_dir in sorted(os.listdir(wavs_dir)):
        instrument_path = os.path.join(wavs_dir, instrument_dir)
        if os.path.isdir(instrument_path):
            wav_files = sorted(glob.glob(os.path.join(instrument_path, '*.wav')))
            if wav_files:
                instruments[instrument_dir] = wav_files
    return instruments


def load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    """Carrega áudio mono no sample rate alvo retornando numpy float32."""
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio


def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Transcrever áudios por instrumento usando PianoTranscription (Bytedance)")
    parser.add_argument("wavs_dir", type=str, help="Diretório raiz contendo subpastas de instrumentos com arquivos .wav")
    parser.add_argument("--checkpoint", required=True, type=str, help="Caminho para checkpoint .pth/.pt contendo state_dict com chave 'model'")
    parser.add_argument("--output_dir", type=str, default="bytedance_midis", help="Diretório para salvar os MIDIs gerados")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo (cpu|cuda)")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID da GPU quando --device=cuda")
    parser.add_argument("--batch_size", type=int, default=1, help="Alinhamento com outros scripts (não utilizado diretamente)")
    parser.add_argument("--model_type", type=str, default="Note_pedal", help="Classe de modelo definida em models.py (ex: Note_pedal)")
    parser.add_argument("--segment_seconds", type=float, default=10.0, help="Duração dos segmentos em segundos (default 10)")
    parser.add_argument("--post_processor", type=str, default="regression", choices=["regression", "onsets_frames"], help="Tipo de pós-processamento")
    args = parser.parse_args()

    wavs_dir = args.wavs_dir
    checkpoint = args.checkpoint
    output_dir = args.output_dir

    if not os.path.exists(wavs_dir):
        logger.error(f"Diretório de entrada não encontrado: {wavs_dir}")
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
    logger.info(
        "Configuração de execução:\n"
        f"  Entrada: {wavs_dir}\n"
        f"  Saída: {output_dir}\n"
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
        instruments = get_all_wav_files(wavs_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    if not instruments:
        logger.warning("Nenhum instrumento com arquivos .wav encontrado")
        return 0

    total_files = sum(len(v) for v in instruments.values())
    processed = 0
    success = 0
    failed = 0

    logger.info(f"Total de {total_files} arquivos para transcrever em {len(instruments)} instrumentos")

    for instrument, wav_files in instruments.items():
        logger.info("="*60)
        logger.info(f"Instrumento: {instrument} ({len(wav_files)} arquivos)")
        instrument_output_dir = os.path.join(output_dir, instrument)
        Path(instrument_output_dir).mkdir(parents=True, exist_ok=True)
        for i, wav_file in enumerate(wav_files, 1):
            basename = os.path.splitext(os.path.basename(wav_file))[0]
            target_midi = os.path.join(instrument_output_dir, f"{basename}_bytedance.mid")
            if os.path.exists(target_midi):
                logger.info(f"[{i}/{len(wav_files)}] Pulando (já existe): {os.path.basename(target_midi)}")
                processed += 1
                success += 1
                continue
            logger.info(f"[{i}/{len(wav_files)}] Transcrevendo: {os.path.basename(wav_file)}")
            try:
                audio = load_audio_mono(wav_file, config.sample_rate)
                _ = transcriber.transcribe(audio, target_midi)
                if os.path.exists(target_midi):
                    logger.info(f"MIDI salvo: {target_midi}")
                    success += 1
                else:
                    logger.error(f"MIDI não criado: {target_midi}")
                    failed += 1
            except Exception as e:
                logger.error(f"Erro ao transcrever {wav_file}: {e}")
                failed += 1
            processed += 1
            overall = (processed / total_files) * 100
            logger.info(f"Progresso geral: {processed}/{total_files} ({overall:.1f}%)")

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
