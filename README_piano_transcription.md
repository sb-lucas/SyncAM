# Piano Transcription Script

Script para transcrever áudios sintéticos usando o framework Piano Transcription.

## Requisitos

Antes de usar o script, instale as dependências necessárias:

```bash
# Navegue para a pasta piano_transcription
cd piano_transcription

# Instale as dependências
pip install -r requirements.txt

# Ou instale manualmente:
pip install torch librosa mido h5py pandas numba mir_eval matplotlib
```

## Como usar

### Uso básico

```bash
python transcribe_piano_transcription.py --samples_dir datasets/maestro_dataset_midi_wavs --output_dir datasets/maestro_dataset_midi_wavs_transcribed_piano
```

### Parâmetros disponíveis

- `--samples_dir`: Diretório contendo arquivos WAV sintéticos (padrão: "samples_wavs")
- `--output_dir`: Diretório para salvar os MIDIs transcritos (padrão: "transcribed_midis_piano")
- `--checkpoint_path`: Caminho para o checkpoint do modelo (padrão: "piano_transcription/models/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth")
- `--model_type`: Tipo do modelo (padrão: "Note_pedal")
  - Note_pedal: Modelo completo com notas e pedal
  - Regress_onset_offset_frame_velocity_CRNN: Modelo apenas de notas
  - Regress_pedal_CRNN: Modelo apenas de pedal
- `--device`: Dispositivo para execução (padrão: "cpu")
  - cpu: Usar CPU
  - cuda: Usar GPU (se disponível)
- `--gpu_id`: ID da GPU para usar quando device=cuda (padrão: 0)
- `--batch_size`: Número de arquivos para processar por lote (padrão: 1)
- `--post_processor_type`: Tipo de pós-processador (padrão: "regression")
  - regression: Algoritmo de pós-processamento de alta resolução (recomendado)
  - onsets_frames: Algoritmo do Google Onsets and Frames (apenas para comparação)
- `--segment_seconds`: Duração dos segmentos para processamento em segundos (padrão: 10.0)

### Exemplos

1. **Processamento básico com CPU:**
```bash
python transcribe_piano_transcription.py \
    --samples_dir datasets/ATEPP_midi_wavs \
    --output_dir datasets/ATEPP_midi_wavs_transcribed_piano \
    --device cpu
```

2. **Processamento com GPU:**
```bash
python transcribe_piano_transcription.py \
    --samples_dir datasets/maestro_dataset_midi_wavs \
    --output_dir datasets/maestro_dataset_midi_wavs_transcribed_piano \
    --device cuda \
    --gpu_id 0
```

3. **Processamento em lotes:**
```bash
python transcribe_piano_transcription.py \
    --samples_dir datasets/MAPS_midi_wavs \
    --output_dir datasets/MAPS_midi_wavs_transcribed_piano \
    --device cpu \
    --batch_size 3
```

4. **Usando modelo personalizado:**
```bash
python transcribe_piano_transcription.py \
    --samples_dir datasets/ATEPP_midi_wavs \
    --output_dir datasets/ATEPP_midi_wavs_transcribed_piano \
    --checkpoint_path /caminho/para/seu/modelo.pth \
    --model_type Note_pedal
```

## Download automático do modelo

O script automaticamente baixa o modelo pré-treinado se ele não estiver presente:
- Modelo: CRNN_note_F1=0.9677_pedal_F1=0.9186.pth
- Fonte: [Zenodo](https://zenodo.org/record/4034264)

## Estrutura de saída

Os arquivos MIDI transcritos serão salvos com o sufixo `_piano_transcription.mid`:

```
output_dir/
├── Acoustic Grand Piano/
│   ├── arquivo1_piano_transcription.mid
│   └── arquivo2_piano_transcription.mid
├── Bright Acoustic Piano/
│   └── ...
└── ...
```

## Performance

### CPU vs GPU

- **CPU**: Mais lento, mas funciona em qualquer máquina
- **GPU**: Muito mais rápido se você tem uma GPU NVIDIA com CUDA

### Otimização

- Use `--batch_size` maior para melhor throughput (mas consome mais memória)
- Use `--segment_seconds` menor se você tem pouca memória GPU
- Para processamento rápido, use GPU com batch_size 2-4

## Diferenças do Basic Pitch

| Característica | Piano Transcription | Basic Pitch |
|---|---|---|
| **Especialização** | Piano específico | Multi-instrumento |
| **Pedal** | ✅ Detecta pedal sustain | ❌ Não detecta pedal |
| **Velocidade** | Rápido | Muito rápido |
| **Precisão** | Alta para piano | Boa para vários instrumentos |
| **Modelos** | PyTorch | TensorFlow |

## Troubleshooting

### Erro "Module not found"
```bash
# Certifique-se de que está no diretório correto
cd /raid/aluno_lucasgris/Repos/music-transcription

# Verifique se as dependências estão instaladas
pip install torch librosa mido
```

### Erro de memória GPU
```bash
# Use CPU ou reduza batch_size/segment_seconds
python transcribe_piano_transcription.py --device cpu
# ou
python transcribe_piano_transcription.py --device cuda --batch_size 1 --segment_seconds 5
```

### Modelo não encontrado
O script deve baixar automaticamente. Se falhar, baixe manualmente:
```bash
wget -O piano_transcription/models/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
```