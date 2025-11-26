# YourMT3+ - Scripts de Transcrição de Pasta

Este repositório contém scripts para transcrever uma pasta de arquivos de áudio recursivamente usando o YourMT3+.

## Scripts Disponíveis

### 1. `transcribe_folder.py` - Script Completo
Script completo e independente que configura todo o ambiente do YourMT3.

**Uso:**
```bash
python transcribe_folder.py --input-folder /path/to/audio/files --output-folder /path/to/midi/output --model-name "YPTF.MoE+Multi (noPS)"
```

**Argumentos:**
- `--input-folder, -i`: Pasta contendo arquivos de áudio (busca recursiva)
- `--output-folder, -o`: Pasta onde salvar os arquivos MIDI
- `--model-name, -m`: Modelo a usar (padrão: "YPTF.MoE+Multi (noPS)")
- `--precision, -p`: Precisão do modelo (padrão: "bf16-mixed")
- `--batch-size, -b`: Tamanho do batch (padrão: 8)
- `--project`: Nome do projeto (padrão: "2024")

### 2. `transcribe_folder_simple.py` - Script Simplificado
Script mais simples que assume que o ambiente YourMT3 já está configurado.

**Uso no Google Colab:**
```python
# Primeiro execute as células de setup do notebook YourMT3+_Demo_2024.ipynb
%cd /content/amt/src
!python /path/to/transcribe_folder_simple.py /content/audio_files /content/midi_output --model "YPTF.MoE+Multi (noPS)"
```

**Uso local:**
```bash
# Certifique-se de estar no diretório correto do YourMT3
cd /path/to/amt/src
python transcribe_folder_simple.py /path/to/audio /path/to/output --model "YPTF.MoE+Multi (noPS)"
```

## Modelos Disponíveis

- `YMT3+`
- `YPTF+Single (noPS)`  
- `YPTF+Multi (PS)`
- `YPTF.MoE+Multi (noPS)` - **Padrão (recomendado)**
- `YPTF.MoE+Multi (PS)`

## Formatos de Áudio Suportados

- WAV (.wav)
- MP3 (.mp3)  
- FLAC (.flac)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)
- WMA (.wma)

## Configuração do Ambiente

### Opção 1: Google Colab (Recomendado)
1. Abra o notebook `YourMT3+_Demo_2024.ipynb` no Google Colab
2. Execute as células de setup (células 1-3)
3. Use o script simplificado:

```python
# Upload dos arquivos de áudio para /content/audio_files
%cd /content/amt/src
!python /content/transcribe_folder_simple.py /content/audio_files /content/midi_output
```

### Opção 2: Instalação Local
1. Clone o repositório do YourMT3:
```bash
git clone https://github.com/mimbres/YourMT3.git
cd YourMT3/src
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
pip install transformers==4.45.1
```

3. Baixe os modelos (siga as instruções do YourMT3)

4. Execute o script:
```bash
python transcribe_folder.py --input-folder /path/to/audio --output-folder /path/to/midi
```

## Exemplos de Uso

### Exemplo 1: Transcrever pasta básica
```bash
python transcribe_folder_simple.py ./audio_files ./midi_output
```

### Exemplo 2: Usar modelo específico com batch maior
```bash
python transcribe_folder_simple.py ./audio_files ./midi_output --model "YMT3+" --batch-size 16
```

### Exemplo 3: Script completo com configuração específica
```bash
python transcribe_folder.py \
  --input-folder /home/user/music \
  --output-folder /home/user/midi_results \
  --model-name "YPTF.MoE+Multi (noPS)" \
  --precision "bf16-mixed" \
  --batch-size 8
```

## Estrutura de Saída

Os arquivos MIDI serão salvos em:
```
output_folder/
└── model_output/
    ├── audio_file1.mid
    ├── audio_file2.mid
    └── ...
```

## Requisitos do Sistema

- **GPU recomendada**: NVIDIA GPU com pelo menos 8GB de VRAM
- **RAM**: Pelo menos 16GB
- **Python**: 3.8 ou superior
- **PyTorch**: Versão compatível com CUDA (se usando GPU)

## Solução de Problemas

### Erro: "Não foi possível importar módulos do YourMT3"
- Certifique-se de estar no diretório correto (`amt/src`)
- Verifique se todas as dependências estão instaladas
- No Colab, execute primeiro as células de setup do notebook

### Erro de memória
- Reduza o `batch-size` (ex: `--batch-size 4` ou `--batch-size 2`)
- Use precisão menor se disponível

### Arquivos não encontrados
- Verifique se a pasta de entrada existe
- Certifique-se de que há arquivos de áudio com extensões suportadas
- Use caminhos absolutos se houver problemas com caminhos relativos

## Performance

**Tempo estimado por arquivo (GPU NVIDIA RTX 4090):**
- Arquivo de 30s: ~5-10 segundos
- Arquivo de 3min: ~30-60 segundos  
- Arquivo de 10min: ~2-5 minutos

**Tempo varia conforme:**
- Duração do áudio
- Complexidade musical
- Modelo usado
- Hardware disponível

## Notas Importantes

1. **Modelos PS vs noPS**: Modelos com "PS" (pitch-shift) podem ter problemas com músicas fora do dataset de treinamento
2. **Qualidade**: O modelo YPTF.MoE+Multi (noPS) é recomendado para melhor qualidade geral
3. **Memória**: Arquivos muito longos podem causar problemas de memória - considere dividir em segmentos menores
4. **Formatos**: O script converte automaticamente diferentes formatos para o formato esperado pelo modelo