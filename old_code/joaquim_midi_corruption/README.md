# music-transcription

Ferramentas de **degradação MIDI** e corrupção controlada para qualquer arquivo MIDI, úteis para benchmark de transcrição e alinhamento áudio–MIDI.

## Estrutura

- **`degrade_any_midi.py`** – Script principal: aplica degradações (temporal, pitch_swap, add_remove) a qualquer `.mid`/`.midi` ou pasta.
- **`midi_corruption_tool.py`** – Corrupção por **ticks** (shift linear, progressivo, por seção, jitter) com `mido`.
- **`mdtk/`** – Subconjunto do [MIDI Degradation Toolkit](https://github.com/JamesOwers/midi_degradation_toolkit): `fileio` (midi ↔ DataFrame), `degradations` (time_shift, pitch_shift, remove_note, add_note), `df_utils`.

## Instalação

```bash
cd music-transcription
pip install -r requirements.txt
```

## Uso: degradação por notas (`degrade_any_midi.py`)

Funciona com **qualquer MIDI** (um arquivo ou uma pasta).

```bash
# Um arquivo – gera todas as degradações (temporal, pitch_swap, add_remove) e níveis 1,2,3
python degrade_any_midi.py meu_arquivo.mid -o saida/

# Uma pasta com vários MIDIs
python degrade_any_midi.py pasta_com_midis/ -o saida/

# Só temporal e pitch_swap, só severidade 2
python degrade_any_midi.py input.mid -o saida/ --types temporal pitch_swap --version 2

# Sem salvar JSON de metadata (só MIDIs corrompidos)
python degrade_any_midi.py input.mid -o saida/ --no-metadata
```

**Tipos de degradação**

| Tipo          | Descrição |
|---------------|-----------|
| `temporal`    | Desloca o onset de 1–3 notas (magnitude aumenta com `--version` 1→3). |
| `pitch_swap`  | Substitui o pitch de 1–3 notas por outro pitch já presente na peça. |
| `add_remove`  | Remove e/ou adiciona notas (estrutural). |

**Severidade (`--version` 1, 2 ou 3)**  
Controla quantidade de edições e, no temporal, a faixa de shift em ms (ex.: ~50–150 ms em v1, até ~400–800 ms em v3).

**Saída**

- MIDIs corrompidos em `output/<tipo>/v<1|2|3>/<nome>.mid`.
- Se não usar `--no-metadata`, um `<nome>.json` por MIDI com `changes` (ground truth das edições).

## Uso: corrupção por ticks (`midi_corruption_tool.py`)

Deslocamentos temporais em **ticks** (útil para desincronização áudio–MIDI):

```bash
# Um arquivo com presets padrão
python midi_corruption_tool.py arquivo.mid -o corrupted/

# Pasta inteira
python midi_corruption_tool.py pasta_midis/ -o corrupted/

# Configuração customizada (JSON)
python midi_corruption_tool.py arquivo.mid -c config.json -o corrupted/ --log log.json
```

Tipos: `linear_shift`, `progressive_shift`, `section_shifts`, `random_jitter`. Ver `README_midi_corruption.md` para formato do JSON de configuração.

## Dependências

- `mido`, `numpy` – `midi_corruption_tool.py`
- `pandas`, `pretty_midi` – `degrade_any_midi.py` e `mdtk`

Tudo listado em `requirements.txt`.
