#!/usr/bin/env python3
"""
Aplica degradação MIDI a qualquer arquivo ou pasta de MIDI (genérico).

Tipos de degradação:
  - temporal   : deslocamento de onset (time_shift) em uma ou mais notas
  - pitch_swap : substituição de pitch por outro existente na peça
  - add_remove : remoções e inserções de notas

Severidade (--version 1, 2 ou 3): controla quantidade e magnitude das edições.
Saída: MIDI corrompido + JSON opcional com campo `changes` (ground truth).

Uso:
  python degrade_any_midi.py input.mid -o out_dir
  python degrade_any_midi.py pasta_midis/ -o out_dir --types temporal pitch_swap add_remove --version 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Garantir que o diretório do script está no path (rodar de qualquer lugar)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from mdtk.degradations import time_shift, pitch_shift, remove_note, add_note
from mdtk import fileio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Severidade: (min_shift_ms, max_shift_ms) temporal; número de edições por tipo
TEMPORAL_SHIFTS_MS = [(50, 150), (150, 400), (400, 800)]
TEMPORAL_NUM_EDITS = [1, 2, 3]
PITCH_NUM_EDITS = [1, 2, 3]
STRUCTURAL_SPEC = [(1, 0), (0, 2), (2, 2)]  # (n_remove, n_add) por nível
SCENARIOS = ["temporal", "pitch_swap", "add_remove"]


def _make_seed(basename: str, scenario: str, version: int, seed_base: int) -> int:
    h = hash(basename) & 0x7FFFFFFF
    scenario_idx = SCENARIOS.index(scenario)
    return seed_base + h + scenario_idx * 10 + version


def _compute_diff_temporal(clean_df, corrupted_df):
    """Diff para time_shift: mesmos (pitch, dur), onset diferente."""
    clean_df = clean_df.sort_values(["onset", "track", "pitch", "dur"]).reset_index(drop=True)
    corrupted_df = corrupted_df.sort_values(["onset", "track", "pitch", "dur"]).reset_index(drop=True)
    changes = []
    for (pitch, dur), g_clean in clean_df.groupby(["pitch", "dur"]):
        g_corr = corrupted_df[(corrupted_df["pitch"] == pitch) & (corrupted_df["dur"] == dur)]
        onsets_clean = set(g_clean["onset"].tolist())
        onsets_corr = set(g_corr["onset"].tolist())
        if onsets_clean != onsets_corr:
            only_clean = onsets_clean - onsets_corr
            only_corr = onsets_corr - onsets_clean
            for old_onset in only_clean:
                for new_onset in only_corr:
                    changes.append({
                        "type": "onset_shift",
                        "onset_ms": int(old_onset),
                        "pitch": int(pitch),
                        "dur_ms": int(dur),
                        "old_onset_ms": int(old_onset),
                        "new_onset_ms": int(new_onset),
                    })
    return changes


def _compute_diff_pitch(clean_df, corrupted_df):
    """Diff para pitch_shift: mesmo (onset, dur), pitch diferente."""
    clean_df = clean_df.sort_values(["onset", "track", "pitch", "dur"]).reset_index(drop=True)
    corrupted_df = corrupted_df.sort_values(["onset", "track", "pitch", "dur"]).reset_index(drop=True)
    changes = []
    for (onset, dur), g_clean in clean_df.groupby(["onset", "dur"]):
        g_corr = corrupted_df[(corrupted_df["onset"] == onset) & (corrupted_df["dur"] == dur)]
        if g_corr.empty:
            continue
        pitches_clean = set(g_clean["pitch"].tolist())
        pitches_corr = set(g_corr["pitch"].tolist())
        if pitches_clean != pitches_corr:
            only_clean = pitches_clean - pitches_corr
            only_corr = pitches_corr - pitches_clean
            for old_pitch in only_clean:
                for new_pitch in only_corr:
                    changes.append({
                        "type": "pitch_swap",
                        "onset_ms": int(onset),
                        "dur_ms": int(dur),
                        "old_pitch": int(old_pitch),
                        "new_pitch": int(new_pitch),
                    })
    return changes


def _compute_diff_add_remove(clean_df, corrupted_df):
    """Diff add/remove: apenas em clean = removida; apenas em corrupted = adicionada."""
    key = lambda r: (r["onset"], r["pitch"], r["dur"])
    set_clean = set(key(r) for _, r in clean_df.iterrows())
    set_corr = set(key(r) for _, r in corrupted_df.iterrows())
    removed = set_clean - set_corr
    added = set_corr - set_clean
    changes = []
    for (onset, pitch, dur) in removed:
        changes.append({"type": "removed", "onset_ms": int(onset), "pitch": int(pitch), "dur_ms": int(dur)})
    for (onset, pitch, dur) in added:
        changes.append({"type": "added", "onset_ms": int(onset), "pitch": int(pitch), "dur_ms": int(dur)})
    return changes


def _apply_temporal(df, version: int, seed: int):
    min_shift, max_shift = TEMPORAL_SHIFTS_MS[version - 1]
    n_calls = TEMPORAL_NUM_EDITS[version - 1]
    current = df.copy()
    for i in range(n_calls):
        current = time_shift(current, min_shift=min_shift, max_shift=max_shift, seed=seed + i)
        if current is None:
            return None
    return current


def _apply_pitch_swap(df, version: int, seed: int):
    n_calls = PITCH_NUM_EDITS[version - 1]
    min_pitch = int(df["pitch"].min())
    max_pitch = int(df["pitch"].max())
    current = df.copy()
    for i in range(n_calls):
        current = pitch_shift(
            current,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            align_pitch=True,
            seed=seed + i,
        )
        if current is None:
            return None
    return current


def _apply_add_remove(df, version: int, seed: int, min_duration: int = 50, max_duration: int = 2000):
    n_remove, n_add = STRUCTURAL_SPEC[version - 1]
    current = df.copy()
    min_pitch = int(df["pitch"].min())
    max_pitch = int(df["pitch"].max())
    for i in range(n_remove):
        current = remove_note(current, seed=seed + i)
        if current is None:
            return None
    for i in range(n_add):
        current = add_note(
            current,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            min_duration=min_duration,
            max_duration=max_duration,
            min_velocity=80,
            max_velocity=120,
            seed=seed + 100 + i,
        )
        if current is None:
            return None
    return current


def process_one_midi(
    midi_path: Path,
    out_root: Path,
    types: list[str],
    versions: list[int],
    seed_base: int,
    save_metadata: bool,
) -> int:
    """
    Processa um único MIDI: aplica todos os cenários/types e versions pedidos.
    Retorna quantidade de arquivos gerados.
    """
    midi_path = Path(midi_path)
    if not midi_path.exists():
        logger.warning("MIDI não encontrado: %s", midi_path)
        return 0

    clean_df = fileio.midi_to_df(str(midi_path))
    if clean_df is None or len(clean_df) == 0:
        logger.warning("MIDI vazio ou falha ao carregar: %s", midi_path)
        return 0

    stem = midi_path.stem
    count = 0

    for scenario in types:
        if scenario not in SCENARIOS:
            logger.warning("Tipo desconhecido ignorado: %s", scenario)
            continue
        for version in versions:
            if version not in (1, 2, 3):
                continue
            seed = _make_seed(stem, scenario, version, seed_base)
            if scenario == "temporal":
                corrupted_df = _apply_temporal(clean_df, version, seed)
            elif scenario == "pitch_swap":
                corrupted_df = _apply_pitch_swap(clean_df, version, seed)
            else:
                corrupted_df = _apply_add_remove(clean_df, version, seed)

            if corrupted_df is None:
                logger.warning("Degradação retornou None: %s %s v%d", stem, scenario, version)
                continue

            out_dir = out_root / scenario / f"v{version}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_midi = out_dir / f"{stem}.mid"
            fileio.df_to_midi(corrupted_df, str(out_midi), existing_midi_path=str(midi_path))

            if save_metadata:
                if scenario == "temporal":
                    changes = _compute_diff_temporal(clean_df, corrupted_df)
                elif scenario == "pitch_swap":
                    changes = _compute_diff_pitch(clean_df, corrupted_df)
                else:
                    changes = _compute_diff_add_remove(clean_df, corrupted_df)
                meta = {
                    "source_midi": str(midi_path),
                    "corrupted_midi": str(out_midi),
                    "scenario": scenario,
                    "version": version,
                    "seed": seed,
                    "changes": changes,
                }
                meta_path = out_dir / f"{stem}.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

            count += 1
            logger.info("Gerado: %s", out_midi)

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Aplica degradação MIDI a qualquer arquivo ou pasta (genérico)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Arquivo .mid/.midi ou diretório com MIDIs",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("degraded_midi"),
        help="Diretório de saída (default: degraded_midi)",
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=SCENARIOS,
        choices=SCENARIOS,
        help="Tipos de degradação (default: temporal pitch_swap add_remove)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Aplicar só uma severidade (1, 2 ou 3). Omitido = todas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base do seed para reprodutibilidade",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Não salvar JSON com changes",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_root = args.output.resolve()
    versions = [args.version] if args.version is not None else [1, 2, 3]

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.mid")) + list(input_path.glob("*.midi"))
        files = sorted(set(files))
    else:
        logger.error("Input não encontrado: %s", input_path)
        sys.exit(1)

    if not files:
        logger.error("Nenhum MIDI encontrado em %s", input_path)
        sys.exit(1)

    logger.info("Processando %d arquivo(s) MIDI.", len(files))
    total = 0
    for f in files:
        total += process_one_midi(
            f,
            out_root,
            types=args.types,
            versions=versions,
            seed_base=args.seed,
            save_metadata=not args.no_metadata,
        )

    logger.info("Concluído: %d arquivo(s) degradados em %s", total, out_root)


if __name__ == "__main__":
    main()
