"""Code to read/write note_dfs from/to midi and csv files."""
import logging
import os

import pandas as pd
import pretty_midi

from mdtk.df_utils import NOTE_DF_SORT_ORDER, clean_df

COLNAMES = NOTE_DF_SORT_ORDER
DEFAULT_VELOCITY = 100


def midi_to_df(midi_path, single_track=False, non_overlapping=False):
    """
    Get the data from a MIDI file and load it into a pandas DataFrame.
    Supports .mid and .midi extensions.
    Returns DataFrame with columns: onset (ms), track, pitch, dur (ms), velocity.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        logging.warning("Error parsing midi file %s. Skipping.", midi_path)
        return None

    notes = []
    for index, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            notes.append(
                {
                    "onset": int(round(note.start * 1000)),
                    "track": index,
                    "pitch": note.pitch,
                    "dur": int(round(note.end * 1000) - round(note.start * 1000)),
                    "velocity": note.velocity,
                }
            )

    if len(notes) == 0:
        logging.warning("WARNING: the midi file located at %s is empty. Returning None.", midi_path)
        return None

    df = clean_df(
        pd.DataFrame(notes), single_track=single_track, non_overlapping=non_overlapping
    )
    return df


def df_to_midi(
    df, midi_path, existing_midi_path=None, excerpt_start=0, excerpt_length=float("Inf")
):
    """Write the notes of a DataFrame out to a MIDI file."""
    assert excerpt_start >= 0, "excerpt_start must not be negative"
    excerpt_start_secs = excerpt_start / 1000

    midi = pretty_midi.PrettyMIDI()
    instruments = [None] * (df.track.max() + 1)

    if existing_midi_path is not None:
        existing_midi = pretty_midi.PrettyMIDI(existing_midi_path)
        excerpt_end_secs = excerpt_start_secs + excerpt_length / 1000

        midi.key_signature_changes = existing_midi.key_signature_changes
        midi.time_signature_changes = existing_midi.time_signature_changes
        midi.lyrics = existing_midi.lyrics

        for i, instrument in enumerate(existing_midi.instruments):
            instruments[i] = pretty_midi.Instrument(
                instrument.program, is_drum=instrument.is_drum, name=instrument.name
            )
            instruments[i].pitch_bends = instrument.pitch_bends
            instruments[i].control_changes = instrument.control_changes
            instruments[i].notes = [
                note
                for note in instrument.notes
                if (note.start < excerpt_start_secs or note.start >= excerpt_end_secs)
            ]

    for track in df.track.unique():
        if instruments[track] is not None:
            continue
        instruments[track] = pretty_midi.Instrument(track)

    midi.instruments = [
        instrument for instrument in instruments if instrument is not None
    ]

    work = df.copy()
    work.loc[:, "start"] = work.onset / 1000 + excerpt_start_secs
    work.loc[:, "end"] = work.start + work.dur / 1000

    for note in work.itertuples():
        midi_note = pretty_midi.Note(
            velocity=note.velocity, pitch=note.pitch, start=note.start, end=note.end
        )
        instruments[int(note.track)].notes.append(midi_note)

    midi.write(midi_path)
