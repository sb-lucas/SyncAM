"""Utility functions and fields for dealing with note_dfs in mdtk format."""
import numpy as np
import pandas as pd

NOTE_DF_SORT_ORDER = ["onset", "track", "pitch", "dur", "velocity"]


def clean_df(df, single_track=False, non_overlapping=False):
    """
    Clean a given note_df by (optionally) flattening the tracks of all notes
    to 0, (optionally) removing overlaps between notes, and sorting the notes
    by ascending onset, track, pitch, and dur, finally removing all additional
    columns.
    """
    if single_track:
        df = df.assign(track=0)

    if non_overlapping:
        df = remove_pitch_overlaps(df)
    else:
        df = df.sort_values(by=NOTE_DF_SORT_ORDER).reset_index(drop=True)

    return df.loc[:, NOTE_DF_SORT_ORDER]


def remove_pitch_overlaps(df):
    """Returns a version of the given df with all same-pitch overlaps removed."""
    if len(df) < 2:
        return df

    df = df.sort_values(by=NOTE_DF_SORT_ORDER).reset_index(drop=True)
    df["offset"] = df["onset"] + df["dur"]
    offset = df["offset"].copy()

    for _, pitch_df in df.groupby(["track", "pitch"]):
        if len(pitch_df) < 2:
            continue
        cum_max = pitch_df["offset"].cummax()
        offset.loc[pitch_df.index] = cum_max.clip(
            upper=pitch_df["onset"].shift(-1, fill_value=cum_max.iloc[-1])
        )

    df["dur"] = offset - df["onset"]
    df = df.loc[df["dur"] != 0, NOTE_DF_SORT_ORDER]
    df = df.reset_index(drop=True)
    return df


def get_random_excerpt(
    note_df,
    min_notes=10,
    excerpt_length=5000,
    first_onset_range=(0, 200),
    iterations=10,
):
    """Take a random excerpt from the given note_df."""
    if len(note_df) < min_notes or iterations == 0:
        return None

    for _ in range(iterations):
        note_index = np.random.choice(list(note_df.index.values)[:-min_notes])
        first_onset = note_df.loc[note_index]["onset"]
        excerpt = pd.DataFrame(
            note_df.loc[
                note_df["onset"].between(first_onset, first_onset + excerpt_length)
            ]
        )
        if len(excerpt) < min_notes:
            excerpt = None
        else:
            break

    if excerpt is None:
        return None

    onset_shift = np.random.randint(first_onset_range[0], first_onset_range[1])
    excerpt["onset"] += onset_shift - first_onset
    excerpt = excerpt.reset_index(drop=True)
    return excerpt
