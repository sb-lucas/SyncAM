import mir_eval
import pretty_midi
import numpy as np

def load_midi_notes(midi_path):
    """
    Parses a MIDI file and extracts note intervals (onset, offset) and pitch values.
    
    Returns:
      - intervals: np.ndarray of shape (N, 2) containing [onset_time, offset_time] in seconds.
      - pitches: np.ndarray of shape (N,) containing MIDI note numbers (e.g., 60 for C4).
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    intervals = []
    pitches = []

    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                intervals.append([note.start, note.end])
                pitches.append(note.pitch)

    intervals = np.array(intervals)
    pitches = np.array(pitches)

    # Sort notes by onset time for clean indexing
    if len(intervals) > 0:
        sort_idx = np.argsort(intervals[:, 0])
        intervals = intervals[sort_idx]
        pitches = pitches[sort_idx]

    return intervals, pitches

def Transcription_score(audio_transcribed_midi_path, midi_path):
    # 1. Extract notes from ground truth and your transcribed MIDI
    ref_intervals, ref_pitches = load_midi_notes(midi_path)
    est_intervals, est_pitches = load_midi_notes(audio_transcribed_midi_path)

    # Return 0.0 early if either reference or estimation contains no notes
    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return 0.0

    # 2. Run mir_eval evaluation
    scores = mir_eval.transcription.evaluate(
        ref_intervals, 
        ref_pitches, 
        est_intervals, 
        est_pitches,
        onset_tolerance=0.05  # Default: 50 ms tolerance window for onsets
    )

    return scores['F-measure']  # Return the F-measure as the transcription score
