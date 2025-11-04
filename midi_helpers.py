import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import mir_eval
import pretty_midi
import os
import random
import librosa.display
import pandas as pd
from tempfile import TemporaryDirectory
import pretty_midi
import soundfile as sf

# Piano Timbres:
#  0	Acoustic Grand Piano
#  1	Bright Acoustic Piano
#  2	Electric Grand Piano
#  3	Honky-tonk Piano
#  4	Rhodes Piano
#  5	Chorused Piano
#  6	Harpsichord
#  7	Clavinet

# Guitar Timbres:
# 24	Acoustic Nylon Guitar
# 27	Electric Clean Guitar
# 29	Overdriven Guitar

# String Timbres:
# 40	Violin
# 42	Cello

# Bass Timbres:
# 32	Acoustic Bass

# Brass Timbres:
# 56	Trumpet
# 57	Trombone

INSTRUMENTS = {'Acoustic Grand Piano':0,
                'Bright Acoustic Piano':1,
                'Electric Grand Piano':2,
                'Honky-tonk Piano':3,
                'Rhodes Piano':4,
                'Chorused Piano':5,
                'Harpsichord':6,
                'Clavinet':7,
                'Acoustic Nylon Guitar':24,
                'Electric Clean Guitar':27,
                'Overdriven Guitar':29,
                'Violin':40,
                'Cello':42,
                'Acoustic Bass':32,
                'Trumpet':56,
                'Trombone':57
    }

def trim_random_midi_segment(midi_path, output_path, segment_length=1.0):
    """
    Trim a random segment from a MIDI file and save it as a new MIDI file.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path to save the trimmed MIDI file
        segment_length: Length of the segment to extract (seconds)
    """

    print(f"Loading MIDI: {midi_path}")
    pm = pretty_midi.PrettyMIDI(midi_path)

    midi_duration = pm.get_end_time()
    if midi_duration <= segment_length:
        print(f"MIDI is too short ({midi_duration:.2f}s), skipping")
        return

    # Choose a random start time so we stay within file bounds
    start_time = random.uniform(0, midi_duration - segment_length)
    end_time = start_time + segment_length
    print(f"Trimming from {start_time:.2f}s to {end_time:.2f}s")

    # Create a new PrettyMIDI object and copy over trimmed notes
    new_pm = pretty_midi.PrettyMIDI()
    for instrument in pm.instruments:
        new_inst = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum)
        for note in instrument.notes:
            if note.start >= start_time and note.start < end_time:
                trimmed_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start_time,  # shift to start at 0
                    end=min(note.end, end_time) - start_time
                )
                new_inst.notes.append(trimmed_note)
        new_pm.instruments.append(new_inst)

    # Create output dir if needed
    new_pm.write(output_path)
    print(f"Saved trimmed MIDI to {output_path}")


def render_midi_to_audio(midi_path, soundfont_path, output_path, program_number, sample_rate=44100):
    """
    Render a MIDI file to audio using a soundfont
    
    Args:
        midi_path: Path to the input MIDI file
        soundfont_path: Path to the soundfont (.sf2) file
        output_path: Path for the output audio file (.wav)
        program_number: integer [0-127], GM instrument number (e.g., 0=piano, 40=violin)
        sample_rate: Sample rate for the output audio (default: 44100)

    """
    # Load MIDI file
    pm = pretty_midi.PrettyMIDI(midi_path)

    for instrument in pm.instruments:
        instrument.program = program_number

    # Generate audio using the soundfont
    audio = pm.fluidsynth(sf2_path=soundfont_path, fs=sample_rate)

    # Save audio file
    sf.write(output_path, audio, sample_rate)

def synthesize_dataset():
    soundfont_file = "../diego/artificial/create_dataset/sf/FluidR3_GM.sf2"

    if not os.path.exists(soundfont_file):
        print(f"Soundfont file not found: {soundfont_file}")
        return

    if not os.path.exists('../sintetizados'):
        os.makedirs("../sintetizados", exist_ok=True)

        for instrument, midi_number in INSTRUMENTS.items():
            instrument = instrument.replace(" ", "_").lower()
            os.makedirs(f"../sintetizados/{instrument}", exist_ok=True)
            print(f"Processing instrument: {instrument}")
            
            for i, midi in enumerate(os.listdir('maestro_midi')):
                output_file = f"../sintetizados/{instrument}/{midi.split('.')[0]}.wav"
                if os.path.exists(output_file):
                    continue

                print(f"Processing MIDI file {i + 1}: {midi}")
                midi_path = f"maestro_midi/{midi}"
                render_midi_to_audio(midi_path, soundfont_file, output_file, midi_number)