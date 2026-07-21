import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import pretty_midi
import os
import random
import pandas as pd
from tempfile import TemporaryDirectory
import pretty_midi
import soundfile as sf
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def render_one_midi(midi_filename, midi_dir, save_dir, soundfont_path, program_number=0, sample_rate=44100):
    """
    Render a MIDI file to audio using a soundfont
    
    Args:
        midi_path: Path to the input MIDI file
        soundfont_path: Path to the soundfont (.sf2) file
        output_path: Path for the output audio file (.wav)
        program_number: integer [0-127], GM instrument number (e.g., 0=piano, 40=violin)
        sample_rate: Sample rate for the output audio (default: 44100)

    """
    midi_path = os.path.join(midi_dir, midi_filename)
    output_filename = os.path.splitext(midi_filename)[0] + ".wav"
    output_path = os.path.join(save_dir, output_filename)

    # Skip already-rendered files
    if os.path.exists(output_path):
        return f"SKIPPED: {midi_filename}"

    # Load MIDI file
    pm = pretty_midi.PrettyMIDI(midi_path)

    for instrument in pm.instruments:
        instrument.program = program_number

    # Generate audio using the soundfont
    audio = pm.fluidsynth(synthesizer=soundfont_path, fs=sample_rate)

    # Save audio file
    sf.write(output_path, audio, sample_rate)

    return f"OK: {midi_filename}"

def render_dataset(midi_dir, save_dir, soundfont_path, max_workers):
    os.makedirs(save_dir, exist_ok=True)

    midi_files = [
        f
        for f in os.listdir(midi_dir)
        if f.lower().endswith((".mid", ".midi"))
    ]

    total = len(midi_files)

    print(f"Rendering {total} MIDI files")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                render_one_midi,
                midi_filename,
                midi_dir,
                save_dir,
                soundfont_path,
            )
            for midi_filename in midi_files
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                print(f"{i}/{total} | {result}")

            except Exception as e:
                print(f"{i}/{total} | FAILED: {e}")


def main():
    random.seed(42)

    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    SOUNDFONT_PATH = os.environ["SOUNDFONT_PATH"]
    datasets = ['maestro', 'maps/ENSTDkAm', 'maps/ENSTDkCl', 'musicnet', 'smd']

    max_workers = int(os.cpu_count() * 0.8)
    print(f"Using {max_workers} workers for rendering")

    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        midi_dir = os.path.join(DATASETS_FOLDER, dataset, "corrupt_midi")
        save_dir = os.path.join(DATASETS_FOLDER, dataset, "synthesized_midi")

        render_dataset(midi_dir=midi_dir, save_dir=save_dir, soundfont_path=SOUNDFONT_PATH, max_workers=max_workers)

if __name__ == "__main__":
    main()