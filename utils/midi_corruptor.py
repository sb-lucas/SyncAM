import pretty_midi
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import os
from dotenv import load_dotenv

from midi_helpers import render_midi_to_audio

def add_remove(midi, num_corrupt_notes):
    # Split the workload: half for removal, half for addition
    num_to_remove = num_corrupt_notes // 2
    num_to_add = num_corrupt_notes - num_to_remove

    # Get the total time of the song to place notes in a valid range
    max_time = midi.get_end_time()

    # Collect all notes to find candidates for removal
    all_notes = []
    for instrument in midi.instruments:
        all_notes.extend(instrument.notes)

    # REMOVE NOTES
    if all_notes:
        notes_to_delete = random.sample(all_notes, min(num_to_remove, len(all_notes)))
        for instrument in midi.instruments:
            # Reconstruct the note list excluding the ones we chose to delete
            instrument.notes = [n for n in instrument.notes if n not in notes_to_delete]

    # ADD NOTES
    # We'll add them to the first instrument found
    if midi.instruments:
        target_inst = midi.instruments[0]
        
        piano_min, piano_max = 21, 108

        for _ in range(num_to_add):
            # Pick a random start time within the song
            start_time = random.uniform(0, max_time)
            # "Natural" piano duration: between 0.1s (staccato) and 1.5s (legato/sustained)
            duration = random.uniform(0.1, 1.5)
            end_time = start_time + duration
            
            new_note = pretty_midi.Note(
                velocity=random.randint(60, 100), # Realistic piano velocity
                pitch=random.randint(piano_min, piano_max),
                start=start_time,
                end=end_time
            )
            target_inst.notes.append(new_note)
            
        # Optional: Sort notes by start time (good practice for MIDI consistency)
        target_inst.notes.sort(key=lambda x: x.start)

    return midi

def pitch_swap(midi, num_corrupt_notes):
    # Flatten all notes from all (piano) instruments into one list
    all_notes = []
    for instrument in midi.instruments:
        all_notes.extend(instrument.notes)

    # Randomly select the specific note objects to corrupt
    notes_to_corrupt = random.sample(all_notes, num_corrupt_notes)

    # Standard Piano Range: A0 (21) to C8 (108)
    piano_min, piano_max = 21, 108

    for note in notes_to_corrupt:
        current_pitch = note.pitch
        
        # Create a list of all possible piano keys EXCEPT the current one
        valid_pitches = [p for p in range(piano_min, piano_max + 1) if p != current_pitch]
        
        # Assign a new random pitch from the valid piano range
        note.pitch = random.choice(valid_pitches)

    return midi

def onset_corruption(midi, shift_amount):
    # convert to seconds
    shift_amount = shift_amount / 1000.0

    for instrument in midi.instruments:
        for note in instrument.notes:
            shift = random.choice([-shift_amount, shift_amount])

            duration = note.end - note.start
            new_start = max(0, note.start + shift)
            new_end = new_start + duration

            note.start = new_start
            note.end = new_end

    return midi

def apply_corruptions(midi_filename, input_path, save_folder):
    midi = pretty_midi.PrettyMIDI(input_path)
    total_notes = sum(len(instrument.notes) for instrument in midi.instruments)

    file_name = os.path.splitext(midi_filename)[0]

    for CA_percentage in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        pitch_path = os.path.join(save_folder, f'{file_name}_pitch_{CA_percentage}.mid')
        add_remove_path = os.path.join(save_folder, f'{file_name}_addRemove_{CA_percentage}.mid')
        onset_path = os.path.join(save_folder, f'{file_name}_onset_{CA_percentage}.mid')

        if os.path.exists(pitch_path) and os.path.exists(add_remove_path) and os.path.exists(onset_path):
            print(f'Skipping {file_name} for CA: {CA_percentage} as all corrupted files already exist.')
            continue
        

        print(f'for ca: {CA_percentage}')
        num_corrupt_notes = total_notes * CA_percentage // 100
        
        S = CA_percentage / 100
        shift_amount = 500 * (S ** 2)  # in ms, matches S = (Δt/500)^(1/2)

        # 1. Pitch Swap (using a fresh copy)
        pitch_midi = pitch_swap(copy.deepcopy(midi), num_corrupt_notes)
        pitch_midi.write(pitch_path)

        # 2. Add/Remove (using a fresh copy)
        add_remove_midi = add_remove(copy.deepcopy(midi), num_corrupt_notes*2)
        add_remove_midi.write(add_remove_path)

        # 3. Onset Shift (using a fresh copy)
        onset_midi = onset_corruption(copy.deepcopy(midi), shift_amount)
        onset_midi.write(onset_path)

def main():

    #we expect a folder of the original dataset
    #in it, there should be two subfolders: 'audio' and 'midi'
    #we are going to corrupt each midi file and save into a another subfolder, called 'corrupt_midi'
    #finally, we synthesize the original midi, form the 'midi' folder and save into a fourth subfolder called 'synthesized_midi'

    random.seed(42)

    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    SOUNDFONT_PATH = os.environ["SOUNDFONT_PATH"]
    datasets = ['maestro', 'smd', 'maps_synth']

    for dataset in datasets:
        print(f'Processing dataset: {dataset}')

        root_folder = os.path.join(DATASETS_FOLDER, dataset)
        midi_folder = os.path.join(root_folder, 'midis')
        corrupt_midi_folder = os.path.join(root_folder, 'corrupt_midi')
        os.makedirs(corrupt_midi_folder, exist_ok=True)

        for midi_file in os.listdir(midi_folder):
            print(f'for file: {midi_file}')
            midi_path = os.path.join(midi_folder, midi_file)

            try:
                apply_corruptions(midi_file, midi_path, corrupt_midi_folder)
            except Exception as e:
                print(f"FAILED: {midi_path}")
                print(f"ERROR: {e}")
                print(f"TYPE: {type(e).__name__}")

if __name__ ==  "__main__":
    main()



