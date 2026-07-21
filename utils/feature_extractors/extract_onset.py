import numpy as np
import pretty_midi
import mir_eval
import madmom
import os
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", message=".*align should be passed.*")

# Override the hardcoded safety limit (1e7)
pretty_midi.pretty_midi.MAX_TICK = 1e8

def get_onsets_from_midi(midi_file, combine=0.03):
    
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Using a set to automatically handle duplicates across tracks
    onsets = set()
    for instrument in midi_data.instruments:
        # Skip drum tracks if you only want melodic alignment
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            onsets.add(note.start)
            
    onsets_sorted = sorted(list(onsets))
    
    # Merging close onsets (Vectorized-ish approach)
    if not onsets_sorted:
        return np.array([])
        
    merged = [onsets_sorted[0]]
    for onset in onsets_sorted[1:]:
        if onset - merged[-1] > combine:
            merged.append(onset)
            
    return np.array(merged)

#_onset_proc = None
#_peak_proc = None

def get_madmom_onsets(audio_file, method="cnn"):
    """
    Extract onset times using madmom's onset processors.
    """
    #global _onset_proc, _peak_proc

    # Lazy-load: The worker checks if it already loaded the model. 
    # If not, it loads it ONCE and keeps it in memory for the next files.
    #if _onset_proc is None:
    if method.lower() == "cnn":
        _onset_proc = madmom.features.onsets.CNNOnsetProcessor()
    elif method.lower() == "rnn":
        _onset_proc = madmom.features.onsets.RNNOnsetProcessor()
    elif method.lower() == "spectral":
        _onset_proc = madmom.features.onsets.SpectralOnsetProcessor()
    else:
        raise ValueError("method must be one of ['cnn', 'rnn', 'spectral']")
    
    _peak_proc = madmom.features.onsets.OnsetPeakPickingProcessor()

    # --- Compute activation (novelty) ---
    activation = _onset_proc(audio_file)

    # --- Peak picking ---
    onsets = _peak_proc(activation)

    return onsets


def process_audio(audio, original_audio_dir, original_audio_onsets_dir):
    """Worker function for a single audio file."""
    file_extension = ".wav" if audio.endswith(".wav") else ".mp3"
    save_path = os.path.join(original_audio_onsets_dir, audio.replace(file_extension, ".npy"))

    if not os.path.exists(save_path):
        audio_path = os.path.join(original_audio_dir, audio)
        onsets = get_madmom_onsets(audio_path)
        np.save(save_path, onsets)

    return audio

def process_midi(midi, corrupt_midi_dir, corrupt_midi_onsets_dir):
    """Worker function for a single MIDI file."""
    file_extension = ".mid" if midi.endswith(".mid") else ".midi"
    onsets_save_path = os.path.join(corrupt_midi_onsets_dir, midi.replace(file_extension, ".npy"))

    if not os.path.exists(onsets_save_path):
        midi_path = os.path.join(corrupt_midi_dir, midi)
        onsets = get_onsets_from_midi(midi_path)
        np.save(onsets_save_path, onsets)

    return midi


def run_parallel(files, worker_function, worker_args, max_workers, description, chunksize=1):
    total = len(files)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            worker_function,
            files,
            *[
                [arg] * total
                for arg in worker_args
            ],
            chunksize=chunksize,
        )

        for i, result in enumerate(results, start=1):
            print(f"{description}: "f"{i}/{total} | {result}")

def main():
    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    datasets = ['maestro', 'maps/ENSTDkAm', 'maps/ENSTDkCl', 'musicnet', 'smd']

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, int(cpu_count * 0.8))
    print(f"Running on {max_workers} CPU cores.")

    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        corrupt_midi_dir = os.path.join(root_dir, "corrupt_midi")
        original_audio_dir = os.path.join(root_dir, "audios")

        corrupt_midi_onsets_dir = os.path.join(root_dir, "corrupt_midi_onsets")
        original_audio_onsets_dir = os.path.join(root_dir, "audios_onsets")
        os.makedirs(corrupt_midi_onsets_dir, exist_ok=True)
        os.makedirs(original_audio_onsets_dir, exist_ok=True)

        # -------------------------
        # Audio onsets
        # -------------------------

        audio_files = [f for f in os.listdir(original_audio_dir) if f.endswith(".wav") or f.endswith(".mp3")]

        run_parallel(
            files=audio_files,
            worker_function=process_audio,
            worker_args=(
                original_audio_dir,
                original_audio_onsets_dir,
            ),
            max_workers=max_workers,
            description="Audio onsets",
            chunksize=1,
        )

        # -------------------------
        # MIDI onsets
        # -------------------------

        midi_files = [f for f in os.listdir(corrupt_midi_dir) if f.endswith(".mid") or f.endswith(".midi")]

        run_parallel(
            files=midi_files,
            worker_function=process_midi,
            worker_args=(
                corrupt_midi_dir,
                corrupt_midi_onsets_dir,
            ),
            max_workers=max_workers,
            description="MIDI onsets",
            chunksize=10,
        )

if __name__ == "__main__":
    main()