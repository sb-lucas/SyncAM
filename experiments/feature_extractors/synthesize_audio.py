import librosa
import os
import numpy as np
import soundfile as sf
import pretty_midi
from concurrent.futures import ProcessPoolExecutor, as_completed

def synthesize_midi_to_audio(midi_file, original_dir, save_dir, sr=22050):
    soundfont_path = '/home/ldsbrandao/exp/FluidR3_GM.sf2'

    midi_path = os.path.join(original_dir, midi_file)
    pm = pretty_midi.PrettyMIDI(midi_path)

    y = pm.fluidsynth(sf2_path=soundfont_path, fs=sr)

    base = os.path.splitext(midi_file)[0]
    save_path = os.path.join(save_dir, base + ".wav")

    sf.write(save_path, y, sr)

    return f"OK: {midi_file}"

def main():
    # Paths to the directories
    root_dir = "/home/ldsbrandao/exp/smd"
    corrupt_midi_dir = os.path.join(root_dir, "corrupt_midi")

    corrupt_midi_audio_dir = os.path.join(root_dir, "corrupt_midi_audio")
    os.makedirs(corrupt_midi_audio_dir, exist_ok=True)

    # Determine optimal number of cores
    max_workers = max(1, int(os.cpu_count() * 0.7))
    print(f"Running on {max_workers} CPU cores.")

    # Process MIDI files in parallel
    midi_files = os.listdir(corrupt_midi_dir)
    tam_midi = len(midi_files)
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_audio = [
                executor.submit(synthesize_midi_to_audio, midi, corrupt_midi_dir, corrupt_midi_audio_dir) 
                for midi in midi_files
            ]
            # Track progress as they complete
            for i, future in enumerate(as_completed(futures_audio)):
                try:
                    result = future.result()
                    print(f"{i+1}/{tam_midi} | {result}")
                except Exception as e:
                    print(f"FAILED on audio {i + 1}. Error: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user. Shutting down.")
        # Cancel pending tasks + stop accepting new ones
        executor.shutdown(wait=False, cancel_futures=True)

        # Force kill remaining child processes
        import multiprocessing
        for p in multiprocessing.active_children():
            p.terminate()

        print("All workers terminated.")


if __name__ == "__main__":
    main()