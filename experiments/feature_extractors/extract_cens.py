import numpy as np
import librosa
import pretty_midi
import os
import soundfile as sf

from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# CENS extraction (audio)
# -----------------------------
def extract_cens_audio(audio_file, original_dir, save_dir, sr=22050, hop_length=64):
    save_path = os.path.join(save_dir, audio_file.replace(".wav", ".npy"))
    if os.path.exists(save_path):
        return "SKIPPED (exists): " + audio_file

    audio_path = os.path.join(original_dir, audio_file)
    y, sr = librosa.load(audio_path, sr=sr)

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    cens = librosa.feature.chroma_cens(
        C=chroma,
        hop_length=hop_length
    )

    np.save(save_path, cens)

    return "OK: " + audio_file
    

def main():
    root_dir = "/home/ldsbrandao/exp/smd"
    corrupt_midi_audio_dir = os.path.join(root_dir, "corrupt_midi_audio")
    synthesized_midi_dir = os.path.join(root_dir, "synthesized_midi")

    corrupt_midi_cens_dir = os.path.join(root_dir, "corrupt_midi_cens")
    synthesized_midi_cens_dir = os.path.join(root_dir, "synthesized_midi_cens")

    os.makedirs(corrupt_midi_cens_dir, exist_ok=True)
    os.makedirs(synthesized_midi_cens_dir, exist_ok=True)

    #define the number of workers for parallel processing
    max_workers = max(1, int(os.cpu_count() * 0.2))
    print(f"Running on {max_workers} CPU cores.")



    #Extract CENS of synthesized audio from original MIDIs
    tam_original = len(os.listdir(synthesized_midi_dir))
    audio_files = os.listdir(synthesized_midi_dir)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_audio = [
                executor.submit(extract_cens_audio, audio, synthesized_midi_dir, synthesized_midi_cens_dir) 
                for audio in audio_files
            ]
            # Track progress as they complete
            for i, future in enumerate(as_completed(futures_audio)):
                try:
                    result = future.result()
                    print(f"{i+1}/{tam_original} | {result}")
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



    #Extract CENS of synthesized audio from corrupt MIDIs
    tam_corrupt = len(os.listdir(corrupt_midi_audio_dir))
    audio_files = os.listdir(corrupt_midi_audio_dir)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_audio = [
                executor.submit(extract_cens_audio, audio, corrupt_midi_audio_dir, corrupt_midi_cens_dir) 
                for audio in audio_files
            ]
            # Track progress as they complete
            for i, future in enumerate(as_completed(futures_audio)):
                try:
                    result = future.result()
                    print(f"{i+1}/{tam_corrupt} | {result}")
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