import numpy as np
import librosa
import pretty_midi
import os
import soundfile as sf
from dotenv import load_dotenv

from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# CENS extraction (audio)
# -----------------------------
def extract_cens_audio(audio_file, original_dir, save_dir, sr=22050, hop_length=64):
    base_name = os.path.splitext(audio_file)[0]
    save_path = os.path.join(save_dir, f"{base_name}.npy")
    
    if os.path.exists(save_path):
        return "SKIPPED (exists): " + audio_file

    print(f"Processing: {audio_file}")
    audio_path = os.path.join(original_dir, audio_file)
    y, sr = librosa.load(audio_path, sr=sr)

    cens = librosa.feature.chroma_cens(
        y=y,
        sr=sr,
        hop_length=hop_length
    )

    np.save(save_path, cens)

    return "OK: " + audio_file
    
def process_audio_folder(input_dir, save_dir, max_workers, description):
    os.makedirs(save_dir, exist_ok=True)

    audio_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".wav", ".mp3")))

    total = len(audio_files)

    print(f"\n{description}")
    print(f"Found {total} audio files")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                extract_cens_audio,
                audio_file,
                input_dir,
                save_dir,
            )
            for audio_file in audio_files
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                print(f"{i}/{total} | {result}")

            except Exception as e:
                print(f"{i}/{total} | FAILED: {e}")


def main():
    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    datasets = ['smd', 'maestro', 'maps/ENSTDkAm', 'maps/ENSTDkCl', 'musicnet']

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, int(cpu_count * 0.8))
    #max_workers = 2
    print(f"Running on {max_workers} CPU cores.")

    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        synthesized_midi_dir = os.path.join(root_dir, "synthesized_midi")
        original_audio_dir = os.path.join(root_dir, "audios")

        corrupt_midi_cens_dir = os.path.join(root_dir, "corrupt_midi_cens")
        original_audio_cens_dir = os.path.join(root_dir, "audios_cens")
        
        # -------------------------
        # 1. Original audio
        # -------------------------
        process_audio_folder(
            input_dir=original_audio_dir,
            save_dir=original_audio_cens_dir,
            max_workers=max_workers,
            description="Original audio CENS",
        )

        # -------------------------
        # 2. Corrupt MIDI audio
        # -------------------------
        process_audio_folder(
            input_dir=synthesized_midi_dir,
            save_dir=corrupt_midi_cens_dir,
            max_workers=max_workers,
            description="Corrupt MIDI audio CENS",
        )
    
        


if __name__ == "__main__":
    main()