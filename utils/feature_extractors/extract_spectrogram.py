import numpy as np
import librosa
import pretty_midi
import os
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv


def get_audio_spectrogram(audio_path, fs=22050, hop_length=64, n_fft=512):
    """Extracts Spectrogram features from an audio file."""
    
    # 1. Load the audio file (forcing mono and the target sample rate)
    audio, _ = librosa.load(audio_path, sr=fs, mono=True)
    
    # 2. Compute the Short-Time Fourier Transform (STFT)
    # n_fft defines the frequency resolution window. 
    stft_matrix = librosa.stft(audio, hop_length=hop_length, n_fft=n_fft)
    
    # 3. Get the magnitude spectrogram (STFT returns complex numbers)
    spectrogram = np.abs(stft_matrix)
    
    return spectrogram

def process_audio(audio, original_dir, save_spectrogram_dir):
    """Worker function for a single audio file."""

    base_name = os.path.splitext(audio)[0]
    save_path = os.path.join(save_spectrogram_dir, f"{base_name}.npy")

    if not os.path.exists(save_path):
        audio_path = os.path.join(original_dir, audio)
        spectrogram = get_audio_spectrogram(audio_path)
        np.save(save_path, spectrogram)

    return audio

def process_audio_folder(input_dir, save_dir, max_workers, description):
    os.makedirs(save_dir, exist_ok=True)

    audio_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".wav", ".mp3")))

    total = len(audio_files)

    print(f"\n{description}")
    print(f"Found {total} audio files")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_audio,
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
    datasets = ['maestro', 'smd', 'maps/ENSTDkAm', 'maps/ENSTDkCl', 'musicnet']

    cpu_count = os.cpu_count() or 1
    max_workers = max(1, int(cpu_count * 0.8))
    print(f"Running on {max_workers} CPU cores.")

    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        synthesized_midi_dir = os.path.join(root_dir, "synthesized_midi")
        original_audio_dir = os.path.join(root_dir, "audios")

        corrupt_midi_spectrogram_dir = os.path.join(root_dir, "corrupt_midi_spectrogram")
        original_audio_spectrogram_dir = os.path.join(root_dir, "audios_spectrogram")
        
        # -------------------------
        # 1. Original audio
        # -------------------------
        process_audio_folder(
            input_dir=original_audio_dir,
            save_dir=original_audio_spectrogram_dir,
            max_workers=max_workers,
            description="Original audio spectrogram",
        )

        # -------------------------
        # 2. Corrupt MIDI audio
        # -------------------------
        process_audio_folder(
            input_dir=synthesized_midi_dir,
            save_dir=corrupt_midi_spectrogram_dir,
            max_workers=max_workers,
            description="Corrupt MIDI audio spectrogram",
        )
    
if __name__ == "__main__":
    main()