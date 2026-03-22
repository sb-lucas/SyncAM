import numpy as np
import librosa
import pretty_midi
import os
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_audio_spectrogram(audio_path, fs=22050, hop_length=64, n_fft=512):
    """Extracts Spectrogram features from an audio file."""
    
    # 1. Load the audio file (forcing mono and the target sample rate)
    audio, _ = librosa.load(audio_path, sr=fs, mono=True)
    
    # 2. Compute the Short-Time Fourier Transform (STFT)
    # n_fft defines the frequency resolution window. 
    stft_matrix = librosa.stft(audio, hop_length=hop_length, n_fft=n_fft)
    
    # 3. Get the magnitude spectrogram (STFT returns complex numbers)
    spectrogram = np.abs(stft_matrix)
    
    # 4. (Optional but recommended) Convert to Log/Decibel scale
    # Neural networks and standard distance metrics usually perform better 
    # on log-scaled amplitudes rather than linear amplitudes.
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Return spectrogram_db if you want the log scale, otherwise return spectrogram
    return spectrogram_db

def process_audio(audio, original_dir, save_spectrogram_dir):
    """Worker function for a single audio file."""

    if audio.endswith(".wav"):
        save_path = os.path.join(save_spectrogram_dir, audio.replace(".wav", ".npy"))

        if not os.path.exists(save_path):
            audio_path = os.path.join(original_dir, audio)
            spectrogram = get_audio_spectrogram(audio_path)
            np.save(save_path, spectrogram)

    return audio

def main():
    root_dir = "/home/ldsbrandao/exp/smd"
    corrupt_midi_audio_dir = os.path.join(root_dir, "corrupt_midi_audio")
    synthesized_midi_dir = os.path.join(root_dir, "synthesized_midi")

    corrupt_midi_spectrogram_dir = os.path.join(root_dir, "corrupt_midi_spectrogram")
    synthesized_midi_spectrogram_dir = os.path.join(root_dir, "synthesized_midi_spectrogram")

    os.makedirs(corrupt_midi_spectrogram_dir, exist_ok=True)
    os.makedirs(synthesized_midi_spectrogram_dir, exist_ok=True)

    #define the number of workers for parallel processing
    max_workers = max(1, int(os.cpu_count() * 0.5))
    print(f"Running on {max_workers} CPU cores.")

    #Extract Spectrogram of synthesized audio from original MIDIs
    tam_original = len(os.listdir(synthesized_midi_dir))
    audio_files = os.listdir(synthesized_midi_dir)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_audio = [
                executor.submit(process_audio, audio, synthesized_midi_dir, synthesized_midi_spectrogram_dir) 
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



    #Extract Spectrogram of synthesized audio from corrupt MIDIs
    tam_corrupt = len(os.listdir(corrupt_midi_audio_dir))
    audio_files = os.listdir(corrupt_midi_audio_dir)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_audio = [
                executor.submit(process_audio, audio, corrupt_midi_audio_dir, corrupt_midi_spectrogram_dir) 
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