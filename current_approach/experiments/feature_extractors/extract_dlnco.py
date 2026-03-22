import numpy as np
import librosa
import pretty_midi
import os
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_audio_dlnco(audio_path, fs=22050, feature_rate=344):
    # Since hop_length = 64 (~3 ms at 22.05 kHz), we set feature_rate = fs / hop_length (~344 Hz) to match the same time resolution

    """Extracts DLNCO features from an audio file."""
    # 1. Load audio (mixdown to mono)
    audio, _ = librosa.load(audio_path, sr=fs, mono=True)
    
    # 2. Calculate the desired sequence length based on duration
    duration_sec = len(audio) / fs
    seq_length = int(np.ceil(duration_sec * feature_rate))
    
    # 3. Extract the onset peaks using the IIR filterbank
    peaks_dict = audio_to_pitch_onset_features(f_audio=audio, Fs=fs)
    
    # 4. Compute DLNCO from the peaks
    dlnco = pitch_onset_features_to_DLNCO(
        f_peaks=peaks_dict, 
        feature_sequence_length=seq_length, 
        feature_rate=feature_rate
    )
    
    return dlnco

def get_midi_dlnco(midi_path, duration_sec=None, feature_rate=344):
    # Since hop_length = 64 (~3 ms at 22.05 kHz), we set feature_rate = fs / hop_length (~344 Hz) to match the same time resolution

    """Extracts DLNCO features directly from a MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # If duration isn't provided (e.g., from the audio file), use the MIDI's end time
    if duration_sec is None:
        duration_sec = midi_data.get_end_time()
        
    seq_length = int(np.ceil(duration_sec * feature_rate))
    
    # 1. Manually build the f_peaks dictionary
    temp_peaks = {}
    
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue # DLNCO evaluates tonal pitch, skip percussion
            
        for note in instrument.notes:
            pitch = note.pitch
            # Docs specify the first row must be positions in MILLISECONDS
            onset_ms = note.start * 1000.0 

            # Use MIDI velocity mapped to [0.0, 1.0] for the magnitude. It ranges from 0 to 127, so we normalize it.
            magnitude = note.velocity / 127.0 
            
            if pitch not in temp_peaks:
                temp_peaks[pitch] = {'onsets': [], 'mags': []}
                
            temp_peaks[pitch]['onsets'].append(onset_ms)
            temp_peaks[pitch]['mags'].append(magnitude)
            
    # 2. Format the dictionary exactly as synctoolbox expects (2xN numpy arrays)
    f_peaks = {}
    for pitch, data in temp_peaks.items():
        onsets = np.array(data['onsets'])
        mags = np.array(data['mags'])
        # Stack them into a 2xN array
        f_peaks[pitch] = np.vstack((onsets, mags))
        
    # 3. Compute DLNCO directly from our custom MIDI peaks
    dlnco = pitch_onset_features_to_DLNCO(
        f_peaks=f_peaks, 
        feature_sequence_length=seq_length, 
        feature_rate=feature_rate
    )
    
    return dlnco

def process_audio(audio, synthesized_midi_dir, synthesized_midi_dlnco_dir):
    """Worker function for a single audio file."""
    if audio.endswith(".wav"):
        save_path = os.path.join(synthesized_midi_dlnco_dir, audio.replace(".wav", ".npy"))
        if not os.path.exists(save_path):
            audio_path = os.path.join(synthesized_midi_dir, audio)
            dlnco = get_audio_dlnco(audio_path)
            np.save(save_path, dlnco)
    return audio

def process_midi(midi, corrupt_midi_dir, corrupt_midi_dlnco_dir):
    """Worker function for a single MIDI file."""
    if midi.endswith(".mid"):
        dlnco_save_path = os.path.join(corrupt_midi_dlnco_dir, midi.replace(".mid", ".npy"))
        if not os.path.exists(dlnco_save_path):
            midi_path = os.path.join(corrupt_midi_dir, midi)
            dlnco = get_midi_dlnco(midi_path)
            np.save(dlnco_save_path, dlnco)
    return midi

def main():
    # Paths to the directories
    root_dir = "/home/ldsbrandao/exp/smd"
    corrupt_midi_dir = os.path.join(root_dir, "corrupt_midi")
    synthesized_midi_dir = os.path.join(root_dir, "synthesized_midi")

    corrupt_midi_dlnco_dir = os.path.join(root_dir, "corrupt_midi_dlnco")
    synthesized_midi_dlnco_dir = os.path.join(root_dir, "synthesized_midi_dlnco")

    os.makedirs(corrupt_midi_dlnco_dir, exist_ok=True)
    os.makedirs(synthesized_midi_dlnco_dir, exist_ok=True)

    # Determine optimal number of cores
    max_workers = max(1, int(os.cpu_count() * 0.8))
    print(f"Running on {max_workers} CPU cores.")

    # --- 1. Parallel Audio Processing ---
    audio_files = os.listdir(synthesized_midi_dir)
    tam_audio = len(audio_files)
    print("Starting audio extraction...")
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_audio = [
                executor.submit(process_audio, audio, synthesized_midi_dir, synthesized_midi_dlnco_dir) 
                for audio in audio_files
            ]
            # Track progress as they complete
            for i, future in enumerate(as_completed(futures_audio)):
                try:
                    result = future.result()
                    print(f"{i+1}/{tam_audio} | {result}")
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

    # --- 2. Parallel MIDI Processing ---
    midi_files = os.listdir(corrupt_midi_dir)
    tam_midi = len(midi_files)
    print("Starting MIDI extraction...")
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_midi = [
                executor.submit(process_midi, midi, corrupt_midi_dir, corrupt_midi_dlnco_dir) 
                for midi in midi_files
            ]
            # Track progress as they complete
            for i, future in enumerate(as_completed(futures_midi)):
                try:
                    result = future.result()
                    print(f"{i+1}/{tam_midi} | {result}")
                except Exception as e:
                    print(f"FAILED on MIDI {i + 1}. Error: {e}")
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