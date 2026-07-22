import numpy as np
import librosa
import pretty_midi
import os
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

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
            # Predict the frame index synctoolbox will calculate
            expected_index = int(np.round(note.start * feature_rate))
            
            # DROP the note if it falls outside the allocated feature matrix
            if expected_index >= seq_length:
                continue

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
    file_extension = ".wav" if audio.endswith(".wav") else ".mp3"
    save_path = os.path.join(synthesized_midi_dlnco_dir, audio.replace(file_extension, ".npy"))

    if not os.path.exists(save_path):
        audio_path = os.path.join(synthesized_midi_dir, audio)
        dlnco = get_audio_dlnco(audio_path)
        np.save(save_path, dlnco)

    return audio

def process_midi(midi, corrupt_midi_dir, corrupt_midi_dlnco_dir):
    """Worker function for a single MIDI file."""
    file_extension = ".mid" if midi.endswith(".mid") else ".midi"
    dlnco_save_path = os.path.join(corrupt_midi_dlnco_dir, midi.replace(file_extension, ".npy"))

    if not os.path.exists(dlnco_save_path):
        midi_path = os.path.join(corrupt_midi_dir, midi)
        dlnco = get_midi_dlnco(midi_path)
        np.save(dlnco_save_path, dlnco)

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

        corrupt_midi_dlnco_dir = os.path.join(root_dir, "corrupt_midi_dlnco")
        original_audio_dlnco_dir = os.path.join(root_dir, "audios_dlnco")
        os.makedirs(corrupt_midi_dlnco_dir, exist_ok=True)
        os.makedirs(original_audio_dlnco_dir, exist_ok=True)
        

        # -------------------------
        # Audio DLNCO
        # -------------------------

        audio_files = [f for f in os.listdir(original_audio_dir) if f.endswith(".wav") or f.endswith(".mp3")]

        run_parallel(
            files=audio_files,
            worker_function=process_audio,
            worker_args=(
                original_audio_dir,
                original_audio_dlnco_dir,
            ),
            max_workers=max_workers,
            description="Audio DLNCO",
            chunksize=1,
        )

        # -------------------------
        # MIDI DLNCO
        # -------------------------

        midi_files = [f for f in os.listdir(corrupt_midi_dir) if f.endswith(".mid") or f.endswith(".midi")]

        run_parallel(
            files=midi_files,
            worker_function=process_midi,
            worker_args=(
                corrupt_midi_dir,
                corrupt_midi_dlnco_dir,
            ),
            max_workers=max_workers,
            description="MIDI DLNCO",
            chunksize=10,
        )


if __name__ == "__main__":
    main()