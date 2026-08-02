import argparse
import os
import subprocess
from dotenv import load_dotenv

# ----------------------------- Argparse Setup ---------------------------------- #
parser = argparse.ArgumentParser(description="Batch Audio Transcription Pipeline")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["basic_pitch", "byte_dance", "transkun"],
    help="Select the transcription model to run.",
)
parser.add_argument(
    "--original",
    action="store_true",
    help="Extract only from the original data."
)
args = parser.parse_args()

# ----------------------------- Environment & Config ---------------------------------- #
load_dotenv()
DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
SOUNDFONT_PATH = os.environ["SOUNDFONT_PATH"]
if args.original:
    datasets = ['maestro', 'smd', 'maps_synth', 'maps_ENSTDkAm', 'maps_ENSTDkCl', 'musicnet']
else:
    datasets = ['maestro', 'smd', 'maps_synth']


# -------------------------------- Basic Pitch ---------------------------------------- #
if args.model == "basic_pitch":
    # Prevent TensorFlow from stealing GPU memory from PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

    for dataset in datasets:
        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        audios_dir = os.path.join(root_dir, "audios")
        audios_transcribed_dir = os.path.join(root_dir, "audios_transcribed_bp")
        os.makedirs(audios_transcribed_dir, exist_ok=True)

        audio_files = [f for f in os.listdir(audios_dir) if not f.startswith(".")]
        tam_original = len(audio_files)

        for i, file in enumerate(audio_files):
            print(f"[{i+1}/{tam_original}] Basic Pitch processing: {file}")
            file_path = os.path.join(audios_dir, file)
            file_basename = os.path.splitext(file)[0]
            output_midi_path = os.path.join(audios_transcribed_dir, file_basename + ".mid")
            if os.path.exists(output_midi_path):
                print(f"Output MIDI already exists for {file}, skipping.")
                continue

            model_output, midi_data, note_events = predict(file_path, basic_pitch_model)
            midi_data.write(output_midi_path)

# -------------------------------- ByteDance ------------------------------------------ #
elif args.model == "byte_dance":
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate

    transcriptor = PianoTranscription(device="cuda")

    for dataset in datasets:
        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        audios_dir = os.path.join(root_dir, "audios")
        audios_transcribed_dir = os.path.join(root_dir, "audios_transcribed_bd")
        os.makedirs(audios_transcribed_dir, exist_ok=True)

        audio_files = [f for f in os.listdir(audios_dir) if not f.startswith(".")]
        tam_original = len(audio_files)

        for i, file in enumerate(audio_files):
            print(f"[{i+1}/{tam_original}] ByteDance processing: {file}")
            audio_file_path = os.path.join(audios_dir, file)
            file_basename = os.path.splitext(file)[0]
            save_path = os.path.join(audios_transcribed_dir, file_basename + ".mid")

            if os.path.exists(save_path):
                print(f"Output MIDI already exists for {file}, skipping.")
                continue

            audio, _ = librosa.load(audio_file_path, sr=sample_rate, mono=True)
            transcribed_dict = transcriptor.transcribe(audio, save_path)

# -------------------------------- Transkun ------------------------------------------- #
elif args.model == "transkun":
    for dataset in datasets:
        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        audios_dir = os.path.join(root_dir, "audios")
        audios_transcribed_dir = os.path.join(root_dir, "audios_transcribed_tk")
        os.makedirs(audios_transcribed_dir, exist_ok=True)

        audio_files = [f for f in os.listdir(audios_dir) if not f.startswith(".")]
        tam_original = len(audio_files)

        for i, file in enumerate(audio_files):
            print(f"[{i+1}/{tam_original}] Transkun processing: {file}")
            audio_path = os.path.join(audios_dir, file)
            file_basename = os.path.splitext(file)[0]
            output_midi = os.path.join(audios_transcribed_dir, file_basename + ".mid")

            if os.path.exists(output_midi):
                print(f"Output MIDI already exists for {file}, skipping.")
                continue

            cmd = ["transkun", audio_path, output_midi, "--device", "cuda"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error processing {file}:\n{result.stderr}")