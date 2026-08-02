import argparse
import os
import sys
import torch
import subprocess
from dotenv import load_dotenv

def run_clamp3_embd(input_dir, output_dir, clamp_script_path):
    """Executes CLaMP 3 feature extraction script with PATH & PYTHONPATH configured."""
    clamp_dir = os.path.dirname(os.path.abspath(clamp_script_path))
    code_dir = os.path.join(clamp_dir, "code")
    
    # Path to Conda bin directory (/home/ldsbrandao/exp/miniconda3/envs/clamp3/bin)
    python_bin_dir = os.path.dirname(sys.executable)

    env = os.environ.copy()
    
    # 1. Force os.system("python ...") inside CLaMP 3 to use the active Conda python
    existing_path = env.get("PATH", "")
    env["PATH"] = f"{python_bin_dir}:{existing_path}"

    # 2. Add CLaMP 3 directories to PYTHONPATH
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{clamp_dir}:{code_dir}:{existing_pythonpath}"

    cmd = [
        sys.executable,
        clamp_script_path,
        input_dir,
        output_dir,
        "--get_global"
    ]
    
    print(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=clamp_dir, env=env)

def main():
    # ----------------------------- Argparse Setup ---------------------------------- #
    parser = argparse.ArgumentParser(description="Batch Audio Transcription Pipeline")
    parser.add_argument(
        "--original",
        action="store_true",
        help="Extract only from the original data."
    )
    args = parser.parse_args()
    # ----------------------------- Environment & Config ---------------------------------- #
    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    if args.original:
        datasets = ['maestro', 'smd', 'maps_synth', 'maps_ENSTDkAm', 'maps_ENSTDkCl', 'musicnet']
    else:
        datasets = ['maestro', 'smd', 'maps_synth']

    #----------------------Load the models once--------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running in strictly sequential mode (0 background workers).")
    print(f"Running on: {device}")
    
    CLAMP3_EMBD_PATH = "/home/ldsbrandao/exp/clamp3/clamp3_embd.py"

    #main loop
    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        original_audio_clamp_dir = os.path.join(root_dir, "audios_clamp")
        original_audio_dir = os.path.join(root_dir, "audios")

        if args.original:
            corrupt_midi_dir = os.path.join(root_dir, "midis")
            corrupt_midi_clamp_dir = os.path.join(root_dir, "original_midi_clamp")
        else:
            corrupt_midi_dir = os.path.join(root_dir, "corrupt_midi")
            corrupt_midi_clamp_dir = os.path.join(root_dir, "corrupt_midi_clamp")

        #----------------------Extract CLAMP from original audios----------------#
        run_clamp3_embd(original_audio_dir, original_audio_clamp_dir, CLAMP3_EMBD_PATH)        

        #----------------------Extract CLAMP from corrupt MIDIs----------------#
        run_clamp3_embd(corrupt_midi_dir, corrupt_midi_clamp_dir, CLAMP3_EMBD_PATH)

if __name__ == "__main__":
    main()