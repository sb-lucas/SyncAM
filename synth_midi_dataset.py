from midi_helpers import synthesize_dataset, render_midi_to_audio

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthesize MIDI dataset to audio.")
    parser.add_argument("--samples_dir", type=str, required=True, help="Directory containing MIDI files to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save synthesized audio files")
    parser.add_argument("--soundfont_file", type=str, default="FluidR3_GM.sf2", help="Path to soundfont file")
    parser.add_argument("--instruments", type=str, nargs='*', help="List of instruments to use for synthesis (default: all)", default=None) 
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers to use")
    args = parser.parse_args()

    synthesize_dataset(args.samples_dir, 
                       args.output_dir, 
                       args.instruments,
                       args.soundfont_file, 
                       args.workers)