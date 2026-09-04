# Assessing Alignment in Automatic Piano Transcription Datasets: A Benchmark for Scoring Methods

Official repository for the 2026 LAMIR paper.

## Description

This work provides a framework for evaluating audio–MIDI alignment in Automatic Music Transcription (AMT) datasets. It introduces STI, an alignment scoring method based on AMT models, and benchmarks it against existing approaches using controlled temporal, pitch, and structural MIDI corruptions. We also include experiments applying these methods to widely used piano datasets to assess their relative alignment quality.

## Structure

```text
SyncAM/
├── experiments/
│   ├── scores/                    # Code for every base scoring method (CS, DTW, STI, etc.)
│   ├── get_scores.py              # Script to run all scoring methods on all audio–MIDI pairs
│   ├── evaluate_scores.py         # Script to evaluate scoring methods using the four proposed metrics
│   └── evaluate_datasets.py       # Script to use the best scoring methods to evaluate AMT datasets
│
├── utils/
│   ├── feature_extractors/        # Scripts to extract features for scoring methods (CENS, embeddings, etc.)
│   └── midi_corruptor.py          # Script to apply controlled corruptions to the MIDI files of the datasets
│
├── .gitignore
├── README.md                      # Project documentation (this file)
└── requirements.txt               # Dependencies for reproducibility
```

## How to Start

### Installing Requirements

The code was developed and tested with **Python 3.11.15**.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Reproducing Results

To reproduce the results, first download the data:

* **MAESTRO:** https://magenta.withgoogle.com/datasets/maestro
* **SMD:** https://zenodo.org/records/13753319
* **MusicNet:** https://zenodo.org/records/5120004
* **MAPS:** https://zenodo.org/records/18160555

Corrupt the MIDIs using `utils/midi_corruptor.py`. Synthesize them with `utils/feature_extractors/synthesize_midi.py`, then extract all required features using the scripts in `utils/feature_extractors/`.

If you want to extract features for the original audio–MIDI pairs, do not corrupt the MIDIs and extract the features using the `--original` flag.

Get the scores for every scoring method using `experiments/get_scores.py`. Finally, evaluate the methods with `experiments/evaluate_scores.py` and evaluate the alignment of the original datasets' audio–MIDI pairs with `experiments/evaluate_datasets.py`. The last two scripts reproduce the tables presented in the paper.