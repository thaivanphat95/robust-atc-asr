# Robust ATC ASR

Research code for robust air traffic control speech recognition.

Associated papers:

- [**Contrastive Regularization for Accent-Robust ASR**](https://arxiv.org/abs/2605.03297)
  (INTERSPEECH 2026)
- **Improving Cross-Dataset Robustness of Air Traffic Control Speech Recognition** (ITSC 2026)

The repository provides CTC and supervised contrastive learning workflows for
repeated transcripts, similar transcripts, synthetic speech, and hybrid
training.

## Quick Start

### Install

Create an environment with a CUDA-compatible PyTorch build when using a GPU,
then install the pinned dependencies:

```bash
pip install -r requirements.txt
```

### Train

Training entry points are under `scripts/`:

| Script | Training strategy |
| --- | --- |
| `train_repeated.py` | SupCon using repeated transcripts |
| `train_alternative.py` | Alternating similar-transcript SupCon and CTC-only batches |
| `train_tts.py` | SupCon using original and synthetic transcript pairs |
| `train_hybrid.py` | Similar-transcript, synthetic-pair, and CTC-only batches |

Warm up the CTC head before SupCon training:

```bash
python scripts/warmup.py \
  --prefix Arctic/8fold/0 \
  --backbone w2v2-large
```

Example repeated-transcript training:

```bash
python scripts/train_repeated.py \
  --prefix Arctic/8fold/0 \
  --backbone w2v2-large
```

Example hybrid ATC training:

```bash
python scripts/prepare_similar.py --prefix UWB_ATCOSIM --coverage 0.5

python scripts/train_hybrid.py \
  --prefix UWB_ATCOSIM \
  --backbone w2v2-robust \
  --similar-csv ngram_50.csv
```

Run any script with `--help` to see its complete CLI.

### Validate

Published Hugging Face models are self-contained and include their processor.

Greedy decoding:

```bash
python scripts/validate.py \
  --model-path thaivanphat95/wav2vec2-robust-uwb-supcon-hybrid \
  --csv files/UWB/val.csv
```

4-gram decoding:

```bash
python scripts/validate.py \
  --model-path thaivanphat95/wav2vec2-robust-uwb-supcon-hybrid-4gram \
  --decoder lm \
  --csv files/UWB/val.csv
```

For an older checkpoint with a separately stored processor, add
`--processor-path`.

## Published Models

Models ending in `-4gram` include a 4-gram language model for decoding.

| Training setup | Greedy decoding | 4-gram decoding |
| --- | --- | --- |
| UWB SupCon Hybrid | [Model](https://huggingface.co/thaivanphat95/wav2vec2-robust-uwb-supcon-hybrid) | [Model](https://huggingface.co/thaivanphat95/wav2vec2-robust-uwb-supcon-hybrid-4gram) |
| UWB+ATCOSIM SupCon Hybrid | [Model](https://huggingface.co/thaivanphat95/wav2vec2-robust-uwb-atcosim-supcon-hybrid) | [Model](https://huggingface.co/thaivanphat95/wav2vec2-robust-uwb-atcosim-supcon-hybrid-4gram) |
| L2-ARCTIC repeated SupCon, 8-fold split 0 | [Model](https://huggingface.co/thaivanphat95/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0) | [Model](https://huggingface.co/thaivanphat95/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0-4gram) |
| L2-ARCTIC repeated SupCon, Arabic L1 holdout | [Model](https://huggingface.co/thaivanphat95/wav2vec2-large-l2-arctic-supcon-repeated-arabic-holdout) | [Model](https://huggingface.co/thaivanphat95/wav2vec2-large-l2-arctic-supcon-repeated-arabic-holdout-4gram) |

## Published Dataset

The synthetic speech used by the TTS and hybrid workflows is published as
[Synthetic ATC Speech](https://huggingface.co/datasets/thaivanphat95/synthetic-atc-speech).
It is distributed as WebDataset TAR shards to make the large audio collection
practical to upload and stream.

## Repository Layout

```text
requirements.txt    Pinned Python dependencies
scripts/            Data preparation, training, validation, and warm-up CLIs
src/                Models, data pipelines, samplers, and training utilities
files/              CSV metadata and dataset splits
dataset/            Local audio files; excluded from Git
weights/            Local processors, checkpoints, and models; excluded from Git
```

Dataset split paths are mirrored below `files/` and
`weights/<backbone>/`:

```text
files/Arctic/8fold/0/
weights/w2v2-large/Arctic/8fold/0/model/
weights/w2v2-large/Arctic/8fold/0/Supcon_Repeated/
```

CSV files must contain `audio_filename` and `transcript`. Audio paths may be
absolute or relative to the repository root.

## Training Workflows

### Repeated Transcripts

`train_repeated.py` forms SupCon groups from samples with identical
transcripts. L2-ARCTIC provides repeated prompts across speakers.

### Similar Transcripts

`prepare_similar.py` groups samples using shared n-grams. The generated CSV is
then consumed by `train_alternative.py` or `train_hybrid.py`.

```bash
python scripts/prepare_similar.py --prefix UWB_ATCOSIM --coverage 0.5
python scripts/train_alternative.py --prefix UWB_ATCOSIM --similar-csv ngram_50.csv
```

### Synthetic Pairs

`train_tts.py` forms SupCon groups from original utterances and synthetic audio
sharing the same transcript.

```bash
python scripts/train_tts.py --prefix UWB_ATCOSIM
```

### Hybrid Training

`train_hybrid.py` combines similar-transcript, synthetic-pair, and CTC-only
batches.

## Datasets

Audio datasets are not distributed with this repository. Download each corpus
from its original source, review its terms of use, and place the extracted
audio under `dataset/`. The included CSV metadata under `files/` uses paths
relative to the repository root.

| Dataset | Description | Source | Expected local path |
| --- | --- | --- | --- |
| UWB-ATCC | English speech recorded from real ATC communications. | [Hugging Face](https://huggingface.co/datasets/Jzuluaga/uwb_atcc) | `dataset/UWB/` |
| ATCOSIM | English simulated ATC speech. | [Hugging Face](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus) | `dataset/ATCOSIM/` |
| ATCO2 1-hour test set | Real-world English ATC evaluation set. | [Hugging Face](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h) | `dataset/ATCO2/audios/` |
| L2-ARCTIC | Non-native English read speech with shared prompts across speakers. | [PSI Lab](https://psi.engr.tamu.edu/l2-arctic-corpus/) | `dataset/Arctic/` |

Metadata layout:

```text
files/Arctic/8fold/<fold>/                 Repeated-transcript cross-validation
files/Arctic/l1_holdout/<split>/           L1 holdout evaluation
files/UWB/                                 UWB training and validation metadata
files/UWB_ATCOSIM/                         Combined UWB and ATCOSIM metadata
files/ATCO2/test.csv                       ATCO2 evaluation metadata
```

Synthetic audio used by `train_tts.py` and `train_hybrid.py` is expected under
`dataset/Synthetic/` and must match the paths listed in each `synthetic.csv`.

## Citation

If you use this code, the published models, or the synthetic dataset, please
cite:

```bibtex
@article{thai2026contrastive,
  title={Contrastive Regularization for Accent-Robust ASR},
  author={Thai, Van-Phat and Dhruv, Aradhya and Pham, Duc-Thinh and Alam, Sameer},
  journal={arXiv preprint arXiv:2605.03297},
  year={2026},
  doi={10.48550/arXiv.2605.03297}
}
```

## License

The original source code in `scripts/` and `src/` is licensed under the
[Apache License 2.0](LICENSE).

This code license does not cover datasets, transcripts, metadata derived from
datasets, synthetic audio, or model weights. These materials remain governed by
their original licenses and terms. See [DATA_LICENSES.md](DATA_LICENSES.md) for
dataset-specific attribution and licensing notes.
