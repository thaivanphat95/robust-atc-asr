# Accent ASR

Training utilities for accent-robust CTC ASR with supervised contrastive
learning. The repository supports repeated-transcript, similar-transcript,
synthetic-pair, and hybrid training.

## Repository Layout

```text
files/      CSV metadata and dataset splits
scripts/    Data preparation, training, validation, and warm-up commands
src/        Shared models, data pipelines, samplers, and training utilities
dataset/    Local audio files; excluded from Git
weights/    Local processors, checkpoints, and trained models; excluded from Git
```

Dataset split paths are mirrored below `files/` and
`weights/<backbone>/`. For example:

```text
files/Arctic/8fold/0/
weights/w2v2-large/Arctic/8fold/0/model/
weights/w2v2-large/Arctic/8fold/0/Supcon_Repeated/
```

CSV files must contain `audio_filename` and `transcript`. Audio paths may be
absolute or relative to the repository root.

## Datasets

Audio datasets are not distributed with this repository. Download each corpus
from its original source, review its terms of use, and place the extracted audio
under `dataset/`. The included CSV metadata under `files/` uses paths relative
to the repository root.

| Dataset | Description | Source | Expected local path |
| --- | --- | --- | --- |
| UWB-ATCC | English air-traffic-control speech recorded from real ATC communications. It is used as the original speech source for similar-transcript, synthetic-pair, and hybrid training. | [Hugging Face](https://huggingface.co/datasets/Jzuluaga/uwb_atcc) | `dataset/UWB/` |
| ATCOSIM | English simulated air-traffic-control speech. It can be combined with UWB-ATCC to increase speaker and acoustic diversity. | [Hugging Face](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus) | `dataset/ATCOSIM/` |
| ATCO2 1-hour test set | A one-hour English ATC evaluation set containing 871 utterances. It is used for evaluation rather than training. | [Hugging Face](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h) | `dataset/ATCO2/audios/` |
| L2-ARCTIC | Non-native English read speech from 24 speakers across Arabic, Hindi, Korean, Mandarin, Spanish, and Vietnamese L1 groups. Shared prompts provide repeated transcripts for repeated-transcript SupCon training. | [PSI Lab](https://psi.engr.tamu.edu/l2-arctic-corpus/) | `dataset/Arctic/` |

The provided metadata is organized as follows:

```text
files/Arctic/8fold/<fold>/                 Repeated-transcript cross-validation
files/Arctic/l1_holdout/<split>/           L1 holdout evaluation
files/UWB/                                 UWB training and validation metadata
files/UWB_ATCOSIM/                         Combined UWB and ATCOSIM metadata
files/ATCO2/test.csv                       ATCO2 evaluation metadata
```

Synthetic audio used by `train_tts.py` and `train_hybrid.py` is expected under
`dataset/Synthetic/` and must match the paths listed in each `synthetic.csv`.

## License

The original source code in `scripts/` and `src/` is licensed under the
[Apache License 2.0](LICENSE).

This code license does not cover datasets, transcripts, metadata derived from
datasets, synthetic audio, or model weights. These materials remain governed by
their original licenses and terms. See [DATA_LICENSES.md](DATA_LICENSES.md) for
dataset-specific attribution and licensing notes.

## Installation

Create an environment with a CUDA-compatible PyTorch build when using a GPU,
then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Workflow

Warm up the CTC head:

```bash
python scripts/warmup.py --prefix Arctic/8fold/0 --backbone w2v2-large
```

Train with repeated transcripts:

```bash
python scripts/train_repeated.py --prefix Arctic/8fold/0 --backbone w2v2-large
```

Generate similar-transcript groups and train with them:

```bash
python scripts/prepare_similar.py --prefix UWB_ATCOSIM --coverage 0.5
python scripts/train_alternative.py --prefix UWB_ATCOSIM --similar-csv ngram_50.csv
python scripts/train_hybrid.py --prefix UWB_ATCOSIM --similar-csv ngram_50.csv
```

Train with original and synthetic transcript pairs:

```bash
python scripts/train_tts.py --prefix UWB_ATCOSIM
```

Evaluate a model:

```bash
python scripts/validate.py \
  --model-path weights/w2v2-large/Arctic/8fold/0/Supcon_Repeated \
  --processor-path weights/w2v2-large/Arctic/8fold/0/processor \
  --csv files/Arctic/8fold/0/test.csv
```

Run any script with `--help` to see its complete CLI.
