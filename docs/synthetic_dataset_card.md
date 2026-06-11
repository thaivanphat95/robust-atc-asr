---
language:
  - en
pretty_name: Synthetic ATC Speech
task_categories:
  - automatic-speech-recognition
  - text-to-speech
tags:
  - arxiv:2605.03297
  - audio
  - air-traffic-control
  - synthetic-speech
  - robust-asr
size_categories:
  - 100K<n<1M
license: other
---

# Synthetic ATC Speech

Synthetic English air-traffic-control speech created for research on robust
automatic speech recognition. The dataset contains 276,304 generated
utterances from 15,660 unique ATC transcripts.

The dataset accompanies:

- [Contrastive Regularization for Accent-Robust ASR](https://arxiv.org/abs/2605.03297)
- [Robust ATC ASR code](https://github.com/thaivanphat95/robust-atc-asr)
- [UWB SupCon Hybrid model](https://huggingface.co/thaivanphat95/wav2vec2-robust-uwb-supcon-hybrid)
- [UWB+ATCOSIM SupCon Hybrid model](https://huggingface.co/thaivanphat95/wav2vec2-robust-uwb-atcosim-supcon-hybrid)

## Dataset Structure

The dataset provides one training split packaged as uncompressed WebDataset TAR
shards. Every sample has a paired `<key>.wav` audio file and `<key>.json`
metadata record. The JSON record contains:

| Field | Description |
| --- | --- |
| `text` | Transcript used to generate the audio |
| `tts_system` | Speech-generation system |
| `voice_source` | Source domain used for the synthetic voice |
| `transcript_source` | Original transcript corpus: `UWB` or `ATCOSIM` |
| `original_audio_filename` | Original relative path used by the research code |

The dataset is balanced across four speech-generation systems and four
voice-source domains:

| Speech-generation system | Source | Samples |
| --- | --- | ---: |
| `afro_tts` | [intronhealth/afro-tts](https://huggingface.co/intronhealth/afro-tts) | 69,076 |
| `cosyvoice_tts` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | 69,076 |
| `glm_tts` | [zai-org/GLM-TTS](https://github.com/zai-org/GLM-TTS) | 69,076 |
| `xtts_v2_tts` | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) | 69,076 |

| Voice source | Source | Samples |
| --- | --- | ---: |
| `atcosim` | [ATCOSIM](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus) | 69,076 |
| `l2arctic` | [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) | 69,076 |
| `marine` | [Marine Radio Chatter: Bridge-to-Bridge Communication](https://www.kaggle.com/datasets/linogova/marine-radio-chatter-bridge-2-bridge-communication) | 69,076 |
| `uwb` | [UWB-ATCC](https://huggingface.co/datasets/Jzuluaga/uwb_atcc) | 69,076 |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset(
    "thaivanphat95/synthetic-atc-speech",
    split="train",
    streaming=True,
)
sample = next(iter(dataset))

print(sample["wav"])
print(sample["json"]["text"])
print(sample["json"]["tts_system"])
print(sample["json"]["voice_source"])
```

The TAR shards are stored under `data/`. `shard_manifest.csv` records the
sample count and byte size of every shard.

## Intended Use

This dataset is released for non-commercial research use. It is intended for
research on:

- Robust ATC automatic speech recognition
- Synthetic-data augmentation
- Cross-domain and cross-dataset robustness
- Analysis of speech-generation diversity

It should not be used as real ATC communication or relied upon in
safety-critical operational systems. Commercial use is not authorized by this
dataset release.

## Source Data

ATC transcripts are derived from UWB-ATCC and ATCOSIM. Voice-source material is
derived from four corpora:

- UWB-ATCC: https://huggingface.co/datasets/Jzuluaga/uwb_atcc
- ATCOSIM: https://huggingface.co/datasets/Jzuluaga/atcosim_corpus
- L2-ARCTIC: https://psi.engr.tamu.edu/l2-arctic-corpus/
- Marine Radio Chatter: https://www.kaggle.com/datasets/linogova/marine-radio-chatter-bridge-2-bridge-communication

Synthetic speech was generated using:

- GLM-TTS: https://github.com/zai-org/GLM-TTS
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- Afro-TTS: https://huggingface.co/intronhealth/afro-tts
- XTTS: https://github.com/coqui-ai/TTS

Users are responsible for reviewing the terms of the source datasets,
source-voice material, and speech-generation systems.

## Limitations

- Audio is synthetic and does not reproduce every property of real ATC speech.
- Generation systems and voice sources may introduce systematic artifacts or
  biases.
- Transcripts are specialized for English ATC communication.
- The dataset may contain generation errors despite metadata integrity checks.

## Citation

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

This dataset is published with `license: other` and is intended for
non-commercial research use only. CC BY-NC 4.0 is not used because it permits
non-commercial uses beyond research and may not satisfy all source-material
terms, including applicable ShareAlike requirements.

Use and redistribution remain subject to the terms of the source transcripts,
source voices, speech-generation systems, and related third-party material.
Where those terms conflict with this research-use notice, the applicable
third-party terms control. The Apache-2.0 license of the associated code
repository does not cover this dataset.
