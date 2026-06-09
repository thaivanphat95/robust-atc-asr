import argparse
import os
import shutil

import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare synthetic ATC audio metadata for Hugging Face."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(PROJECT_DIR, "files", "UWB_ATCOSIM", "synthetic.csv"),
        help="Source synthetic metadata CSV.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(PROJECT_DIR, "dataset", "Synthetic"),
        help="Synthetic audio directory and Hugging Face upload root.",
    )
    parser.add_argument(
        "--card",
        default=os.path.join(PROJECT_DIR, "docs", "synthetic_dataset_card.md"),
        help="Dataset card copied to <dataset-dir>/README.md.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    source = pd.read_csv(args.csv)

    required = {"audio_filename", "transcript", "TTS", "source"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{args.csv} missing required columns: {sorted(missing)}")

    metadata = source.rename(
        columns={
            "transcript": "text",
            "TTS": "tts_system",
            "source": "voice_source",
        }
    ).copy()
    metadata["file_name"] = metadata["audio_filename"].map(
        lambda path: os.path.relpath(
            os.path.join(PROJECT_DIR, path),
            dataset_dir,
        )
    )
    metadata["transcript_source"] = metadata["file_name"].map(
        lambda path: path.split(os.sep, maxsplit=1)[0]
    )
    metadata = metadata[
        ["file_name", "text", "tts_system", "voice_source", "transcript_source"]
    ]

    if metadata["file_name"].duplicated().any():
        raise ValueError("Duplicate audio paths found in synthetic metadata.")

    missing_audio = [
        path
        for path in metadata["file_name"]
        if not os.path.isfile(os.path.join(dataset_dir, path))
    ]
    if missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} audio files are missing; first: {missing_audio[0]}"
        )

    path_metadata = os.path.join(dataset_dir, "metadata.csv")
    metadata.to_csv(path_metadata, index=False)
    shutil.copyfile(args.card, os.path.join(dataset_dir, "README.md"))

    print(f"Rows: {len(metadata)}")
    print(f"Unique transcripts: {metadata['text'].nunique()}")
    print(f"Wrote: {path_metadata}")
    print(f"Wrote: {os.path.join(dataset_dir, 'README.md')}")


if __name__ == "__main__":
    main()
