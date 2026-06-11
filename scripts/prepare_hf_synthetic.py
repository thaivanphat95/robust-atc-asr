import argparse
import csv
import io
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
GIB = 1024**3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package synthetic ATC speech as Hugging Face WebDataset shards."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_DIR / "files" / "UWB_ATCOSIM" / "synthetic.csv",
        help="Source synthetic metadata CSV.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=PROJECT_DIR / "dataset" / "Synthetic",
        help="Root directory containing the synthetic WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "dataset" / "Synthetic_HF",
        help="Directory containing the dataset card, manifest, and TAR shards.",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=PROJECT_DIR / "docs" / "synthetic_dataset_card.md",
        help="Dataset card copied to <output-dir>/README.md.",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=1.0,
        help="Approximate maximum size of each uncompressed TAR shard in GiB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing TAR shards in the output directory.",
    )
    return parser.parse_args()


def tar_member_size(size):
    return 512 + ((size + 511) // 512) * 512


def add_bytes(tar, name, content):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(content))


def add_file(tar, name, path):
    info = tar.gettarinfo(str(path), arcname=name)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with path.open("rb") as audio:
        tar.addfile(info, audio)


def resolve_audio_path(audio_filename):
    path = Path(audio_filename)
    return path if path.is_absolute() else PROJECT_DIR / path


def validate_source(source, csv_path):
    required = {"audio_filename", "transcript", "TTS", "source"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    if source["audio_filename"].duplicated().any():
        raise ValueError("Duplicate audio paths found in synthetic metadata.")

    for audio_filename in source["audio_filename"]:
        path = resolve_audio_path(audio_filename)
        if not path.is_file():
            raise FileNotFoundError(f"Missing audio file: {path}")


def open_shard(data_dir, shard_index):
    path = data_dir / f"train-{shard_index:05d}.tar"
    return path, tarfile.open(path, mode="w")


def main():
    args = parse_args()
    if args.shard_size_gb <= 0:
        raise ValueError("--shard-size-gb must be greater than zero.")

    csv_path = args.csv.resolve()
    audio_root = args.audio_root.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    source = pd.read_csv(csv_path)
    validate_source(source, csv_path)

    if data_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{data_dir} already exists. Use --overwrite to replace its shards."
            )
        shutil.rmtree(data_dir)

    data_dir.mkdir(parents=True)
    shutil.copyfile(args.card, output_dir / "README.md")

    target_bytes = int(args.shard_size_gb * GIB)
    manifest = []
    shard_index = 0
    shard_samples = 0
    shard_estimated_bytes = 1024
    shard_path, shard = open_shard(data_dir, shard_index)

    for row_index, row in source.iterrows():
        audio_path = resolve_audio_path(row["audio_filename"]).resolve()
        try:
            transcript_source = audio_path.relative_to(audio_root).parts[0]
        except ValueError as error:
            raise ValueError(f"{audio_path} is outside --audio-root {audio_root}") from error

        key = f"{row_index:09d}"
        metadata = {
            "text": row["transcript"],
            "tts_system": row["TTS"],
            "voice_source": row["source"],
            "transcript_source": transcript_source,
            "original_audio_filename": row["audio_filename"],
        }
        json_bytes = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        sample_bytes = tar_member_size(audio_path.stat().st_size) + tar_member_size(
            len(json_bytes)
        )

        if shard_samples and shard_estimated_bytes + sample_bytes > target_bytes:
            shard.close()
            manifest.append(
                {
                    "shard": f"data/{shard_path.name}",
                    "samples": shard_samples,
                    "bytes": shard_path.stat().st_size,
                }
            )
            print(
                f"Wrote {shard_path.name}: {shard_samples:,} samples, "
                f"{shard_path.stat().st_size / GIB:.2f} GiB"
            )
            shard_index += 1
            shard_samples = 0
            shard_estimated_bytes = 1024
            shard_path, shard = open_shard(data_dir, shard_index)

        add_file(shard, f"{key}.wav", audio_path)
        add_bytes(shard, f"{key}.json", json_bytes)
        shard_samples += 1
        shard_estimated_bytes += sample_bytes

    shard.close()
    if shard_samples:
        manifest.append(
            {
                "shard": f"data/{shard_path.name}",
                "samples": shard_samples,
                "bytes": shard_path.stat().st_size,
            }
        )
        print(
            f"Wrote {shard_path.name}: {shard_samples:,} samples, "
            f"{shard_path.stat().st_size / GIB:.2f} GiB"
        )

    manifest_path = output_dir / "shard_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["shard", "samples", "bytes"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Rows: {len(source):,}")
    print(f"Unique transcripts: {source['transcript'].nunique():,}")
    print(f"Shards: {len(manifest):,}")
    print(f"Wrote: {manifest_path}")
    print(f"Upload root: {output_dir}")


if __name__ == "__main__":
    main()
