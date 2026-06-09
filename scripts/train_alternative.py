import argparse
import os
import sys

import torch
from transformers import EarlyStoppingCallback

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.collators import DataCollatorCTCWithPadding
from src.configs import ALTERNATIVE_CONFIG
from src.core import BaseSupConTrainer, compute_metrics_factory
from src.data_pipeline import compute_supcon_stats, load_similar_train_val, suggest_ctc_batches_per_supcon
from src.model import W2V2SupCon
from src.runtime import build_training_args, log_gpu_info
from src.samplers import TranscriptAlternateBatchSampler


def csv_filename(value):
    if os.path.basename(value) != value or not value.endswith(".csv"):
        raise argparse.ArgumentTypeError("value must be a CSV filename from generated/")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SupCon using alternating similar and CTC-only batches."
    )
    parser.add_argument(
        "--prefix",
        default="UWB_ATCOSIM",
        help="Dataset split path relative to files/ and weights/<backbone>/.",
    )
    parser.add_argument("--backbone", default="w2v2-robust")
    parser.add_argument(
        "--similar-csv",
        type=csv_filename,
        default="ngram_50.csv",
        help="Similar-group CSV filename inside the dataset's generated/ folder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=sorted(ALTERNATIVE_CONFIG),
        default=16,
    )
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    similar_tag = os.path.splitext(args.similar_csv)[0].removeprefix("ngram_")

    path_data = os.path.join(PROJECT_DIR, "files", args.prefix)
    path_weight = os.path.join(PROJECT_DIR, "weights", args.backbone, args.prefix)
    path_model = os.path.join(path_weight, "model")
    path_write = os.path.join(path_weight, f"Supcon_{similar_tag}")

    train_csv = os.path.join(path_data, "generated", args.similar_csv)
    val_csv = os.path.join(path_data, "val.csv")
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Similar-group CSV not found: {train_csv}")

    log_gpu_info()

    model = W2V2SupCon(path_model=path_model)
    processor = model.processor
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    train_ds, val_ds = load_similar_train_val(train_csv, val_csv, processor)
    stats = compute_supcon_stats(train_ds)

    group_size, samples_per_group, grad_acc = ALTERNATIVE_CONFIG[args.batch_size]
    ctc_batches_per_supcon = suggest_ctc_batches_per_supcon(stats["known_ratio"], max_ctc_steps=4)

    sampler = TranscriptAlternateBatchSampler(train_ds, batch_size=args.batch_size, group_size=group_size,
                                            samples_per_group=samples_per_group, ctc_batches_per_supcon=ctc_batches_per_supcon)

    steps_per_epoch = max(1, len(sampler))
    eval_steps = max(1, steps_per_epoch // 2)

    print(f"Total samples: {len(train_ds)}, val: {len(val_ds)}, batch: {args.batch_size}, epochs: {args.epochs}, steps/epoch: {steps_per_epoch}")
    print(f"SupCon stats -> known: {stats['known']}/{stats['total']} ({stats['known_ratio']:.4f}), "
        f"unknown: {stats['unknown']}/{stats['total']} ({stats['unknown_ratio']:.4f})")
    print(f"Alternative schedule -> 1 SupCon batch then {ctc_batches_per_supcon} CTC-only batch(es), group_size={group_size}, samples_per_group={samples_per_group}")

    training_args = build_training_args(path_write, args.epochs, args.batch_size, grad_acc, eval_steps)

    trainer = BaseSupConTrainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_ds, eval_dataset=val_ds,
                                compute_metrics=compute_metrics_factory(processor), batch_sampler=sampler,
                                callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(path_write)
    model.base.save_pretrained(path_write)
    torch.save(model.proj.state_dict(), os.path.join(path_write, "supcon_proj.pt"))


if __name__ == "__main__":
    main()
