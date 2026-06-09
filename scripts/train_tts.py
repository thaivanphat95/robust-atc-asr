import argparse
import os
import sys

import torch
from transformers import EarlyStoppingCallback

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.collators import DataCollatorCTCWithPadding
from src.configs import TTS_CONFIG
from src.core import BaseSupConTrainer, compute_metrics_factory
from src.data_pipeline import compute_synthetic_pairing_stats, load_synthetic_train, load_val_orig_only, suggest_synthetic_schedule
from src.model import W2V2SupCon
from src.runtime import build_training_args, log_gpu_info
from src.samplers import SyntheticAlternateBatchSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SupCon using original and synthetic transcript pairs."
    )
    parser.add_argument(
        "--prefix",
        default="UWB_ATCOSIM",
        help="Dataset split path relative to files/ and weights/<backbone>/.",
    )
    parser.add_argument("--backbone", default="w2v2-robust")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=sorted(TTS_CONFIG),
        default=16,
    )
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    path_data = os.path.join(PROJECT_DIR, "files", args.prefix)
    path_weight = os.path.join(PROJECT_DIR, "weights", args.backbone, args.prefix)
    path_model = os.path.join(path_weight, "model")
    path_write = os.path.join(path_weight, "Supcon_TTS")

    ctc_orig_only = True

    log_gpu_info()

    model = W2V2SupCon(path_model=path_model)
    processor = model.processor
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    orig_csv = os.path.join(path_data, "train.csv")
    synth_csv = os.path.join(path_data, "synthetic.csv")
    val_csv = os.path.join(path_data, "val.csv")

    train_ds = load_synthetic_train(orig_csv, synth_csv, processor)
    val_ds = load_val_orig_only(val_csv, processor)

    group_size, samples_per_group, grad_acc = TTS_CONFIG[args.batch_size]
    pair_stats = compute_synthetic_pairing_stats(train_ds)
    mixed_batches, ctc_only_batches = suggest_synthetic_schedule(pair_stats["exclusive_ratio"])

    sampler = SyntheticAlternateBatchSampler(train_ds, batch_size=args.batch_size, group_size=group_size, samples_per_group=samples_per_group,
                                             mixed_batches=mixed_batches, ctc_only_batches=ctc_only_batches)

    steps_per_epoch = max(1, len(sampler))
    eval_steps = max(1, steps_per_epoch // 2)

    print(f"Total samples: {len(train_ds)}, val: {len(val_ds)}, batch: {args.batch_size}, epochs: {args.epochs}, steps/epoch: {steps_per_epoch}")
    print(
        f"Original without synthetic pairs: {pair_stats['exclusive_orig']}/{pair_stats['total_orig']} "
        f"({pair_stats['exclusive_ratio']:.4f}), eligible transcripts: {pair_stats['eligible_transcripts']}"
    )
    print(f"Synthetic schedule -> mixed (CTC+SupCon) : ctc-only = {mixed_batches}:{ctc_only_batches}")

    training_args = build_training_args(path_write, args.epochs, args.batch_size, grad_acc, eval_steps)

    trainer = BaseSupConTrainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_ds, eval_dataset=val_ds,
                                compute_metrics=compute_metrics_factory(processor), batch_sampler=sampler, ctc_orig_only=ctc_orig_only,
                                callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(path_write)
    model.base.save_pretrained(path_write)
    torch.save(model.proj.state_dict(), os.path.join(path_write, "supcon_proj.pt"))


if __name__ == "__main__":
    main()
