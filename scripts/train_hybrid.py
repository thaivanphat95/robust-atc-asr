import argparse
import os
import sys

import torch
from transformers import EarlyStoppingCallback

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.collators import DataCollatorCTCWithPadding
from src.configs import HYBRID_CONFIG
from src.core import BaseSupConTrainer, compute_metrics_factory
from src.data_pipeline import load_hybrid_train, load_val_orig_only, suggest_three_mode_schedule
from src.model import W2V2SupCon
from src.runtime import build_training_args, log_gpu_info
from src.samplers import HybridAlternateBatchSampler


def csv_filename(value):
    if os.path.basename(value) != value or not value.endswith(".csv"):
        raise argparse.ArgumentTypeError("value must be a CSV filename from generated/")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SupCon using similar, synthetic, and CTC-only batches."
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
        choices=sorted(HYBRID_CONFIG),
        default=16,
    )
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    path_data = os.path.join(PROJECT_DIR, "files", args.prefix)
    path_weight = os.path.join(PROJECT_DIR, "weights", args.backbone, args.prefix)
    path_model = os.path.join(path_weight, "model")
    path_write = os.path.join(path_weight, "Supcon_Hybrid")

    ctc_orig_only = True

    log_gpu_info()

    model = W2V2SupCon(path_model=path_model)
    processor = model.processor
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    orig_csv = os.path.join(path_data, "generated", args.similar_csv)
    synth_csv = os.path.join(path_data, "synthetic.csv")
    val_csv = os.path.join(path_data, "val.csv")
    if not os.path.isfile(orig_csv):
        raise FileNotFoundError(f"Similar-group CSV not found: {orig_csv}")

    group_size, samples_per_group, grad_acc = HYBRID_CONFIG[args.batch_size]

    train_ds, mode_stats = load_hybrid_train(orig_csv, synth_csv, processor)
    val_ds = load_val_orig_only(val_csv, processor)
    mode_schedule = suggest_three_mode_schedule(mode_stats, cycle_base=6)

    sampler = HybridAlternateBatchSampler(train_ds, batch_size=args.batch_size, group_size=group_size, samples_per_group=samples_per_group, mode_schedule=mode_schedule)

    steps_per_epoch = max(1, len(sampler))
    eval_steps = max(1, steps_per_epoch // 2)

    total_orig = max(1, mode_stats["total_orig"])
    sim_ratio = mode_stats["sim_orig"] / total_orig
    synthetic_ratio = mode_stats["tts_orig"] / total_orig
    ctc_ratio = mode_stats["ctc_orig"] / total_orig

    print(f"Total train rows: {len(train_ds)}, val rows: {len(val_ds)}, batch: {args.batch_size}, epochs: {args.epochs}, steps/epoch: {steps_per_epoch}")
    print("Mode ratio from original CSV -> "
    f"sim: {mode_stats['sim_orig']}/{mode_stats['total_orig']} ({sim_ratio:.4f}), "
    f"synthetic: {mode_stats['tts_orig']}/{mode_stats['total_orig']} ({synthetic_ratio:.4f}), "
    f"ctc-only: {mode_stats['ctc_orig']}/{mode_stats['total_orig']} ({ctc_ratio:.4f})")
    print(f"Synthetic support -> eligible transcripts: {mode_stats['eligible_tts_transcripts']}, kept synthetic rows: {mode_stats['kept_synth']}")
    print("Alternating cycle -> " + ", ".join([f"{k}:{v}" for k, v in mode_schedule.items()]) + f" (cycle_len={sum(mode_schedule.values())})")

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
