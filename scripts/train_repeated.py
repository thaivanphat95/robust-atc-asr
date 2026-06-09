import argparse
import os
import sys

import torch
from transformers import EarlyStoppingCallback

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.collators import DataCollatorCTCWithPadding
from src.configs import REPEATED_CONFIG
from src.core import BaseSupConTrainer, compute_metrics_factory
from src.data_pipeline import compute_supcon_stats, load_repeated_train_val
from src.model import W2V2SupCon
from src.runtime import build_training_args, log_gpu_info
from src.samplers import TranscriptGroupedBatchSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SupCon using repeated transcripts."
    )
    parser.add_argument(
        "--prefix",
        default="Arctic/8fold/0",
        help="Dataset split path relative to files/ and weights/<backbone>/.",
    )
    parser.add_argument("--backbone", default="w2v2-large")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=sorted(REPEATED_CONFIG),
        default=16,
    )
    parser.add_argument("--epochs", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()

    path_data = os.path.join(PROJECT_DIR, "files", args.prefix)
    path_weight = os.path.join(PROJECT_DIR, "weights", args.backbone, args.prefix)
    path_model = os.path.join(path_weight, "model")
    path_write = os.path.join(path_weight, "Supcon_Repeated")

    log_gpu_info()

    model = W2V2SupCon(path_model=path_model, proj_dim=256)
    processor = model.processor
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    train_ds, val_ds = load_repeated_train_val(path_data, processor)
    stats = compute_supcon_stats(train_ds)

    group_size, samples_per_group, grad_acc = REPEATED_CONFIG[args.batch_size]
    sampler = TranscriptGroupedBatchSampler(train_ds, batch_size=args.batch_size, group_size=group_size, samples_per_group=samples_per_group)

    steps_per_epoch = max(1, len(sampler))
    eval_steps = max(1, steps_per_epoch // 2)

    print(f"Total samples: {len(train_ds)}, val: {len(val_ds)}, batch: {args.batch_size}, epochs: {args.epochs}, steps/epoch: {steps_per_epoch}")
    print(f"Repeated positives -> known: {stats['known']}/{stats['total']} ({stats['known_ratio']:.4f}), "
          f"group_size={group_size}, samples_per_group={samples_per_group}")

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
