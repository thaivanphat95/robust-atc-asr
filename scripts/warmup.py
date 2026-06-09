import argparse
import os
import sys

from transformers import AutoProcessor, Trainer, Wav2Vec2ForCTC

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.collators import DataCollatorCTCWithPadding
from src.core import compute_metrics_factory
from src.data_pipeline import load_repeated_train_val
from src.runtime import build_warmup_args, log_gpu_info


def freeze_for_ctc_head_warmup(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Warm up a pretrained CTC head.")
    parser.add_argument(
        "--prefix",
        default="Arctic/8fold/0",
        help="Dataset split path relative to files/ and weights/<backbone>/.",
    )
    parser.add_argument("--backbone", default="w2v2-large")
    parser.add_argument(
        "--pretrained-model",
        default="facebook/wav2vec2-large-960h",
        help="Hugging Face model name or local pretrained model path.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=4,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    path_data = os.path.join(PROJECT_DIR, "files", args.prefix)
    path_weight = os.path.join(PROJECT_DIR, "weights", args.backbone, args.prefix)
    path_write = os.path.join(path_weight, "model")

    num_gpus = max(1, log_gpu_info())

    processor = AutoProcessor.from_pretrained(args.pretrained_model)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.pretrained_model,
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        vocab_size=processor.tokenizer.vocab_size,
        ignore_mismatched_sizes=True,
    )
    freeze_for_ctc_head_warmup(model)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, include_metadata=False)
    train_ds, val_ds = load_repeated_train_val(path_data, processor)

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * num_gpus))
    eval_steps = max(1, steps_per_epoch // 10)

    print(f"CTC-head warm-up -> train: {len(train_ds)}, val: {len(val_ds)}, batch: {args.batch_size}, steps: {steps_per_epoch}")
    print(f"Pretrained model: {args.pretrained_model}")
    print(f"Output: {path_write}")

    training_args = build_warmup_args(path_write, args.batch_size, steps_per_epoch, eval_steps)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_factory(processor),
    )

    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(path_write)
    processor.save_pretrained(path_write)


if __name__ == "__main__":
    main()
