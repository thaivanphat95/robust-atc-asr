import torch
from transformers import TrainingArguments


def log_gpu_info():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    return num_gpus


def build_training_args(output_dir, epochs, batch_size, grad_acc_steps, eval_steps):
    return TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        max_grad_norm=1, bf16=True, remove_unused_columns=False, gradient_accumulation_steps=grad_acc_steps,
        learning_rate=1e-5, save_steps=eval_steps, eval_steps=eval_steps, eval_strategy="steps", logging_steps=eval_steps,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False, save_total_limit=1)


def build_warmup_args(output_dir, batch_size, steps_per_epoch, eval_steps):
    return TrainingArguments(output_dir=output_dir, max_steps=steps_per_epoch, per_device_train_batch_size=batch_size,
        max_grad_norm=0.5, bf16=False, fp16=False, remove_unused_columns=True, gradient_accumulation_steps=1,
        warmup_steps=steps_per_epoch, learning_rate=3e-6, save_steps=eval_steps, eval_steps=eval_steps,
        eval_strategy="steps", logging_steps=eval_steps, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False, save_total_limit=1)
