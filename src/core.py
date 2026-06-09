from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from evaluate import load
from torch.utils.data import DataLoader
from transformers import Trainer


@lru_cache(maxsize=1)
def get_wer_metric():
    return load("wer")


def compute_metrics_factory(processor):
    wer_metric = get_wer_metric()

    def compute_metrics(pred):
        pred_logits = pred.predictions
        if isinstance(pred_logits, (tuple, list)):
            pred_logits = pred_logits[0]
        pred_ids = np.argmax(pred_logits, axis=-1)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


def masked_mean_pool(hidden, attention_mask):
    m = attention_mask.unsqueeze(-1).float()
    denom = m.sum(dim=1).clamp(min=1.0)
    return (hidden * m).sum(dim=1) / denom


def supcon_loss(z, labels, temperature=0.1):
    device = z.device
    n = z.size(0)

    sim = (z @ z.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float().to(device)
    logits_mask = torch.ones_like(pos_mask) - torch.eye(n, device=device)
    pos_mask = pos_mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
    loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
    return loss.mean()


def get_encoder(base):
    if hasattr(base, "wav2vec2"):
        return base.wav2vec2
    if hasattr(base, "wav2vec2_conformer"):
        return base.wav2vec2_conformer
    if hasattr(base, "conformer"):
        return base.conformer
    raise AttributeError(f"Unknown backbone: {type(base)}")


def wav_attn_to_feat_lens(model, input_attention_mask, t_enc, device, batch_size=None):
    if input_attention_mask is None:
        if batch_size is None:
            raise ValueError("batch_size must be provided when input_attention_mask is None")
        return torch.full((batch_size,), t_enc, device=device, dtype=torch.long)

    wav_lens = input_attention_mask.to(device).sum(dim=1).to(torch.long)
    feat_lens = model.base._get_feat_extract_output_lengths(wav_lens).to(torch.long)
    return torch.clamp(feat_lens, min=1, max=t_enc)


def make_w2v2_frame_mask(model, input_attention_mask, t_enc, device, batch_size=None):
    feat_lens = wav_attn_to_feat_lens(model, input_attention_mask, t_enc, device, batch_size=batch_size)
    arange = torch.arange(t_enc, device=device).unsqueeze(0)
    return (arange < feat_lens.unsqueeze(1)).to(torch.long)


def ctc_loss_from_logits(logits, labels, input_lengths, blank_id, pad_value=-100):
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    target_lengths = (labels != pad_value).sum(dim=1).to(torch.long)
    targets = labels.masked_select(labels != pad_value).to(torch.long)
    return F.ctc_loss(log_probs, targets, input_lengths.to(torch.long), target_lengths, blank=blank_id, reduction="mean", zero_infinity=True)


class BaseSupConTrainer(Trainer):
    def __init__(self, supcon_lambda=0.05, supcon_temp=0.1, supcon_ramp_ratio=0.1, ctc_orig_only=False, ignore_unknown_supcon=True,
                  batch_sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_lambda = supcon_lambda
        self.supcon_temp = supcon_temp
        self.supcon_ramp_ratio = supcon_ramp_ratio
        self.ctc_orig_only = ctc_orig_only
        self.ignore_unknown_supcon = ignore_unknown_supcon
        self._batch_sampler = batch_sampler

    def get_train_dataloader(self):
        if self._batch_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(self.train_dataset, batch_sampler=self._batch_sampler, collate_fn=self.data_collator,
                           num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        sup_ids = inputs.pop("supcon_id", None)
        is_synth = inputs.pop("is_synth", None)

        enc = get_encoder(model.base)
        enc_out = enc(input_values=inputs["input_values"], attention_mask=inputs.get("attention_mask", None), return_dict=True)
        hidden = enc_out.last_hidden_state
        logits = model.base.lm_head(hidden)

        feat_lens = wav_attn_to_feat_lens(model, inputs.get("attention_mask", None), hidden.size(1), hidden.device, batch_size=hidden.size(0))
        blank_id = model.processor.tokenizer.pad_token_id

        if self.ctc_orig_only and is_synth is not None:
            ctc_mask = is_synth.to(hidden.device) == 0
            if int(ctc_mask.sum().item()) == 0:
                ctc_loss = 0.0 * logits.sum()
            else:
                ctc_loss = ctc_loss_from_logits(logits[ctc_mask], inputs["labels"][ctc_mask], feat_lens[ctc_mask], blank_id=blank_id)
        else:
            ctc_loss = ctc_loss_from_logits(logits, inputs["labels"], feat_lens, blank_id=blank_id)

        loss = ctc_loss
        sup_loss = None
        lam = 0.0
        valid_n = 0

        if sup_ids is not None:
            sup_ids = sup_ids.to(hidden.device)
            valid = (sup_ids != -1) if self.ignore_unknown_supcon else torch.ones_like(sup_ids, dtype=torch.bool)
            valid_n = int(valid.sum().item())

            if valid_n >= 2:
                enc_mask = make_w2v2_frame_mask(model, inputs.get("attention_mask", None), hidden.size(1), hidden.device, batch_size=hidden.size(0))
                pooled = masked_mean_pool(hidden, enc_mask)
                z = model.proj(pooled)
                z_v = z[valid]
                sid_v = sup_ids[valid]

                sup_loss = supcon_loss(z_v, sid_v, temperature=self.supcon_temp)
                t = self.state.global_step
                t_max = max(1, int(self.state.max_steps * self.supcon_ramp_ratio))
                lam = self.supcon_lambda * min(1.0, t / t_max)
                loss = loss + lam * sup_loss

        log_every = self.args.logging_steps
        if model.training and self.state.global_step > 0 and (self.state.global_step % log_every == 0):
            self.log({"ctc_loss": float(ctc_loss.detach().cpu()), "supcon_valid_n": float(valid_n), "supcon_lam": float(lam),
                       "supcon_term": 0.0 if sup_loss is None else float((lam * sup_loss).detach().cpu())})

        if return_outputs:
            return loss, {"logits": logits}
        return loss
