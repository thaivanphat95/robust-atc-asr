import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCTC, AutoProcessor


class SupConProjection(nn.Module):
    def __init__(self, in_dim, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class W2V2SupCon(nn.Module):
    def __init__(self, path_model: str, proj_dim=256):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(path_model)
        config = AutoConfig.from_pretrained(path_model)
        config.ctc_loss_reduction = "mean"
        config.pad_token_id = self.processor.tokenizer.pad_token_id
        config.vocab_size = self.processor.tokenizer.vocab_size

        self.base = AutoModelForCTC.from_pretrained(
            path_model,
            config=config,
            ignore_mismatched_sizes=True,
        )
        print(f"Loaded {self.base.config.model_type} model from {path_model}")

        self.proj = SupConProjection(in_dim=self.base.config.hidden_size, proj_dim=proj_dim)

    def forward(self, input_values, attention_mask=None, labels=None, output_hidden_states=False, **kwargs):
        return self.base(input_values=input_values, attention_mask=attention_mask, labels=labels, output_hidden_states=output_hidden_states, return_dict=True)
