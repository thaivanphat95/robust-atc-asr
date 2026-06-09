from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import AutoProcessor


def _safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True
    include_metadata: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if self.include_metadata:
            if "supcon_id" in features[0]:
                batch["supcon_id"] = torch.tensor([_safe_int(f.get("supcon_id", -1), -1) for f in features], dtype=torch.long)
            else:
                raise KeyError("supcon_id missing in features passed to collator")

            if "is_synth" in features[0]:
                batch["is_synth"] = torch.tensor([_safe_int(f.get("is_synth", 0), 0) for f in features], dtype=torch.long)

        return batch
