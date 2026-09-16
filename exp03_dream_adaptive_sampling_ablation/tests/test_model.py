from __future__ import annotations

from types import SimpleNamespace

import torch

from dream_ablation.model import DreamLogitsProvider


class Tokenizer:
    mask_token_id = 9
    eos_token_id = 8
    pad_token_id = 8
    bos_token_id = 7


class Model:
    def __init__(self) -> None:
        self.attention_mask = None

    def __call__(self, *, input_ids, attention_mask):
        self.attention_mask = attention_mask
        logits = torch.arange(
            input_ids.shape[1] * 3, dtype=torch.float32
        ).reshape(1, input_ids.shape[1], 3)
        return SimpleNamespace(logits=logits)


def test_forward_uses_dream_full_attention_sentinel_and_aligns_logits() -> None:
    model = Model()
    provider = DreamLogitsProvider(model, Tokenizer(), torch.device("cpu"))
    logits = provider.forward_logits(torch.tensor([[1, 2, 3]]))
    assert model.attention_mask == "full"
    assert logits.tolist() == [[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]
