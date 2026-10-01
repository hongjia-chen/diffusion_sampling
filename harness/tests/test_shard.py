from __future__ import annotations

import pytest
import torch
from torch import nn

from sdar_harness.model import ModelOptions, layer_split, shard_decoder


def test_layer_split_is_even_with_extras_first() -> None:
    assert layer_split(36, 2) == [18, 18]
    assert layer_split(28, 3) == [10, 9, 9]
    with pytest.raises(ValueError):
        layer_split(2, 3)


def test_devices_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ModelOptions(model_id="x", devices=("cuda:0", "cuda:0"))
    assert ModelOptions(model_id="x", device="cuda:3").device_list == ["cuda:3"]


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None,
                past_key_value=None):
        assert attention_mask.device == hidden_states.device
        cos, sin = position_embeddings
        assert cos.device == hidden_states.device
        return (self.linear(hidden_states) + cos + sin,)


class Backbone(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.rotary_emb = nn.Identity()
        self.layers = nn.ModuleList(Layer() for _ in range(layers))
        self.norm = nn.LayerNorm(4)

    def forward(self, input_ids, attention_mask, cache=None):
        hidden = self.embed_tokens(input_ids)
        cos = torch.ones_like(hidden)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask,
                           position_embeddings=(cos, cos), past_key_value=cache)[0]
        return self.norm(hidden)


class LM(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.model = Backbone(layers)
        self.lm_head = nn.Linear(4, 10)


def test_sharded_forward_matches_unsharded_on_cpu() -> None:
    torch.manual_seed(0)
    reference = LM(4)
    sharded = LM(4)
    sharded.load_state_dict(reference.state_dict())
    cpu = torch.device("cpu")
    assert shard_decoder(sharded, [cpu, cpu]) == [2, 2]
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    expected = reference.lm_head(reference.model(ids, mask))
    got = sharded.lm_head(sharded.model(ids, mask))
    assert torch.equal(expected, got)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_sharded_forward_places_stages_on_their_cards() -> None:
    torch.manual_seed(0)
    reference = LM(4)
    sharded = LM(4)
    sharded.load_state_dict(reference.state_dict())
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    shard_decoder(sharded, devices)
    assert sharded.model.layers[1].linear.weight.device == devices[0]
    assert sharded.model.layers[2].linear.weight.device == devices[1]
    ids = torch.tensor([[1, 2, 3]], device=devices[1])
    mask = torch.ones(1, 3, 3, dtype=torch.bool, device=devices[1])
    got = sharded.lm_head(sharded.model(ids, mask))
    assert got.device == devices[1]
    expected = reference.lm_head(reference.model(ids.cpu(), mask.cpu()))
    torch.testing.assert_close(got.cpu(), expected)
