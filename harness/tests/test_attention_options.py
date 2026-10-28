"""Opt-in attention options against SDAR's own (patched) modeling code, on CPU.

A tiny random SDAR is built from a local checkpoint's config and its source
`modeling_sdar.py` patched by `prepare_sdar_local.patch_modeling`, then:

- the prefix buffer gives bit-identical forwards and decodes (fixed, dynamic,
  and self-verified adaptive with multi-row verifier microbatches);
- sdpa_nomask gives the same decodes and allclose hidden states;
- with neither option the patched file decodes bit-identically to the file
  with only the original three patches (the code path is the original);
- `check_attention_hooks` refuses an option the modeling code cannot honour.

Needs SDAR's remote code (transformers 4.52):

    PYTHONPATH=vendor_env/tf452 python -m pytest -q tests/test_attention_options.py
"""
from __future__ import annotations

import importlib.util
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip(
    "transformers.modeling_layers",
    reason="SDAR's remote code needs transformers 4.52: PYTHONPATH=vendor_env/tf452",
)
if importlib.util.find_spec("flash_attn") is not None:
    pytest.skip("flash_attn is installed, so SDAR's forward would take its CUDA-only flash branch",
                allow_module_level=True)

from sdar_harness.blockwise import (  # noqa: E402
    BlockCacheProvider, BufferedCache, ExpandedCache, generate_blockwise,
)
from sdar_harness.model import (  # noqa: E402
    FAMILIES, MaskedDiffusionProvider, block_causal_mask, check_attention_hooks,
)
from sdar_harness.scheduler import GenerationConfig  # noqa: E402

HARNESS = Path(__file__).resolve().parents[1]
REPO = HARNESS.parent
LOCAL = next((p for p in sorted((REPO / "models").glob("SDAR-*-local")) if (p / "PATCHES.md").exists()), None)
if LOCAL is None:
    pytest.skip("no models/SDAR-*-local checkpoint dir to take SDAR's code from", allow_module_level=True)

spec = importlib.util.spec_from_file_location("prepare_sdar_local", HARNESS / "scripts" / "prepare_sdar_local.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

torch.set_num_threads(min(8, torch.get_num_threads()))

BLOCK = 8
VOCAB = 151936
MASK = 151669
EOS = 151643
TINY = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            intermediate_size=128, num_hidden_layers=2, vocab_size=VOCAB, max_position_embeddings=1024)


def snapshot_modeling() -> str:
    lines = (LOCAL / "PATCHES.md").read_text().splitlines()
    snapshot = Path(next(line for line in lines if line.startswith("Source snapshot: ")).split(": ", 1)[1])
    return (snapshot / "modeling_sdar.py").read_text(encoding="utf-8")


def original_three(text: str) -> str:
    """The file as it was before the 2026-10-27 hooks: only the first three patches."""
    for _, old, new in prepare.PATCHES[:3]:
        text = text.replace(old, new, 1)
    return text


def model_dir(root: Path, name: str, modeling: str) -> Path:
    target = root / name
    target.mkdir()
    for file in ("config.json", "configuration_sdar.py"):
        shutil.copyfile((LOCAL / file).resolve(), target / file)
    (target / "modeling_sdar.py").write_text(modeling, encoding="utf-8")
    return target


class Tok:
    mask_token_id, eos_token_id, pad_token_id, bos_token_id = MASK, EOS, EOS, None
    name_or_path = "tiny/sdar"


def tiny_provider(path: Path, *, kernel: str = "sdpa_mask", buffer: bool = False):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    config.update(TINY)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)
    model = model.eval().requires_grad_(False)
    check_attention_hooks(model, kernel, buffer)
    provider = MaskedDiffusionProvider(
        model, Tok(), torch.device("cpu"), replace(FAMILIES["sdar"], block_length=BLOCK)
    )
    provider.attention_kernel, provider.prefix_buffer = kernel, buffer
    return provider


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("sdar_attention_options")
    text = snapshot_modeling()
    return {
        "patched": model_dir(root, "patched", prepare.patch_modeling(text)),
        "original": model_dir(root, "original", original_three(text)),
    }


def prompt() -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    # 13 tokens: one full prompt block plus a remainder that lands in the first decode block.
    return torch.randint(0, 1000, (1, 13), generator=generator)


CONFIGS = {
    "fixed1": GenerationConfig(method="fixed", fixed_k=1, continuation_length=24, max_candidates=BLOCK),
    "dynamic": GenerationConfig(method="dynamic", threshold=0.5, continuation_length=24, max_candidates=BLOCK),
    # Self-verified adaptive: microbatch 3 makes the verifier run several multi-row calls.
    "adaptive": GenerationConfig(method="adaptive", delta=3.0, continuation_length=24, max_candidates=BLOCK,
                                 verifier_microbatch_size=3),
}


def decode(provider, config):
    result = generate_blockwise(provider, prompt(), config)
    return result.full_continuation_token_ids, result.denoising_steps, result.verifier_rows


def test_local_models_carry_every_patch():
    assert (LOCAL / "modeling_sdar.py").read_text(encoding="utf-8") == prepare.patch_modeling(snapshot_modeling()), (
        f"{LOCAL} is stale: rebuild with harness/scripts/prepare_sdar_local.py --force"
    )


@pytest.mark.parametrize("method", sorted(CONFIGS))
def test_defaults_run_the_original_code_path(dirs, method):
    patched = decode(tiny_provider(dirs["patched"]), CONFIGS[method])
    original = decode(tiny_provider(dirs["original"]), CONFIGS[method])
    assert patched == original


@pytest.mark.parametrize("method", sorted(CONFIGS))
def test_prefix_buffer_decodes_bit_identically(dirs, method):
    plain = decode(tiny_provider(dirs["patched"]), CONFIGS[method])
    buffered = decode(tiny_provider(dirs["patched"], buffer=True), CONFIGS[method])
    assert plain == buffered
    if method == "adaptive":
        assert plain[2] > 0, "the adaptive arm must exercise the multi-row verifier"


@pytest.mark.parametrize("method", sorted(CONFIGS))
def test_sdpa_nomask_decodes_the_same(dirs, method):
    masked = decode(tiny_provider(dirs["patched"]), CONFIGS[method])
    for buffer in (False, True):
        assert decode(tiny_provider(dirs["patched"], kernel="sdpa_nomask", buffer=buffer), CONFIGS[method]) == masked


def block_forward_setup(provider):
    """A two-block cached prefix, and the block after it with a few positions filled."""
    from transformers.cache_utils import DynamicCache

    total = 3 * BLOCK
    generator = torch.Generator().manual_seed(5)
    ids = torch.randint(0, 1000, (1, total), generator=generator)
    attention = block_causal_mask(total, BLOCK, device=torch.device("cpu"))
    positions = torch.arange(total)
    cache = DynamicCache()
    provider.commit(ids[:, :2 * BLOCK], attention[:2 * BLOCK, :2 * BLOCK], positions[:2 * BLOCK], cache)
    low, high = 2 * BLOCK, 3 * BLOCK
    block = ids[:, low:high].clone()
    block[0, 3:] = MASK
    return cache, attention[low:high, :high], positions[low:high], block


def rows_of(block: torch.Tensor, rows: int) -> torch.Tensor:
    probes = block.repeat(rows, 1)
    for row in range(rows):
        probes[row, 3 + row % (BLOCK - 3)] = 17 + row
    return probes


def test_buffer_forward_is_bitwise_cat_as_rows_change(dirs):
    provider = tiny_provider(dirs["patched"], buffer=True)
    cache, mask, positions, block = block_forward_setup(provider)
    plain = BlockCacheProvider(replace_attrs(provider, buffer=False), cache, mask, positions)
    buffered = BlockCacheProvider(provider, cache, mask, positions)
    assert type(buffered._kwargs(block)["past_key_values"]) is BufferedCache
    for rows in (1, 3, 2, 5, 1):   # grows twice (1 -> 3 -> 5), shrinks in between
        probes = rows_of(block, rows)
        assert torch.equal(plain.hidden_states(probes), buffered.hidden_states(probes)), rows
    assert buffered._buffer.prefix_copies == 3 * TINY["num_hidden_layers"]


def test_nomask_forward_is_close_and_flags_are_passed(dirs):
    provider = tiny_provider(dirs["patched"], kernel="sdpa_nomask")
    cache, mask, positions, block = block_forward_setup(provider)
    masked = BlockCacheProvider(replace_attrs(provider, kernel="sdpa_mask"), cache, mask, positions)
    nomask = BlockCacheProvider(provider, cache, mask, positions)
    assert nomask._kwargs(block)["full_attention"] is True
    assert "full_attention" not in masked._kwargs(block)
    assert type(masked._kwargs(block)["past_key_values"]) is ExpandedCache
    probes = rows_of(block, 4)
    torch.testing.assert_close(nomask.hidden_states(probes), masked.hidden_states(probes), rtol=1e-5, atol=1e-5)


def replace_attrs(provider, *, kernel=None, buffer=None):
    """A shallow twin of `provider` with other attention options (same model)."""
    import copy

    twin = copy.copy(provider)
    if kernel is not None:
        twin.attention_kernel = kernel
    if buffer is not None:
        twin.prefix_buffer = buffer
    return twin


def test_hooks_are_refused_on_unpatched_code(dirs):
    original = tiny_provider(dirs["original"])
    check_attention_hooks(original.model, "sdpa_mask", False)   # the default needs nothing
    with pytest.raises(RuntimeError, match="full_attention hook"):
        check_attention_hooks(original.model, "sdpa_nomask", False)
    with pytest.raises(RuntimeError, match="concat_block hook"):
        check_attention_hooks(original.model, "sdpa_mask", True)
    patched = tiny_provider(dirs["patched"])
    check_attention_hooks(patched.model, "sdpa_nomask", True)
