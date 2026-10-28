"""CPU tests for profile_attention.py's monkeypatched SDAR attention variants.

The harness's own tests drive the decoder with fakes (tests/test_blockwise.py),
which never reach `SDARAttention.forward`. These need the real one, so they
build a tiny random SDAR from a local checkpoint's own config and remote code
(models/SDAR-*-local), with the real tokenizer's vocabulary size so that
bench_verifier_rows' probe builders work unchanged.

    cd MI_decoding && CUDA_VISIBLE_DEVICES= \\
        PYTHONPATH=../harness/vendor_env/tf452:../harness:.. python -m pytest -q tests
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

MI_DIR = Path(__file__).resolve().parents[1]
REPO = MI_DIR.parent
for path in (MI_DIR, REPO / "harness"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytest.importorskip(
    "transformers.modeling_layers",
    reason="SDAR's remote code needs transformers 4.52: PYTHONPATH=../harness/vendor_env/tf452",
)
if importlib.util.find_spec("flash_attn") is not None:
    pytest.skip("flash_attn is installed, so SDAR's own forward would take its CUDA-only path; "
                "these CPU tests need it absent (as on SDS)", allow_module_level=True)
SOURCES = [REPO / "models" / name for name in ("SDAR-8B-Chat-b16-local", "SDAR-1.7B-Chat-b16-local")]
SOURCES += sorted(p for p in (REPO / "models").glob("SDAR-*-local") if p not in SOURCES)
SOURCE = next((p for p in SOURCES if (p / "modeling_sdar.py").exists() and (p / "config.json").exists()), None)
if SOURCE is None:
    pytest.skip("no models/SDAR-*-local checkpoint dir to take SDAR's code from", allow_module_level=True)

import profile_attention as pa  # noqa: E402
from bench_verifier_rows import build_prefix  # noqa: E402
from sdar_harness.model import FAMILIES, MaskedDiffusionProvider  # noqa: E402

torch.set_num_threads(min(8, torch.get_num_threads()))   # tiny shapes; a busy shared host thrashes otherwise

BLOCK = 8
VOCAB = 151936
MASK = 151669
EOS = 151643
TINY = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            intermediate_size=128, num_hidden_layers=2, vocab_size=VOCAB, max_position_embeddings=1024)


class Tok:
    mask_token_id, eos_token_id, pad_token_id, bos_token_id = MASK, EOS, EOS, None
    name_or_path = "tiny/sdar"


def tiny_model(dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(SOURCE, trust_remote_code=True, local_files_only=True)
    config.update(TINY)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype)
    return model.eval().requires_grad_(False)


def provider_for(model):
    conventions = replace(FAMILIES["sdar"], block_length=BLOCK)
    return MaskedDiffusionProvider(model, Tok(), torch.device("cpu"), conventions)


@pytest.fixture(scope="module", params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def tiny(request):
    model = tiny_model(request.param)
    provider = provider_for(model)
    generator = torch.Generator().manual_seed(1)
    # 40 tokens committed in 16-token chunks: a multi-commit prefix, like a real one.
    view = build_prefix(provider, BLOCK, 40, generator, 16)
    plan = pa.make_replay(provider, BLOCK, 40, 4, generator)
    return model, provider, view._cache, plan


def run(model, provider, cache, plan, variant, **kwargs):
    with pa.patched(model, variant, block=BLOCK, **kwargs) as store:
        return pa.replay(provider, plan, cache), store


@pytest.mark.parametrize("attention", ["sdpa_mask", "sdpa_nomask"])
@pytest.mark.parametrize("verify", [False, True])
def test_buffer_is_bitwise_cat_within_a_block_and_after_a_commit(tiny, attention, verify):
    model, provider, cache, plan = tiny
    cat, _ = run(model, provider, cache, plan, pa.Variant(attention, "cat"))
    store = pa.PrefixBuffers(verify=verify)
    buffered, _ = run(model, provider, cache, plan, pa.Variant(attention, "buffer"), buffers=store)
    assert len(cat) == len(plan.before) + len(plan.after) == 4
    for step, (a, b) in enumerate(zip(cat, buffered)):
        assert torch.equal(a, b), f"step {step} differs"
    # One prefix copy per layer per block: block n, then block n+1 after its commit.
    assert store.prefix_copies == 2 * model.config.num_hidden_layers
    # The steps really differ (a stale tail would otherwise go unnoticed).
    assert not torch.equal(cat[0], cat[2])


def test_buffer_layout_matches_the_cat():
    store = pa.PrefixBuffers(verify=True)
    base = torch.randn(1, 2, 5, 3)
    for rows in (4, 2, 4):                      # shrink and regrow within one block: no re-copy
        past = base.expand(rows, -1, -1, -1)
        new = torch.randn(rows, 2, 4, 3)
        keys, values = store.assemble(0, past, past, new, new)
        reference = torch.cat([past, new], dim=-2)
        assert torch.equal(keys, reference) and keys.stride() == reference.stride()
        assert keys.is_contiguous()
    assert store.prefix_copies == 1
    store.assemble(0, base.expand(8, -1, -1, -1), base.expand(8, -1, -1, -1),
                   torch.randn(8, 2, 4, 3), torch.randn(8, 2, 4, 3))
    assert store.prefix_copies == 2              # more rows than allocated: re-copied
    grown = torch.cat([base, torch.randn(1, 2, 4, 3)], dim=-2)
    store.assemble(0, grown, grown, torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3))
    assert store.prefix_copies == 3              # a new prefix: re-copied
    assert store.slots[0].sources[0] is grown


def test_nomask_is_close_to_mask(tiny):
    model, provider, cache, plan = tiny
    masked, _ = run(model, provider, cache, plan, pa.Variant("sdpa_mask", "cat"))
    unmasked, _ = run(model, provider, cache, plan, pa.Variant("sdpa_nomask", "cat"))
    tolerance = 1e-5 if masked[0].dtype == torch.float32 else 2e-2
    for a, b in zip(masked, unmasked):
        torch.testing.assert_close(b.float(), a.float(), atol=tolerance, rtol=tolerance)


def test_sdpa_mask_reproduces_the_original_bit_for_bit(tiny):
    model, provider, cache, plan = tiny
    assert pa.mirror_of_original(model) == pa.REFERENCE      # no flash_attn here
    original = pa.replay(provider, plan, cache)
    patched_out, _ = run(model, provider, cache, plan, pa.REFERENCE)
    assert pa.bitwise(original, patched_out)


def test_the_patch_is_undone(tiny):
    model, provider, cache, plan = tiny
    cls, module = pa.sdar_attention(model)
    forward_before = cls.__dict__["forward"]
    norm_before = module.flash_rms_norm
    before = pa.replay(provider, plan, cache)
    with pa.patched(model, pa.Variant("sdpa_nomask", "buffer"), block=BLOCK):
        assert cls.__dict__["forward"] is not forward_before
        assert cls.__dict__["forward"].profile_variant == "sdpa_nomask/buffer/torch"
        with pytest.raises(RuntimeError, match="do not nest"):
            with pa.patched(model, pa.REFERENCE, block=BLOCK):
                pass
    assert cls.__dict__["forward"] is forward_before
    assert module.flash_rms_norm is norm_before
    with pytest.raises(ValueError, match="boom"):
        with pa.patched(model, pa.Variant("sdpa_mask", "buffer"), block=BLOCK):
            raise ValueError("boom")
    assert cls.__dict__["forward"] is forward_before
    assert module.flash_rms_norm is norm_before
    assert cls not in pa._ACTIVE
    assert pa.bitwise(before, pa.replay(provider, plan, cache))


def fake_flash(calls):
    """flash_attn_func's contract ([b, s, h, d] in and out, GQA by head count) on SDPA."""
    def flash_attn_func(q, k, v, causal=False, softmax_scale=None):
        assert not causal and q.shape[-1] == k.shape[-1] and q.stride(-1) == 1
        calls.append((tuple(q.shape), tuple(k.shape)))
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale, enable_gqa=True)
        return out.transpose(1, 2)
    return flash_attn_func


def test_flash_plumbing_with_an_emulated_kernel(tiny):
    model, provider, cache, plan = tiny
    calls = []
    kernels = pa.Kernels(flash=fake_flash(calls))
    reference, _ = run(model, provider, cache, plan, pa.REFERENCE)
    synced, _ = run(model, provider, cache, plan, pa.Variant("flash_sync", "cat"), kernels=kernels, check_mask=True)
    layers = model.config.num_hidden_layers
    # 4 denoising forwards + 1 single-block commit, every layer, all on the flash path.
    assert len(calls) == 5 * layers
    unsynced, _ = run(model, provider, cache, plan, pa.Variant("flash", "cat"), kernels=kernels)
    buffered, _ = run(model, provider, cache, plan, pa.Variant("flash", "buffer"), kernels=kernels,
                      buffers=pa.PrefixBuffers(verify=True))
    assert pa.bitwise(synced, unsynced) and pa.bitwise(unsynced, buffered)
    tolerance = 1e-5 if reference[0].dtype == torch.float32 else 2e-2
    for a, b in zip(reference, synced):
        torch.testing.assert_close(b.float(), a.float(), atol=tolerance, rtol=tolerance)


def test_flash_variants_refuse_without_flash_attn(tiny):
    model, *_ = tiny
    with pytest.raises(RuntimeError, match="flash_attn"):
        with pa.patched(model, pa.Variant("flash", "cat"), block=BLOCK, kernels=pa.Kernels(flash_error="absent")):
            pass
    assert pa.Kernels(flash_error="x").missing(pa.Variant("flash_sync", "cat")) is not None
    assert pa.Kernels().missing(pa.REFERENCE) is None


def test_triton_norm_switch_is_applied_and_restored(tiny):
    model, provider, cache, plan = tiny
    _, module = pa.sdar_attention(model)
    calls = []

    def rms_norm_fn(x, weight, bias=None, eps=1e-6):
        calls.append(tuple(x.shape))
        dtype = x.dtype
        x = x.float()
        return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)

    kernels = pa.Kernels(rms_norm=rms_norm_fn)
    plain, _ = run(model, provider, cache, plan, pa.REFERENCE, kernels=kernels)
    assert not calls
    triton, _ = run(model, provider, cache, plan, pa.Variant("sdpa_mask", "cat", "triton"), kernels=kernels)
    assert calls
    assert module.flash_rms_norm is None
    assert pa.bitwise(plain, triton)        # same formula, so the switch itself changes nothing


def test_shape_rule_agrees_with_the_mask_and_catches_a_wrong_block(tiny):
    model, provider, cache, plan = tiny
    generator = torch.Generator().manual_seed(2)
    with pa.patched(model, pa.Variant("sdpa_nomask", "cat"), block=BLOCK, check_mask=True):
        build_prefix(provider, BLOCK, 48, generator, 16)        # chunked, block-causal commits
        build_prefix(provider, BLOCK, 48, generator, BLOCK)     # single-block commits
        pa.replay(provider, plan, cache)
    with pytest.raises(AssertionError, match="shape rule"):
        with pa.patched(model, pa.Variant("sdpa_nomask", "cat"), block=2 * BLOCK, check_mask=True):
            pa.replay(provider, plan, cache)


def test_end_to_end_on_cpu(tmp_path):
    """main() on a saved tiny checkpoint: every phase runs and the JSON says so."""
    checkpoint = tmp_path / f"SDAR-tiny-Chat-b{BLOCK}-local"
    checkpoint.mkdir()
    for item in SOURCE.iterdir():
        if item.suffix == ".safetensors" or item.name in ("config.json", "model.safetensors.index.json"):
            continue
        if item.is_file():
            shutil.copy(item, checkpoint / item.name)
    tiny_model().save_pretrained(checkpoint)
    payload = pa.main([
        "--model", str(checkpoint), "--device", "cpu", "--dtype", "float32",
        "--rows", "1", "2", "4", "--contexts", "16", "32", "--rounds", "2", "--reps", "2",
        "--check-cells", "2x16", "--profile-cells", "2x32", "--profile-calls", "2",
        "--prefill-chunk", "16", "--output-dir", str(tmp_path / "out"), "--tag", "smoke",
    ])
    written = json.loads((tmp_path / "out" / "smoke.json").read_text())
    assert (tmp_path / "out" / "smoke.txt").read_text().startswith("== SDAR-tiny-Chat-b8-local")
    (record,) = written["models"]
    assert record["block"] == BLOCK and record["shape"]["kv_heads"] == 2
    assert set(record["skipped_variants"]) == {v for v in pa.DEFAULT_VARIANTS if "flash" in v}
    ran = set(record["variants"])
    assert ran == {"original", "sdpa_mask/cat/torch", "sdpa_mask/buffer/torch",
                   "sdpa_nomask/cat/torch", "sdpa_nomask/buffer/torch"}
    checks = {r["variant"]: r for r in record["correctness"]}
    assert checks["original"]["bitwise_equal_mirror"] is True
    assert checks["sdpa_mask/buffer/torch"]["bitwise_equal_cat"] is True
    assert checks["sdpa_nomask/buffer/torch"]["bitwise_equal_cat"] is True
    assert checks["sdpa_mask/cat/torch"]["max_abs_logit_diff"] == 0.0
    timing = [r for r in record["timing"] if r["status"] == "ok"]
    assert len(timing) == len(ran) * 3 * 2 and all(r["n"] == 4 for r in timing)
    assert all(r["buffer_bytes"] > 0 for r in timing if "/buffer/" in r["variant"])
    assert record["fits"]["sdpa_mask/cat/torch"]["own"] is not None
    profiles = {r["variant"]: r for r in record["profile"]}
    assert profiles["sdpa_mask/cat/torch"]["prefix_regions_per_call"] == record["shape"]["layers"]
    assert profiles["sdpa_mask/cat/torch"]["sdpa_dispatch_per_call"]
    assert profiles["original"]["prefix_ms"] is None
    assert payload["models"][0]["model"] == checkpoint.name
