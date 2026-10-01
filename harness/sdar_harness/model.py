## Loads a masked diffusion LM and its tokenizer
## Constructs prompts, runs forward passes, and aligns logits with token positions

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch


# Deliberately unset. The previous experiment hardcoded
# "Dream-org/Dream-v0-Instruct-7B"; this one is meant to run a smaller
# checkpoint that has not been chosen yet, and a stale default is the kind of
# thing that silently produces a run nobody wanted. `--model` is required.
DEFAULT_MODEL: str | None = None


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


@dataclass(frozen=True)
class Conventions:
    """Per-checkpoint calling conventions the sampler has to get right.

    These are not preferences. Getting either one wrong yields logits that are
    off by one position or an attention pattern that is causal when it should
    be bidirectional, and the sampler will still run and still produce text.

    ``logits_alignment``
        "shift" reproduces Dream's lm-eval wrapper, which shifts model output
        one position right so that ``logits[:, p]`` predicts ``input_ids[:, p]``.
        "none" is for checkpoints whose head already predicts the token at its
        own position, which is the usual masked-diffusion convention.

    ``attention_mask``
        Dream's remote code takes the string sentinel "full" to mean
        bidirectional attention. Other checkpoints expect either a real tensor
        or nothing at all. "omit" passes no attention_mask argument.
    """

    logits_alignment: Literal["shift", "none"]
    attention_mask: Literal["full", "ones", "omit", "block_causal"]
    # Which auto class loads the checkpoint. SDAR's `AutoModel` resolves to
    # `SDARModel`, the backbone, which has no `lm_head`.
    loader: Literal["auto_model", "auto_model_for_causal_lm"] = "auto_model"
    # Block diffusion only: attention is causal ACROSS blocks and bidirectional
    # WITHIN one. None means a full-canvas dLLM with no block structure.
    block_length: int | None = None


# Verified against Dream's `generation_utils.py` and its lm-eval wrapper in the
# previous experiment (exp03_dream_adaptive_sampling_ablation).
DREAM = Conventions(logits_alignment="shift", attention_mask="full")

# Verified against SDAR's own reference inference, `generate.py` in
# JetAstra/SDAR: block lower-triangular attention, position-aligned logits (it
# samples from `model(...).logits` and writes back at the same indices, with no
# shift), and `AutoModelForCausalLM`. `block_length` is set per checkpoint --
# the released variants are b4 (default), b8, b16, b32 and b64.
SDAR = Conventions(
    logits_alignment="none",
    attention_mask="block_causal",
    loader="auto_model_for_causal_lm",
    block_length=64,
)

# Starting point for a checkpoint whose head is already position-aligned and
# which wants an ordinary all-ones mask. NOT yet verified against any specific
# model -- whichever checkpoint this experiment settles on has to be checked
# against its own reference generation code before a real run. `verify_lean_path`
# will not catch an alignment error: both read paths would be wrong together.
GENERIC_MASKED_DIFFUSION = Conventions(logits_alignment="none", attention_mask="ones")

FAMILIES: dict[str, Conventions] = {
    "dream": DREAM,
    "sdar": SDAR,
    "generic": GENERIC_MASKED_DIFFUSION,
}

# JetLM/SDAR-<size>-Chat is the block-4 default; -b8 ... -b64 are the other
# released block sizes. Local copies (prepare_sdar_local.py) append "-local".
_SDAR_CHAT_NAME = re.compile(r"SDAR-.+-Chat(?:-b(\d+))?(?:-local)?")


def trained_block_length(model_id: str) -> int | None:
    """Block length an SDAR chat checkpoint was trained with, read from its name.

    The released config.json files are identical across block sizes, so the
    name is the only record of it. None when the name does not follow the
    JetLM convention (Sci variants, derived checkpoints): then nothing is checked.
    """
    match = _SDAR_CHAT_NAME.fullmatch(model_id.rstrip("/").split("/")[-1])
    if match is None:
        return None
    return int(match.group(1)) if match.group(1) else 4


def declared_stop_token_ids(model: Any, tokenizer: Any) -> tuple[int, ...]:
    """The checkpoint's own stop set: generation_config EOS plus tokenizer EOS.

    SDAR-1.7B-Chat declares [151645 <|im_end|>, 151643 <|endoftext|>] and its
    reference `generate.py` stops on any of them; Dream declares only 151643.
    """
    declared: list[int] = []
    config = getattr(model, "generation_config", None)
    value = getattr(config, "eos_token_id", None) if config is not None else None
    if isinstance(value, int):
        declared.append(value)
    elif value is not None:
        declared.extend(int(item) for item in value)
    if tokenizer.eos_token_id is not None:
        declared.append(int(tokenizer.eos_token_id))
    return tuple(sorted(set(declared)))


def assert_shared_vocabulary(first: Any, second: Any, limit: int) -> None:
    """Refuse to pair two tokenizers that disagree on any ID below `limit`.

    The cross-model sampler moves token IDs between SDAR and Dream unchanged,
    which is only sound if every ID either model can commit decodes to the
    same token in both (DESIGN.md F4). Checked once, at load.
    """
    ids = list(range(limit))
    left = first.convert_ids_to_tokens(ids)
    right = second.convert_ids_to_tokens(ids)
    mismatched = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    if mismatched:
        sample = ", ".join(f"{i}: {left[i]!r} vs {right[i]!r}" for i in mismatched[:5])
        raise ValueError(
            f"{len(mismatched)} token IDs below {limit} differ between the "
            f"proposer and the verifier ({sample})"
        )


def block_causal_mask(
    total_length: int,
    block_length: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Block lower-triangular attention, [total_length, total_length] bool.

    Position i attends to position j exactly when j's block is at or before
    i's, so attention is bidirectional within a block and causal across them.
    Transcribed from SDAR's `generate.py`:

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks))
        attn = block_mask.repeat_interleave(block_length, 0) \
                         .repeat_interleave(block_length, 1)

    `total_length` must be a whole number of blocks; the reference rounds the
    canvas up to one (`total_length = num_blocks * block_length`) rather than
    truncating, and the prompt is NOT special-cased -- it simply occupies the
    leading blocks under the same rule.
    """
    if block_length < 1:
        raise ValueError("block_length must be positive")
    if total_length % block_length:
        raise ValueError(
            f"total_length {total_length} is not a whole number of "
            f"{block_length}-token blocks"
        )
    blocks = total_length // block_length
    mask = torch.tril(torch.ones(blocks, blocks, dtype=torch.bool, device=device))
    return mask.repeat_interleave(block_length, dim=0).repeat_interleave(
        block_length, dim=1
    )


def layer_split(num_layers: int, stages: int) -> list[int]:
    """Decoder layers per pipeline stage, as even as possible, extras first.

    The embedding sits on the first stage and the untied LM head on the last;
    for SDAR/Qwen3 they are the same [vocab, hidden] size, so an even layer
    split is also an even weight split.
    """
    if stages < 1 or stages > num_layers:
        raise ValueError(f"cannot split {num_layers} layers over {stages} stages")
    base, extra = divmod(num_layers, stages)
    return [base + (1 if stage < extra else 0) for stage in range(stages)]


def _move_args_hook(device: torch.device):
    """Forward pre-hook moving a module's positional tensor arguments to `device`."""

    def hook(module: Any, args: tuple) -> tuple:
        return tuple(
            value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for value in args
        )

    return hook


class _StageInputs:
    """Places each decoder layer's inputs on that layer's card.

    The backbone hands every layer the SAME attention mask, position ids and
    rotary tables, created wherever the caller made them. Moving them in each
    layer's hook would copy them once per layer; instead each (tensor, card)
    pair is copied once per forward and reused. The memo is cleared by a
    pre-hook on the backbone, so nothing outlives the forward that made it.

    `hidden_states` really does change per layer and is simply moved; the move
    is a no-op except at a stage boundary. The cache object is never touched:
    each layer's K/V lives, and stays, on that layer's card.
    """

    SHARED_KWARGS = ("attention_mask", "position_ids", "cache_position", "position_embeddings")

    def __init__(self) -> None:
        self._memo: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self, module: Any, args: tuple, kwargs: dict) -> None:
        self._memo.clear()

    def _place(self, value: Any, device: torch.device) -> Any:
        if isinstance(value, tuple):
            return tuple(self._place(item, device) for item in value)
        if not isinstance(value, torch.Tensor) or value.device == device:
            return value
        key = (id(value), device)
        hit = self._memo.get(key)
        if hit is None or hit[0] is not value:
            # Keep the source alive alongside its copy so its id cannot be
            # recycled by an unrelated tensor within the same forward.
            hit = (value, value.to(device, non_blocking=True))
            self._memo[key] = hit
        return hit[1]

    def hook(self, device: torch.device):
        def pre_hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            if args and isinstance(args[0], torch.Tensor):
                args = (args[0].to(device, non_blocking=True),) + tuple(args[1:])
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = kwargs["hidden_states"].to(device, non_blocking=True)
            for name in self.SHARED_KWARGS:
                if name in kwargs:
                    kwargs[name] = self._place(kwargs[name], device)
            return args, kwargs

        return pre_hook


def shard_decoder(model: Any, devices: list[torch.device]) -> list[int]:
    """Pipeline-split a Qwen-style causal LM over `devices`, in place.

    embed_tokens + rotary on the first card, contiguous runs of decoder layers
    on each card, final norm + lm_head on the last. Hidden states cross a card
    boundary once per boundary per forward. This is plain model parallelism:
    the stages run one after another, so it buys memory, not speed, and
    computes the same function as the single-card model.
    """
    backbone = model.model
    layers = backbone.layers
    split = layer_split(len(layers), len(devices))
    first, last = devices[0], devices[-1]
    backbone.embed_tokens.to(first)
    backbone.embed_tokens.register_forward_pre_hook(_move_args_hook(first))
    backbone.rotary_emb.to(first)
    # SDAR's rotary moves inv_freq to the activations' card but not the
    # position ids, which arrive from the sampler on the last card.
    backbone.rotary_emb.register_forward_pre_hook(_move_args_hook(first))
    inputs = _StageInputs()
    backbone.register_forward_pre_hook(inputs.reset, with_kwargs=True)
    index = 0
    for device, count in zip(devices, split):
        for _ in range(count):
            layers[index].to(device)
            layers[index].register_forward_pre_hook(inputs.hook(device), with_kwargs=True)
            index += 1
    backbone.norm.to(last)
    model.lm_head.to(last)
    # `project` feeds lm_head hidden states gathered on the last card already;
    # this only matters if a caller hands it something from elsewhere.
    model.lm_head.register_forward_pre_hook(_move_args_hook(last))
    return split


@dataclass(frozen=True)
class ModelOptions:
    model_id: str
    family: str = "dream"
    revision: str | None = None
    device: str = "cuda:0"
    # More than one entry pipeline-splits the model over these cards
    # (`shard_decoder`). `device` is then ignored: sampler state lives on the
    # LAST card, where the hidden states and logits come out.
    devices: tuple[str, ...] | None = None
    dtype: str = "bfloat16"
    cache_dir: str | None = None
    local_files_only: bool = False
    # At verifier microbatch sizes above ~2 the full [B, S, V] fallback is not a
    # degraded path, it is an immediate OOM. Fail loudly instead.
    require_lean_logits: bool = False
    # Overrides the family default. The SDAR checkpoints ship one block length
    # each (-b8 ... -b64), so this must match the checkpoint or the attention
    # pattern will not be the one it was trained with.
    block_length: int | None = None
    # Read logits only over token IDs [0, vocab_limit) and renormalize over
    # that range (DESIGN.md F4). SDAR and Dream agree on 0..151664 and differ
    # above it -- SDAR's 151666 is Dream's mask token -- so both run at 151665.
    # None keeps the checkpoint's whole vocabulary.
    vocab_limit: int | None = None
    # Tokens that end the answer: truncate at the first one, and early-stop on
    # it (DESIGN.md F2). None takes the checkpoint's own declaration,
    # `generation_config.eos_token_id`, plus the tokenizer's EOS.
    stop_token_ids: tuple[int, ...] | None = None
    # Decode an SDAR checkpoint at a block length other than the one it was
    # trained with. Out of distribution; for robustness ablations only.
    allow_block_mismatch: bool = False

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required; there is no default checkpoint")
        if self.family not in FAMILIES:
            raise ValueError(
                f"unknown model family {self.family!r}; known: {sorted(FAMILIES)}"
            )
        if self.block_length is not None and self.block_length < 1:
            raise ValueError("block_length must be positive")
        trained = trained_block_length(self.model_id) if self.family == "sdar" else None
        decoded = self.conventions.block_length
        if trained is not None and decoded != trained and not self.allow_block_mismatch:
            raise ValueError(
                f"{self.model_id} was trained with block length {trained} but would "
                f"decode with {decoded}; pass --block-length {trained}, or "
                "--allow-block-mismatch for an out-of-distribution ablation"
            )
        if self.vocab_limit is not None and self.vocab_limit < 1:
            raise ValueError("vocab_limit must be positive")
        if self.devices is not None and len(set(self.devices)) != len(self.devices):
            raise ValueError(f"duplicate entries in devices {self.devices}")

    @property
    def device_list(self) -> list[str]:
        return list(self.devices) if self.devices else [self.device]

    @property
    def conventions(self) -> Conventions:
        base = FAMILIES[self.family]
        if self.block_length is None:
            return base
        return replace(base, block_length=self.block_length)


class MaskedDiffusionProvider:
    """Thin wrapper around a remote-code masked diffusion checkpoint.

    Two read paths expose the same numbers:

    ``forward_logits``  materializes the whole [B, S, V] logits tensor. Simple,
    but at a 150k vocabulary a single 8-row verifier microbatch costs well over
    a gigabyte, which does not fit alongside 7B bfloat16 weights on a 16 GB card.

    ``hidden_states`` + ``logits_from_hidden`` keep only the [B, S, H] backbone
    output and apply the vocabulary projection to the handful of positions the
    scheduler actually reads. Prefer this path.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        conventions: Conventions = DREAM,
        *,
        vocab_limit: int | None = None,
        stop_token_ids: Any = None,
        devices: list[torch.device] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        # `device` is where inputs are placed and outputs come back: the only
        # card the sampler's own tensors ever touch. `devices` is every card
        # the weights occupy, which is what timing must synchronize.
        self.device = device
        self.devices = list(devices) if devices else [device]
        self.layer_split: list[int] | None = None
        self.conventions = conventions
        self.vocab_limit = vocab_limit
        required = {
            "mask_token_id": tokenizer.mask_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Tokenizer is missing required IDs: {missing}")
        self.mask_token_id = int(tokenizer.mask_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.pad_token_id = (
            int(tokenizer.pad_token_id)
            if tokenizer.pad_token_id is not None
            else self.eos_token_id
        )
        self.bos_token_id = (
            int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None
        )
        self.stop_token_ids = frozenset(
            int(value) for value in (stop_token_ids or (self.eos_token_id,))
        )
        if vocab_limit is not None:
            outside = sorted(value for value in self.stop_token_ids if value >= vocab_limit)
            if outside:
                raise ValueError(
                    f"stop tokens {outside} lie outside vocab_limit {vocab_limit}; "
                    "they could never be committed"
                )
        self._backbone = getattr(model, "model", None)
        self._lm_head = getattr(model, "lm_head", None)

    @classmethod
    def from_pretrained(cls, options: ModelOptions) -> "MaskedDiffusionProvider":
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        common = {
            "revision": options.revision,
            "cache_dir": options.cache_dir,
            "local_files_only": options.local_files_only,
            "trust_remote_code": True,
        }
        common = {key: value for key, value in common.items() if value is not None}
        tokenizer = AutoTokenizer.from_pretrained(options.model_id, **common)
        auto_class = (
            AutoModelForCausalLM
            if options.conventions.loader == "auto_model_for_causal_lm"
            else AutoModel
        )
        model = auto_class.from_pretrained(
            options.model_id,
            torch_dtype=resolve_dtype(options.dtype),
            low_cpu_mem_usage=True,
            **common,
        )
        devices = [torch.device(name) for name in options.device_list]
        split = None
        if len(devices) > 1:
            if getattr(model, "model", None) is None or getattr(model, "lm_head", None) is None:
                raise ValueError("multi-device loading needs a .model/.lm_head checkpoint")
            split = shard_decoder(model, devices)
            device = devices[-1]
        else:
            device = devices[0]
            model = model.to(device)
        model = model.eval().requires_grad_(False)
        provider = cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            devices=devices,
            conventions=options.conventions,
            vocab_limit=options.vocab_limit,
            stop_token_ids=(
                options.stop_token_ids
                if options.stop_token_ids is not None
                else declared_stop_token_ids(model, tokenizer)
            ),
        )
        provider.layer_split = split
        if not provider.verify_lean_path() and options.require_lean_logits:
            raise RuntimeError(
                "The lean logits path is unavailable, but require_lean_logits is "
                "set. The full [B, S, V] fallback cannot fit a verifier "
                "microbatch above ~2; refusing to continue rather than OOM "
                "partway through a run."
            )
        return provider

    # ------------------------------------------------------------------
    # Calling conventions
    # ------------------------------------------------------------------

    def _attention_kwargs(self, input_ids: torch.Tensor) -> dict[str, Any]:
        mode = self.conventions.attention_mask
        if mode == "omit":
            return {}
        if mode == "full":
            return {"attention_mask": "full"}
        if mode == "block_causal":
            block = self.conventions.block_length
            if block is None:
                raise ValueError("block_causal attention requires a block_length")
            length = int(input_ids.shape[1])
            if length % block:
                raise ValueError(
                    f"block_causal attention needs a whole number of blocks, "
                    f"got {length} tokens at block_length {block}. The canvas "
                    f"must be padded up to a block boundary, as SDAR's "
                    f"generate.py does."
                )
            mask = block_causal_mask(length, block, device=input_ids.device)
            return {"attention_mask": mask.unsqueeze(0)}
        return {"attention_mask": torch.ones_like(input_ids)}

    def _align(self, logits: torch.Tensor) -> torch.Tensor:
        if self.conventions.logits_alignment == "none":
            return logits
        return torch.cat((logits[:, :1, :], logits[:, :-1, :]), dim=1)

    # ------------------------------------------------------------------
    # Memory-lean read path
    # ------------------------------------------------------------------

    @property
    def supports_lean_logits(self) -> bool:
        return self._backbone is not None and self._lm_head is not None

    # ------------------------------------------------------------------
    # Block diffusion (see sdar_harness/blockwise.py)
    # ------------------------------------------------------------------

    @property
    def backbone(self) -> Any:
        if self._backbone is None:
            raise RuntimeError("This checkpoint does not expose .model")
        return self._backbone

    @property
    def lm_head(self) -> Any:
        if self._lm_head is None:
            raise RuntimeError("This checkpoint does not expose .lm_head")
        return self._lm_head

    @torch.inference_mode()
    def commit(
        self,
        block_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Any,
    ) -> None:
        """Write one finished block's K/V into the cache.

        Separate from the denoising forwards, which all run `store_kv=False` so
        that the K/V of *masked* positions never enters the cache. SDAR's
        reference does exactly this: one extra forward per block, after the
        block is complete. It is real cost and the caller counts it.
        """
        rows = int(block_ids.shape[0])
        if rows != 1:
            raise ValueError("a block is committed once, at batch 1")
        self.model(
            input_ids=block_ids.to(self.device, non_blocking=True),
            attention_mask=attention_mask.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )

    @torch.inference_mode()
    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final backbone hidden states, [B, S, H]. No vocabulary projection."""
        if not self.supports_lean_logits:
            raise RuntimeError("This checkpoint does not expose .model / .lm_head")
        input_ids = input_ids.to(self.device, non_blocking=True)
        outputs = self._backbone(input_ids=input_ids, **self._attention_kwargs(input_ids))
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise RuntimeError(
                f"Unexpected hidden shape {tuple(hidden.shape)} for "
                f"input {tuple(input_ids.shape)}"
            )
        return hidden

    def logits_from_hidden(
        self,
        hidden: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Aligned logits at ``(rows[i], positions[i])``, returned as [N, V].

        Identical to ``forward_logits(input_ids)[rows, positions, :]`` but only
        ever projects N vectors through the vocabulary head instead of B*S.
        """
        if self.conventions.logits_alignment == "shift":
            # forward_logits prepends logits[:, :1] and drops the last position,
            # so aligned[:, p] == raw[:, p - 1] for p >= 1, and raw[:, 0] at p=0.
            source = (positions - 1).clamp(min=0)
        else:
            source = positions
        selected = hidden[rows, source, :]
        return self.project(selected)

    def project(self, selected: torch.Tensor) -> torch.Tensor:
        """Vocabulary projection of already-gathered hidden vectors, [N, V']."""
        return self.limit_vocab(self._lm_head(selected))

    def limit_vocab(self, logits: torch.Tensor) -> torch.Tensor:
        """Drop logits at IDs >= vocab_limit (a view; no copy)."""
        if self.vocab_limit is None or logits.shape[-1] <= self.vocab_limit:
            return logits
        return logits[..., : self.vocab_limit]

    @torch.inference_mode()
    def verify_lean_path(self, *, length: int = 12, tolerance: float | None = None) -> bool:
        """Check that the lean path reproduces ``forward_logits``.

        Runs once at load on a tiny input. If the checkpoint's internals differ
        from what ``logits_from_hidden`` assumes, this degrades to the original
        full-logits path with a warning rather than silently changing results.

        This checks the two read paths against each other. It cannot tell you
        that ``Conventions.logits_alignment`` is right for this checkpoint --
        both paths apply the same convention, so a wrong one agrees with itself.

        The two paths are mathematically identical but not bit-identical: the
        vocabulary projection runs at a different GEMM shape, so BLAS may
        accumulate in a different order. The check asserts a relative tolerance
        plus exact agreement of the argmax, which is what the sampler's
        decisions actually depend on.
        """
        if not self.supports_lean_logits:
            warnings.warn(
                "Checkpoint exposes no .model/.lm_head pair; falling back to "
                "full [B, S, V] logits. Expect high GPU memory use.",
                stacklevel=2,
            )
            return False
        if os.environ.get("AUTOSPEC_DISABLE_LEAN_LOGITS") == "1":
            self._backbone = None
            return False
        # A block-diffusion checkpoint is read under a block-causal mask, which
        # only exists for a whole number of blocks. The default probe is far
        # shorter than one block, so round it up -- otherwise `_attention_kwargs`
        # raises, this degrades to the full-logits path, and a run started with
        # require_lean_logits refuses to load a model that is in fact fine.
        block = self.conventions.block_length
        if block is not None:
            length = max(1, math.ceil(length / block)) * block
        self.probe_length_used = length
        probe = torch.randint(0, 1000, (2, length), device=self.device)
        rows = torch.tensor([0, 1, 0], device=self.device)
        positions = torch.tensor([0, length - 1, length // 2], device=self.device)
        try:
            reference = self.forward_logits(probe)[rows, positions, :]
            lean = self.logits_from_hidden(self.hidden_states(probe), rows, positions)
            scale = reference.float().abs().max().item() or 1.0
            gap = (reference.float() - lean.float()).abs().max().item() / scale
            argmax_agrees = torch.equal(reference.argmax(-1), lean.argmax(-1))
        except Exception as error:  # noqa: BLE001 - degrade, never crash the run
            warnings.warn(
                f"Lean logits self-check raised {error!r}; falling back to full "
                "[B, S, V] logits. Expect high GPU memory use.",
                stacklevel=2,
            )
            self._backbone = None
            return False
        if tolerance is None:
            tolerance = 1e-5 if reference.dtype == torch.float32 else 1e-2
        if gap > tolerance or not argmax_agrees:
            warnings.warn(
                f"Lean logits self-check failed (relative gap {gap:.3g}, argmax "
                f"agrees: {argmax_agrees}); falling back to full [B, S, V] "
                "logits. Expect high GPU memory use.",
                stacklevel=2,
            )
            self._backbone = None
            return False
        self.lean_path_relative_gap = gap
        return True

    # ------------------------------------------------------------------
    # Original read path (kept for tests, probes, and fallback)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device, non_blocking=True)
        outputs = self.model(input_ids=input_ids, **self._attention_kwargs(input_ids))
        logits = outputs.logits
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise RuntimeError(
                f"Unexpected logits shape {tuple(logits.shape)} for "
                f"input {tuple(input_ids.shape)}"
            )
        return self.limit_vocab(self._align(logits))

    def render_prompt(self, prompt: str, *, use_chat_template: bool = True) -> str:
        if not use_chat_template:
            return prompt
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("The selected tokenizer has no chat template.")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode_prompt(self, rendered_prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        if encoded.shape[0] != 1 or encoded.shape[1] == 0:
            raise ValueError("Prompt tokenization produced an empty or batched value.")
        return encoded.to(self.device)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @property
    def excluded_token_ids(self) -> set[int]:
        # Matches ADJUST's `mask_eos: false`, whose only exclusion is
        # `logits[:, :, mask_token_id] = -inf`. EOS/pad stays a committable
        # token; bos is banned only under `mask_eos: true`, which eval never uses.
        return {self.mask_token_id}
