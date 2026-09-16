## Loads the Dream model and tokenizer
## Constructs prompts, runs forward passes, and aligns logits with token positions

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch


DEFAULT_MODEL = "Dream-org/Dream-v0-Instruct-7B"


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


@dataclass(frozen=True)
class ModelOptions:
    model_id: str = DEFAULT_MODEL
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    cache_dir: str | None = None
    local_files_only: bool = False
    # At verifier microbatch sizes above ~2 the full [B, S, V] fallback is not a
    # degraded path, it is an immediate OOM. Fail loudly instead.
    require_lean_logits: bool = False


class DreamLogitsProvider:
    """Thin wrapper around the official remote-code Dream checkpoint.

    Dream's lm-eval wrapper aligns the model output by shifting logits one
    position to the right. This class uses that same convention.

    Two read paths expose the same numbers:

    ``forward_logits``  materializes the whole [B, S, V] logits tensor. Simple,
    but at V=152064 a single 8-row verifier microbatch costs well over a
    gigabyte, which does not fit alongside 7B bfloat16 weights on a 16 GB card.

    ``hidden_states`` + ``logits_from_hidden`` keep only the [B, S, H] backbone
    output (H=3584, so ~45x smaller) and apply the vocabulary projection to the
    handful of positions the scheduler actually reads. Prefer this path.
    """

    def __init__(self, model: Any, tokenizer: Any, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        required = {
            "mask_token_id": tokenizer.mask_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Dream tokenizer is missing required IDs: {missing}")
        self.mask_token_id = int(tokenizer.mask_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.pad_token_id = (
            int(tokenizer.pad_token_id)
            if tokenizer.pad_token_id is not None
            else self.eos_token_id
        )
        self.bos_token_id = (
            int(tokenizer.bos_token_id)
            if tokenizer.bos_token_id is not None
            else None
        )
        self._backbone = getattr(model, "model", None)
        self._lm_head = getattr(model, "lm_head", None)

    @classmethod
    def from_pretrained(cls, options: ModelOptions) -> "DreamLogitsProvider":
        from transformers import AutoModel, AutoTokenizer

        common = {
            "revision": options.revision,
            "cache_dir": options.cache_dir,
            "local_files_only": options.local_files_only,
            "trust_remote_code": True,
        }
        common = {key: value for key, value in common.items() if value is not None}
        tokenizer = AutoTokenizer.from_pretrained(options.model_id, **common)
        model = AutoModel.from_pretrained(
            options.model_id,
            torch_dtype=resolve_dtype(options.dtype),
            **common,
        )
        device = torch.device(options.device)
        model = model.to(device).eval().requires_grad_(False)
        provider = cls(model=model, tokenizer=tokenizer, device=device)
        if not provider.verify_lean_path() and options.require_lean_logits:
            raise RuntimeError(
                "The lean logits path is unavailable, but require_lean_logits is "
                "set. The full [B, S, V] fallback cannot fit a verifier "
                "microbatch above ~2; refusing to continue rather than OOM "
                "partway through a run."
            )
        return provider

    # ------------------------------------------------------------------
    # Memory-lean read path
    # ------------------------------------------------------------------

    @property
    def supports_lean_logits(self) -> bool:
        return self._backbone is not None and self._lm_head is not None

    @torch.inference_mode()
    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final backbone hidden states, [B, S, H]. No vocabulary projection."""
        if not self.supports_lean_logits:
            raise RuntimeError("This checkpoint does not expose .model / .lm_head")
        input_ids = input_ids.to(self.device, non_blocking=True)
        outputs = self._backbone(input_ids=input_ids, attention_mask="full")
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise RuntimeError(
                f"Unexpected Dream hidden shape {tuple(hidden.shape)} for "
                f"input {tuple(input_ids.shape)}"
            )
        return hidden

    def logits_from_hidden(
        self,
        hidden: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Shifted logits at ``(rows[i], positions[i])``, returned as [N, V].

        Identical to ``forward_logits(input_ids)[rows, positions, :]`` but only
        ever projects N vectors through the vocabulary head instead of B*S.
        """
        # forward_logits prepends logits[:, :1] and drops the last position, so
        # shifted[:, p] == raw[:, p - 1] for p >= 1 and shifted[:, 0] == raw[:, 0].
        source = (positions - 1).clamp_(min=0)
        selected = hidden[rows, source, :]
        return self._lm_head(selected)

    @torch.inference_mode()
    def verify_lean_path(self, *, length: int = 12, tolerance: float | None = None) -> bool:
        """Check that the lean path reproduces ``forward_logits``.

        Runs once at load on a tiny input. If the checkpoint's internals differ
        from what ``logits_from_hidden`` assumes, this degrades to the original
        full-logits path with a warning rather than silently changing results.

        The two paths are mathematically identical but not bit-identical: the
        vocabulary projection runs at a different GEMM shape ([N, H] @ [H, V]
        instead of [B, S, H] @ [H, V]), so BLAS may accumulate in a different
        order. On float32 the gap is about one ULP; in bfloat16 it is larger.
        The check therefore asserts a relative tolerance plus exact agreement of
        the argmax, which is what the sampler's decisions actually depend on.
        """
        if not self.supports_lean_logits:
            warnings.warn(
                "Dream checkpoint exposes no .model/.lm_head pair; falling back "
                "to full [B, S, V] logits. Expect high GPU memory use.",
                stacklevel=2,
            )
            return False
        if os.environ.get("DREAM_DISABLE_LEAN_LOGITS") == "1":
            self._backbone = None
            return False
        # Any low token ids are valid for a 150k vocabulary.
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
        attention_mask = "full"
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise RuntimeError(
                f"Unexpected Dream logits shape {tuple(logits.shape)} for "
                f"input {tuple(input_ids.shape)}"
            )
        return torch.cat((logits[:, :1, :], logits[:, :-1, :]), dim=1)

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
        # ADJUST runs every downstream eval with `mask_eos: false`, whose only
        # exclusion is `logits[:, :, mask_token_id] = -inf`. EOS/pad stays a
        # committable token (they assert on this in `evaluate_lm`), and bos is
        # banned only under `mask_eos: true`, which eval never uses.
        return {self.mask_token_id}
