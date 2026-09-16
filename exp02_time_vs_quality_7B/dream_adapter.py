"""

This module owns every model-specific operation used by the experiment:

* loading Dream-Base or Dream-Instruct;
* preparing unconditional, raw-prefix, or chat-conditioned inputs;
* converting a 2-D padding mask to Dream's full-attention representation;
* applying Dream's one-position output-logit alignment;
* filtering structural special tokens consistently;
* ranking remaining canvas positions by top-1 probability;
* executing deterministic, atomic top-k Dream denoising steps;
* scoring fixed token events under the same probability distribution; and
* committing one token without mutating the prior state.

The joint-probability probe should use this adapter instead of calling the
Hugging Face model directly. That keeps all Dream-specific conventions in one
place and prevents the base and conditional passes from drifting apart.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

ConditioningMode = Literal["unconditional", "prefix", "chat"]

BASE_MODEL_ID = "Dream-org/Dream-v0-Base-7B"
INSTRUCT_MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"

## This is one denoising state
@dataclass(frozen=True)
class DreamState:
    """One visible prefix followed by a fixed generation canvas.

    `canvas_mask` marks every location belonging to the generated canvas,
    including locations that have already been committed. The remaining masked
    positions are computed by intersecting `canvas_mask` with the current MASK
    tokens in `input_ids`.
    """

    input_ids: torch.Tensor     ## Shape of [batch, sequence length]; Input sequence at every denoising step
    attention_mask: torch.Tensor    # Identifies which sequence positions are real vs. padded
    canvas_mask: torch.Tensor ##Differentiates between prompt tokens and sequence tokens
    prefix_length: int
    canvas_start: int # Index where generated canvas starts
    canvas_length: int
    mode: ConditioningMode ## One of unconditional, prefix, or chat (prompted)
    visible_text: str | None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if self.input_ids.shape[0] < 1:
            raise ValueError("input_ids must contain at least one batch row.")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids.")
        if self.canvas_mask.shape != self.input_ids.shape:
            raise ValueError("canvas_mask must match input_ids.")
        if self.canvas_mask.dtype != torch.bool:
            raise TypeError("canvas_mask must be a boolean tensor.")
        if self.canvas_start != self.prefix_length:
            raise ValueError("canvas_start must equal prefix_length.")
        if self.canvas_start + self.canvas_length != self.input_ids.shape[1]:
            raise ValueError("The canvas must occupy the suffix of the sequence.")
        # Avoid a device synchronization every time ``dataclasses.replace``
        # constructs a new CUDA state inside the timed denoising loop.
        if self.canvas_mask.device.type == "cpu":
            expected_canvas = torch.zeros_like(self.canvas_mask)
            expected_canvas[:, self.canvas_start :] = True
            if not torch.equal(self.canvas_mask, expected_canvas):
                raise ValueError(
                    "canvas_mask must mark exactly the canvas suffix in every row."
                )


@dataclass(frozen=True)
class TokenPrediction:
    """Top-1 token prediction and confidence at one masked canvas position."""

    sequence_position: int
    canvas_position: int
    token_id: int
    token_text: str
    probability: float
    log_probability: float
    raw_token_id: int
    raw_token_text: str
    raw_probability: float
    excluded_probability_mass: float

    @property
    def changed_by_filter(self) -> bool:
        """Whether structural-token filtering changed the winning token."""

        return self.token_id != self.raw_token_id


@dataclass(frozen=True)
class DreamStepResult:
    """One atomic parallel-unmasking step.

    All tensor fields stay on the adapter device. This is deliberate: the
    timed generation path must not synchronize by copying trace data to CPU.
    """

    state: DreamState
    selected_canvas_positions: torch.Tensor
    selected_sequence_positions: torch.Tensor
    selected_token_ids: torch.Tensor
    selected_confidences: torch.Tensor
    remaining_masks_before: int
    remaining_masks_after: int


@dataclass(frozen=True)
class DreamGenerationResult:
    """Completed Dream denoising output, retained entirely as device tensors."""

    state: DreamState
    continuation_token_ids: torch.Tensor
    denoising_steps: int


@dataclass(frozen=True)
class ModelLoadOptions:
    """Options used by :meth:`DreamAdapter.from_pretrained`."""

    model_id: str
    revision: str | None = None
    cache_dir: Path | None = None
    local_files_only: bool = False
    dtype: torch.dtype = torch.bfloat16
    device: str | torch.device = "cuda:0"


class DreamAdapter:
    """A small, inference-only wrapper around a Dream language model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str | torch.device,
        filter_special_tokens: bool = True,
    ) -> None:
        
        ## Stores model, tokenizer, device, token IDs, and model configs

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.filter_special_tokens = bool(filter_special_tokens)

        self.mask_token_id = self._require_token_id("mask_token_id")
        self.bos_token_id = self._require_token_id("bos_token_id")
        self.config = getattr(model, "config", None)
        if self.config is None:
            raise ValueError("Dream model has no config attribute.")

    @classmethod
    def from_pretrained(
        cls,
        options: ModelLoadOptions,
        *,
        filter_special_tokens: bool = True,
    ) -> "DreamAdapter":
        """Load a frozen Dream checkpoint and tokenizer."""

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install Dream's tested Transformers version first: "
                "pip install transformers==4.46.2"
            ) from exc

        device = torch.device(options.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
        if device.type == "cpu":
            warnings.warn(
                "Dream-7B is extremely slow and large on CPU.",
                stacklevel=2,
            )
        if options.dtype == torch.bfloat16 and device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "This GPU does not report BF16 support. Select float16 "
                    "intentionally or use a BF16-capable GPU."
                )

        common: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": options.local_files_only,
        }
        if options.revision is not None:
            common["revision"] = options.revision
        if options.cache_dir is not None:
            common["cache_dir"] = str(options.cache_dir)


        ## Loads the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(options.model_id, **common)
        ## Loads the model
        model = AutoModel.from_pretrained(
            options.model_id,
            torch_dtype=options.dtype,
            low_cpu_mem_usage=True,
            **common,
        )
        model = model.to(device).eval()
        model.requires_grad_(False)

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            filter_special_tokens=filter_special_tokens,
        )
    
    ## Maps Task (Unconditional, Prefix, Prompted) to model
    @staticmethod
    def default_model_id(mode: ConditioningMode) -> str:
        """Choose Base for unconditional/prefix and Instruct for chat."""

        if mode == "chat":
            return INSTRUCT_MODEL_ID
        if mode in {"unconditional", "prefix"}:
            return BASE_MODEL_ID
        raise ValueError(f"Unsupported conditioning mode: {mode!r}")

    def _require_token_id(self, name: str) -> int:
        token_id = getattr(self.tokenizer, name, None)
        if token_id is None:
            raise ValueError(f"Tokenizer has no {name}.")
        return int(token_id)

    ## Builds input for the three states
    # For example: [BOS] + [MASK]^72 for unconditional
    def prepare_initial_state(
        self,
        *,
        mode: ConditioningMode,
        canvas_length: int,
        text: str | None = None,
    ) -> DreamState:
        """Build a visible prefix and append a fully masked canvas."""

        if canvas_length < 1:
            raise ValueError("canvas_length must be at least 1.")
        if mode in {"prefix", "chat"} and not text:
            raise ValueError(f"text is required for mode={mode!r}.")
        if mode == "unconditional" and text is not None:
            raise ValueError("text is not used in unconditional mode.")

        if mode == "unconditional":
            prefix_ids = torch.tensor(
                [[self.bos_token_id]],
                dtype=torch.long,
            )
            visible_text = None
        elif mode == "prefix":
            assert text is not None
            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            prefix_ids = torch.tensor(
                [[self.bos_token_id, *map(int, text_ids)]],
                dtype=torch.long,
            )
            visible_text = text
        elif mode == "chat":
            assert text is not None
            prefix_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if not isinstance(prefix_ids, torch.Tensor):
                raise TypeError(
                    "apply_chat_template(..., return_tensors='pt') must "
                    "return a torch.Tensor."
                )
            if prefix_ids.ndim == 1:
                prefix_ids = prefix_ids.unsqueeze(0)
            if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
                raise ValueError(
                    "Expected one chat-formatted sequence, got shape "
                    f"{tuple(prefix_ids.shape)}."
                )
            prefix_ids = prefix_ids.to(dtype=torch.long)
            visible_text = text
        else:
            raise ValueError(f"Unsupported conditioning mode: {mode!r}")

        prefix_length = int(prefix_ids.shape[1])
        masks = torch.full(
            (1, canvas_length),
            self.mask_token_id,
            dtype=torch.long,
        )
        input_ids = torch.cat((prefix_ids, masks), dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        canvas_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        canvas_mask[:, prefix_length:] = True

        return DreamState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            canvas_mask=canvas_mask,
            prefix_length=prefix_length,
            canvas_start=prefix_length,
            canvas_length=canvas_length,
            mode=mode,
            visible_text=visible_text,
        )


    ## Constructs [BOS, saved_prefix_tokens, MASK * L]
    def prepare_prefix_batch(
        self,
        prefix_token_ids: Sequence[Sequence[int]] | torch.Tensor,
        *,
        canvas_length: int,
        add_bos: bool = True,
    ) -> DreamState:
        """Build ``[BOS, prefix, MASK^L]`` for fixed-length tokenized prefixes.

        The OpenWebText experiment saves Dream token IDs once and reuses them
        for every schedule. Accepting token IDs here avoids retokenization and
        makes it explicit that every ``k`` sees the identical model input.
        All rows must have the same prefix length; the planned dataset uses
        ``P=64`` for every row.
        """

        if canvas_length < 1:
            raise ValueError("canvas_length must be at least 1.")

        prefixes = torch.as_tensor(prefix_token_ids, dtype=torch.long)
        if prefixes.ndim != 2:
            raise ValueError("prefix_token_ids must have shape [batch, prefix].")
        if prefixes.shape[0] < 1 or prefixes.shape[1] < 1:
            raise ValueError("prefix_token_ids must contain a non-empty batch.")

        vocab_size = int(getattr(self.config, "vocab_size"))
        if torch.any((prefixes < 0) | (prefixes >= vocab_size)):
            raise IndexError("A prefix token ID is outside the model vocabulary.")

        prefixes = prefixes.to(self.device)
        if add_bos:
            bos = torch.full(
                (prefixes.shape[0], 1),
                self.bos_token_id,
                device=self.device,
                dtype=torch.long,
            )
            prefixes = torch.cat((bos, prefixes), dim=1)

        prefix_length = int(prefixes.shape[1])
        masks = torch.full(
            (prefixes.shape[0], canvas_length),
            self.mask_token_id,
            device=self.device,
            dtype=torch.long,
        )
        input_ids = torch.cat((prefixes, masks), dim=1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        canvas_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        canvas_mask[:, prefix_length:] = True

        return DreamState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            canvas_mask=canvas_mask,
            prefix_length=prefix_length,
            canvas_start=prefix_length,
            canvas_length=canvas_length,
            mode="prefix",
            visible_text=None,
        )

    @staticmethod
    def align_logits(raw_logits: torch.Tensor) -> torch.Tensor:
        """Apply Dream's official one-position output alignment.

        Raw outputs ``[z_0, z_1, ..., z_{n-1}]`` become
        ``[z_0, z_0, z_1, ..., z_{n-2}]``. Position zero is never scored in
        our canvas, so the duplicated first output is harmless.
        """

        if raw_logits.ndim != 3 or raw_logits.shape[1] < 1:
            raise ValueError(
                "Expected logits [batch, sequence, vocabulary], got "
                f"{tuple(raw_logits.shape)}."
            )
        return torch.cat((raw_logits[:, :1], raw_logits[:, :-1]), dim=1)

    @staticmethod
    def _prepare_model_attention(
        attention_mask: torch.Tensor | None,
    ) -> tuple[str | torch.Tensor, torch.Tensor | None]:
        """Convert a 2-D padding mask to Dream's model inputs.

        For an unpadded sequence, Dream's fast path is the string ``"full"``
        with no explicit position IDs. For padded batches, this mirrors the
        pairwise boolean mask and position-index construction in Dream's
        generation implementation.
        """

        if attention_mask is None:
            return "full", None
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence].")
        if torch.all(attention_mask != 0):
            return "full", None

        valid = attention_mask.to(dtype=torch.bool)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(~valid, 1)
        pairwise = torch.logical_and(
            valid.unsqueeze(1).unsqueeze(-2),
            valid.unsqueeze(1).unsqueeze(-1),
        )
        return pairwise, position_ids

    def forward_aligned_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Take Model input IDs, Run Dream once and return aligned, unnormalized logits."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.device != self.device:
            raise ValueError(
                f"input_ids are on {input_ids.device}, adapter uses {self.device}."
            )
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids.")
            attention_mask = attention_mask.to(self.device)

        model_attention, position_ids = self._prepare_model_attention(attention_mask)

        ## Runs the model
        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                model_attention,
                position_ids,
                use_cache=False,
                return_dict=True,
            )
        raw_logits = getattr(outputs, "logits", None)
        if raw_logits is None:
            raise AttributeError("Dream model output has no logits attribute.")
        if raw_logits.shape[:2] != input_ids.shape:
            raise ValueError(
                "Model logits do not match input shape: "
                f"input={tuple(input_ids.shape)}, logits={tuple(raw_logits.shape)}."
            )
        return self.align_logits(raw_logits)

    def remaining_canvas_mask(self, state: DreamState) -> torch.Tensor:
        """Return canvas locations that are still represented by MASK."""

        self._validate_state_device(state)
        return state.canvas_mask & (state.input_ids == self.mask_token_id)

    def remaining_positions(self, state: DreamState) -> torch.Tensor:
        """Return remaining masked canvas positions for the single sequence."""

        if state.input_ids.shape[0] != 1:
            raise ValueError(
                "remaining_positions is a single-sequence probe API; use "
                "remaining_canvas_mask for batched generation."
            )
        return torch.nonzero(
            self.remaining_canvas_mask(state)[0],
            as_tuple=False,
        ).flatten()

    def collect_excluded_token_ids(self, vocab_size: int) -> set[int]:
        """Combine tokenizer and model-config structural special-token IDs."""

        values = list(getattr(self.tokenizer, "all_special_ids", []))
        for source in (self.tokenizer, self.config):
            for name in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "mask_token_id",
            ):
                values.append(getattr(source, name, None))

        result: set[int] = set()
        for value in values:
            if value is not None and 0 <= int(value) < vocab_size:
                result.add(int(value))
        return result

    ## Turn logits into probabitlies
    def _filtered_log_probs(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, set[int], torch.Tensor]:
        """Return FP32 log probabilities and removed raw probability mass."""

        if logits.ndim != 2:
            raise ValueError("Expected logits [events, vocabulary].")
        logits_fp32 = logits.float()
        raw_log_probs = F.log_softmax(logits_fp32, dim=-1)

        excluded: set[int] = set()
        excluded_mass = torch.zeros(
            logits_fp32.shape[0],
            device=logits_fp32.device,
            dtype=torch.float32,
        )
        scoring_logits = logits_fp32

        if self.filter_special_tokens:
            excluded = self.collect_excluded_token_ids(logits_fp32.shape[-1])
            if excluded:
                excluded_tensor = torch.tensor(
                    sorted(excluded),
                    device=logits_fp32.device,
                    dtype=torch.long,
                )
                excluded_mass = torch.exp(
                    torch.logsumexp(
                        raw_log_probs.index_select(-1, excluded_tensor),
                        dim=-1,
                    )
                )
                scoring_logits = logits_fp32.clone()
                scoring_logits.index_fill_(-1, excluded_tensor, -torch.inf)

        if torch.isneginf(scoring_logits).all(dim=-1).any():
            raise ValueError("Special-token filtering removed the whole vocabulary.")

        return F.log_softmax(scoring_logits, dim=-1), excluded, excluded_mass

    def _filtered_top1(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return filtered top-1 token IDs and probabilities in FP32.

        Unlike :meth:`_filtered_log_probs`, this avoids materializing the full
        log-probability tensor. Dream-mode decoding only needs the winning
        token and its normalized probability at each remaining position.
        """

        if logits.ndim != 2:
            raise ValueError("Expected logits [events, vocabulary].")
        scoring_logits = logits.float()

        if self.filter_special_tokens:
            excluded = self.collect_excluded_token_ids(scoring_logits.shape[-1])
            if excluded:
                if len(excluded) >= scoring_logits.shape[-1]:
                    raise ValueError(
                        "Special-token filtering removed the whole vocabulary."
                    )
                excluded_tensor = torch.tensor(
                    sorted(excluded),
                    device=scoring_logits.device,
                    dtype=torch.long,
                )
                scoring_logits.index_fill_(-1, excluded_tensor, -torch.inf)

        best_logits, best_ids = scoring_logits.max(dim=-1)
        log_normalizers = torch.logsumexp(scoring_logits, dim=-1)
        confidences = torch.exp(best_logits - log_normalizers)
        return best_ids, confidences

    def _dream_step_with_count(
        self,
        state: DreamState,
        *,
        commit_count: int,
        remaining_count: int,
    ) -> tuple[DreamState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-resident implementation shared by timed and traced decoding."""

        batch_size = state.input_ids.shape[0]
        canvas_slice = slice(
            state.canvas_start,
            state.canvas_start + state.canvas_length,
        )
        remaining_canvas = self.remaining_canvas_mask(state)[:, canvas_slice]
        masked_canvas_positions = torch.argsort(
            remaining_canvas.to(dtype=torch.int8),
            dim=-1,
            descending=True,
            stable=True,
        )[:, :remaining_count]

        # Dream-mode experiment batches are fixed-length and unpadded. Passing
        # ``None`` selects Dream's full-attention fast path without testing an
        # all-ones CUDA mask (which would synchronize on every step).
        aligned_logits = self.forward_aligned_logits(state.input_ids, None)
        canvas_logits = aligned_logits[:, canvas_slice, :]
        masked_logits = canvas_logits.gather(
            1,
            masked_canvas_positions.unsqueeze(-1).expand(
                -1,
                -1,
                canvas_logits.shape[-1],
            ),
        )
        best_ids, confidences = self._filtered_top1(
            masked_logits.reshape(-1, masked_logits.shape[-1])
        )
        best_ids = best_ids.reshape(batch_size, remaining_count)
        confidences = confidences.reshape(batch_size, remaining_count)

        # Masked candidates are in ascending sequence order. Stable descending
        # sort therefore implements the specified lower-position tie-break.
        confidence_order = torch.argsort(
            confidences,
            dim=-1,
            descending=True,
            stable=True,
        )
        selected_candidate_indices = confidence_order[:, :commit_count]
        selected_canvas_positions = masked_canvas_positions.gather(
            1,
            selected_candidate_indices,
        )
        selected_positions = selected_canvas_positions + state.canvas_start
        selected_ids = best_ids.gather(1, selected_candidate_indices)
        selected_confidences = confidences.gather(
            1,
            selected_candidate_indices,
        )

        # The single scatter is the atomic commitment boundary: every token
        # above came from the same pre-commit forward pass.
        new_input_ids = state.input_ids.clone()
        new_input_ids.scatter_(1, selected_positions, selected_ids)
        new_state = replace(state, input_ids=new_input_ids)
        return (
            new_state,
            selected_positions,
            selected_ids,
            selected_confidences,
        )

    def dream_step(
        self,
        state: DreamState,
        *,
        k: int,
    ) -> DreamStepResult:
        """Run one deterministic top-``k`` parallel-unmasking step.

        This public step API is intended for untimed trajectory collection.
        :meth:`generate_dream` uses the same implementation without per-step
        host transfers or synchronization.
        """

        if k < 1:
            raise ValueError("k must be at least 1.")
        self._validate_state_device(state)
        if not torch.all(state.attention_mask != 0).item():
            raise ValueError("dream_step requires an unpadded state.")

        counts = self.remaining_canvas_mask(state).sum(dim=-1)
        first_count = int(counts[0].item())
        if first_count < 1:
            raise ValueError("The state has no remaining masks.")
        if not torch.all(counts == first_count).item():
            raise ValueError(
                "Every batch row must have the same number of remaining masks."
            )

        commit_count = min(k, first_count)
        (
            next_state,
            selected_positions,
            selected_ids,
            selected_confidences,
        ) = self._dream_step_with_count(
            state,
            commit_count=commit_count,
            remaining_count=first_count,
        )
        return DreamStepResult(
            state=next_state,
            selected_canvas_positions=(
                selected_positions - state.canvas_start
            ),
            selected_sequence_positions=selected_positions,
            selected_token_ids=selected_ids,
            selected_confidences=selected_confidences,
            remaining_masks_before=first_count,
            remaining_masks_after=first_count - commit_count,
        )

    def generate_dream(
        self,
        initial_state: DreamState,
        *,
        k: int,
    ) -> DreamGenerationResult:
        """Fill a fresh canvas with deterministic atomic top-``k`` decoding.

        The loop performs no detokenization, CPU transfer, serialization, or
        trajectory recording. It is therefore suitable for the synchronized
        timed region in the experiment. ``initial_state`` must contain a fully
        masked canvas in every batch row.
        """

        if k < 1:
            raise ValueError("k must be at least 1.")
        self._validate_state_device(initial_state)
        if not torch.all(initial_state.attention_mask != 0).item():
            raise ValueError("generate_dream requires an unpadded state.")
        initial_remaining = self.remaining_canvas_mask(initial_state)
        if not torch.all(initial_remaining == initial_state.canvas_mask).item():
            raise ValueError(
                "generate_dream requires a fully masked initial canvas."
            )

        state = initial_state
        remaining = initial_state.canvas_length
        denoising_steps = 0
        while remaining:
            commit_count = min(k, remaining)
            state, _, _, _ = self._dream_step_with_count(
                state,
                commit_count=commit_count,
                remaining_count=remaining,
            )
            remaining -= commit_count
            denoising_steps += 1

        if torch.any(self.remaining_canvas_mask(state)).item():
            raise RuntimeError("Dream generation finished with uncommitted masks.")
        continuation = state.input_ids[
            :,
            state.canvas_start : state.canvas_start + state.canvas_length,
        ]
        return DreamGenerationResult(
            state=state,
            continuation_token_ids=continuation,
            denoising_steps=denoising_steps,
        )

    def rank_remaining_positions(
        self,
        state: DreamState,
        aligned_logits: torch.Tensor,
        *,
        top_n: int,
    ) -> list[TokenPrediction]:
        """Rank remaining canvas positions by their top-1 probability."""

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        self._validate_state_device(state)
        if aligned_logits.shape[:2] != state.input_ids.shape:
            raise ValueError("aligned_logits must match state.input_ids.")
        if aligned_logits.shape[0] != 1:
            raise ValueError("Ranking currently expects one state.")

        positions = self.remaining_positions(state)
        if positions.numel() == 0:
            return []

        position_logits = aligned_logits[0].index_select(0, positions)
        log_probs, _, excluded_mass = self._filtered_log_probs(position_logits)
        best_logp, best_id = log_probs.max(dim=-1)

        raw_log_probs = F.log_softmax(position_logits.float(), dim=-1)
        raw_best_logp, raw_best_id = raw_log_probs.max(dim=-1)

        positions_list = positions.detach().cpu().tolist()
        best_logp_list = best_logp.detach().cpu().tolist()
        best_id_list = best_id.detach().cpu().tolist()
        raw_best_logp_list = raw_best_logp.detach().cpu().tolist()
        raw_best_id_list = raw_best_id.detach().cpu().tolist()
        excluded_mass_list = excluded_mass.detach().cpu().tolist()

        order = sorted(
            range(len(positions_list)),
            key=lambda index: (-best_logp_list[index], positions_list[index]),
        )[: min(top_n, len(positions_list))]

        predictions: list[TokenPrediction] = []
        for index in order:
            token_id = int(best_id_list[index])
            raw_token_id = int(raw_best_id_list[index])
            sequence_position = int(positions_list[index])
            predictions.append(
                TokenPrediction(
                    sequence_position=sequence_position,
                    canvas_position=sequence_position - state.canvas_start,
                    token_id=token_id,
                    token_text=self.decode_token(token_id),
                    probability=math.exp(float(best_logp_list[index])),
                    log_probability=float(best_logp_list[index]),
                    raw_token_id=raw_token_id,
                    raw_token_text=self.decode_token(raw_token_id),
                    raw_probability=math.exp(float(raw_best_logp_list[index])),
                    excluded_probability_mass=float(excluded_mass_list[index]),
                )
            )
        return predictions

    def event_log_probabilities(
        self,
        aligned_logits: torch.Tensor,
        *,
        sequence_positions: Sequence[int] | torch.Tensor,
        token_ids: Sequence[int] | torch.Tensor,
        batch_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score fixed token events under the adapter's probability policy.

        Two common uses are supported:

        * one base state with several events: logits batch size is 1 and
          ``batch_indices`` is omitted;
        * one event per conditional state: number of events equals the logits
          batch size and ``batch_indices`` is omitted.

        An explicit ``batch_indices`` vector supports any other arrangement.
        The returned tensor is FP32 and remains on the logits device.
        """

        if aligned_logits.ndim != 3:
            raise ValueError("aligned_logits must be [batch, sequence, vocabulary].")

        positions = torch.as_tensor(
            sequence_positions,
            device=aligned_logits.device,
            dtype=torch.long,
        ).flatten()
        targets = torch.as_tensor(
            token_ids,
            device=aligned_logits.device,
            dtype=torch.long,
        ).flatten()
        if positions.numel() == 0:
            return torch.empty(0, device=aligned_logits.device, dtype=torch.float32)
        if positions.shape != targets.shape:
            raise ValueError("sequence_positions and token_ids must have equal length.")

        event_count = positions.numel()
        batch_size, sequence_length, vocab_size = aligned_logits.shape
        if batch_indices is None:
            if batch_size == 1:
                batches = torch.zeros(
                    event_count,
                    device=aligned_logits.device,
                    dtype=torch.long,
                )
            elif batch_size == event_count:
                batches = torch.arange(
                    batch_size,
                    device=aligned_logits.device,
                    dtype=torch.long,
                )
            else:
                raise ValueError(
                    "Omit batch_indices only for one shared state or one event "
                    "per batch row."
                )
        else:
            batches = torch.as_tensor(
                batch_indices,
                device=aligned_logits.device,
                dtype=torch.long,
            ).flatten()
            if batches.shape != positions.shape:
                raise ValueError("batch_indices must match the number of events.")

        if torch.any((batches < 0) | (batches >= batch_size)):
            raise IndexError("A batch index is outside aligned_logits.")
        if torch.any((positions < 0) | (positions >= sequence_length)):
            raise IndexError("A sequence position is outside aligned_logits.")
        if torch.any((targets < 0) | (targets >= vocab_size)):
            raise IndexError("A token ID is outside the vocabulary.")

        event_logits = aligned_logits[batches, positions]
        log_probs, excluded, _ = self._filtered_log_probs(event_logits)
        if excluded and any(int(token) in excluded for token in targets.tolist()):
            raise ValueError("Cannot score an excluded structural special token.")
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def commit_token(
        self,
        state: DreamState,
        *,
        sequence_position: int,
        token_id: int,
    ) -> DreamState:
        """Return a new state with one currently masked canvas token revealed."""

        self._validate_state_device(state)
        if state.input_ids.shape[0] != 1:
            raise ValueError("commit_token is a single-sequence probe API.")
        if not 0 <= sequence_position < state.input_ids.shape[1]:
            raise IndexError("sequence_position is outside the state.")
        if not bool(state.canvas_mask[0, sequence_position].item()):
            raise ValueError("Can only commit a token inside the generation canvas.")
        if int(state.input_ids[0, sequence_position].item()) != self.mask_token_id:
            raise ValueError("The requested canvas position is already committed.")
        vocab_size = int(getattr(self.config, "vocab_size"))
        if not 0 <= token_id < vocab_size:
            raise IndexError("token_id is outside the model vocabulary.")
        if self.filter_special_tokens:
            excluded = self.collect_excluded_token_ids(vocab_size)
            if token_id in excluded:
                raise ValueError("Cannot commit an excluded structural special token.")

        new_input_ids = state.input_ids.clone()
        new_input_ids[0, sequence_position] = int(token_id)
        return replace(state, input_ids=new_input_ids)

    def decode_token(self, token_id: int) -> str:
        """Decode exactly one token while preserving visible whitespace."""

        text = self.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return repr(text)

    def _validate_state_device(self, state: DreamState) -> None:
        if state.input_ids.device != self.device:
            raise ValueError(
                f"State is on {state.input_ids.device}, adapter uses {self.device}."
            )
        if state.attention_mask.device != self.device:
            raise ValueError("state.attention_mask is on the wrong device.")
        if state.canvas_mask.device != self.device:
            raise ValueError("state.canvas_mask is on the wrong device.")
