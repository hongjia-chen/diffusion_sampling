"""Shared Dream-7B model adapter for factorization experiments.

This module owns every model-specific operation used by the experiment:

* loading Dream-Base or Dream-Instruct;
* preparing unconditional, raw-prefix, or chat-conditioned inputs;
* converting a 2-D padding mask to Dream's full-attention representation;
* applying Dream's one-position output-logit alignment;
* filtering structural special tokens consistently;
* ranking remaining canvas positions by top-1 probability;
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
        if self.input_ids.shape[0] != 1:
            raise ValueError("DreamState currently supports one sequence at a time.")
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
        if int(self.canvas_mask.sum().item()) != self.canvas_length:
            raise ValueError("canvas_mask must contain exactly canvas_length true values.")


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
