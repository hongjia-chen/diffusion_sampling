"""CPU unit tests for the adaptive Dream scheduler policy."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch

from adaptive_scheduler import (
    AdaptiveSchedulerConfig,
    adaptive_step,
    generate_adaptive,
)


@dataclass(frozen=True)
class FakeState:
    input_ids: torch.Tensor
    canvas_mask: torch.Tensor
    canvas_start: int
    canvas_length: int


class IndependentAdapter:
    """Position-specific proposals unaffected by prior commitments."""

    mask_token_id = 9
    filter_special_tokens = True

    def collect_excluded_token_ids(self, vocab_size: int) -> set[int]:
        return {self.mask_token_id}

    def remaining_canvas_mask(self, state: FakeState) -> torch.Tensor:
        return state.canvas_mask & (state.input_ids == self.mask_token_id)

    def forward_aligned_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = torch.full((*input_ids.shape, 10), -5.0)
        for sequence_position in range(1, input_ids.shape[1]):
            token_id = sequence_position
            logits[:, sequence_position, token_id] = (
                8.0 - sequence_position
            )
            logits[:, sequence_position, 0] = 0.0
        return logits


class DependentAdapter(IndependentAdapter):
    """Makes the second frozen event unlikely after accepting the first."""

    def forward_aligned_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = super().forward_aligned_logits(input_ids, attention_mask)
        if int(input_ids[0, 1]) == 1:
            # Candidate token 2 remains the frozen event even though token 7
            # becomes the conditional argmax.
            logits[0, 2, 2] = -8.0
            logits[0, 2, 7] = 8.0
        return logits


def make_state(length: int) -> FakeState:
    input_ids = torch.tensor([[0] + [9] * length], dtype=torch.long)
    canvas_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    canvas_mask[:, 1:] = True
    return FakeState(
        input_ids=input_ids,
        canvas_mask=canvas_mask,
        canvas_start=1,
        canvas_length=length,
    )


class AdaptiveSchedulerTests(unittest.TestCase):
    def test_short_circuit_and_frozen_token(self) -> None:
        adapter = DependentAdapter()
        initial = make_state(3)
        result = adaptive_step(
            adapter,
            initial,
            config=AdaptiveSchedulerConfig(tau=0.01, max_candidates=3),
            remaining_count=3,
        )

        self.assertEqual(result.candidate_token_ids.tolist(), [1, 2, 3])
        self.assertEqual(result.accepted_prefix_length, 1)
        self.assertEqual(result.conditional_probes, 1)
        self.assertEqual(len(result.conditional_log_factors), 1)
        self.assertGreater(result.cumulative_errors[-1], 0.01)
        self.assertEqual(result.state.input_ids.tolist(), [[0, 1, 9, 9]])
        # The caller-owned state was not partially mutated by probing.
        self.assertEqual(initial.input_ids.tolist(), [[0, 9, 9, 9]])

    def test_large_threshold_accepts_every_frozen_candidate(self) -> None:
        result = adaptive_step(
            DependentAdapter(),
            make_state(3),
            config=AdaptiveSchedulerConfig(tau=100.0, max_candidates=3),
            remaining_count=3,
        )
        self.assertEqual(result.accepted_prefix_length, 3)
        self.assertEqual(result.conditional_probes, 2)
        self.assertEqual(result.state.input_ids.tolist(), [[0, 1, 2, 3]])

    def test_generation_uses_tail_candidate_count(self) -> None:
        result = generate_adaptive(
            IndependentAdapter(),
            make_state(5),
            config=AdaptiveSchedulerConfig(tau=0.0, max_candidates=3),
        )
        self.assertEqual(
            result.continuation_token_ids.tolist(),
            [[1, 2, 3, 4, 5]],
        )
        self.assertEqual(result.scheduler_steps, 2)
        self.assertEqual(result.conditional_probes, 3)
        self.assertEqual(result.accepted_prefix_histogram, (0, 1, 1))

    def test_rejects_unimplemented_microbatching(self) -> None:
        with self.assertRaises(NotImplementedError):
            AdaptiveSchedulerConfig(
                tau=0.1,
                max_candidates=8,
                conditional_microbatch_size=2,
            )


if __name__ == "__main__":
    unittest.main()
