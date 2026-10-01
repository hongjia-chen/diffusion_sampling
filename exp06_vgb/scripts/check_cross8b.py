"""Real-model consistency check for the block-verifier cross path (DESIGN.md §10).

Runs `generate_cross` on a few tasks with every proposer call, verification
call and remask pass re-computed by an independent, uncached full-sequence
forward (prompt + canvas so far, block-causal mask, no KV cache). A wrong
position, a stale or missing verifier cache, a wrong mask or mask token shows
up as nats; bf16 kernel noise shows up as ~1e-2. Also checks that each output
reproduces the pilot record of the same task (cross-process determinism), and
records per-card peak memory on the longest HumanEval+ prompt.

    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../exp05_adaptive_re/vendor_env/tf452:. \\
        python scripts/check_cross8b.py --roles 1p7b_proposes
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import adaptive_re.cross as cross  # noqa: E402
from adaptive_re.model import MaskedDiffusionProvider, ModelOptions  # noqa: E402
from adaptive_re.scheduler import _filtered_log_probs  # noqa: E402
from run_benchmark import parse_method, select_sample, task_seed  # noqa: E402

SMALL = PROJECT.parent / "models/SDAR-1.7B-Chat-b64-local"
BIG = PROJECT.parent / "models/SDAR-8B-Chat-b64-local"
DATA = PROJECT.parent / "data"
CANVAS = {"gsm8k": 256, "mbpp_plus": 256, "humaneval_plus": 512}
VOCAB_LIMIT = 151665
SAMPLING = {"fixed": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}


def load_models(roles: str):
    small = ModelOptions(model_id=str(SMALL), family="sdar", block_length=64, device="cuda:1",
                         local_files_only=True, require_lean_logits=True, vocab_limit=VOCAB_LIMIT)
    big = ModelOptions(model_id=str(BIG), family="sdar", block_length=64, devices=("cuda:0", "cuda:1"),
                       local_files_only=True, require_lean_logits=True, vocab_limit=VOCAB_LIMIT)
    proposer, verifier = (small, big) if roles == "1p7b_proposes" else (big, small)
    return MaskedDiffusionProvider.from_pretrained(proposer), MaskedDiffusionProvider.from_pretrained(verifier)


class Recorder:
    """Wraps the cross module's calls and re-computes each one without a cache."""

    def __init__(self, proposer, verifier):
        self.proposer, self.verifier = proposer, verifier
        self.prefix: list[torch.Tensor] = []     # blocks committed to the verifier cache
        self.worst = {"propose_logp": 0.0, "verify_marginal": 0.0, "verify_conditional": 0.0,
                      "remask_logp": 0.0}
        self.counts = {"propose_calls": 0, "propose_token_mismatch": 0, "verify_calls": 0,
                       "accept_mismatch": 0, "remask_passes": 0, "remask_argmax_mismatch": 0,
                       "remask_choice_mismatch": 0, "block_mode": 0}
        self.worst_cases: dict[str, list] = {"propose": [], "verify": [], "remask": []}
        self.task = None
        self.orig = (cross._propose, cross.dream_verify, cross.remask_pass, verifier.commit)
        cross._propose, cross.dream_verify, cross.remask_pass = self.propose, self.verify, self.remask
        original_commit = verifier.commit

        def commit(block_ids, *args, **kwargs):
            self.prefix.append(block_ids.detach().to("cpu").clone())
            return original_commit(block_ids, *args, **kwargs)

        verifier.commit = commit

    def reset(self, task=None) -> None:
        self.prefix = []
        self.task = task

    def keep(self, kind: str, gap: float, **case) -> None:
        """Keep the 5 largest gaps per call kind, with enough state to replay them."""
        cases = self.worst_cases[kind]
        cases.append(dict(case, gap=gap, task=self.task))
        cases.sort(key=lambda c: -c["gap"])
        del cases[5:]

    def full(self, block: torch.Tensor) -> tuple[torch.Tensor, int]:
        prefix = torch.cat(self.prefix, dim=1) if self.prefix else block[:, :0].cpu()
        return torch.cat((prefix, block.detach().cpu()), dim=1), int(prefix.shape[1])

    def reference(self, provider, rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Uncached log-probs at (row i, positions[i]), in groups of 4 rows."""
        out = []
        for low in range(0, rows.shape[0], 4):
            chunk = rows[low : low + 4]
            hidden = provider.hidden_states(chunk)
            index = torch.arange(chunk.shape[0], device=hidden.device)
            logits = provider.logits_from_hidden(hidden, index, positions[low : low + 4].to(hidden.device))
            out.append(_filtered_log_probs(logits, provider.excluded_token_ids).float().cpu())
        return torch.cat(out)

    def single(self, provider, state: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Uncached log-probs at several positions of ONE state, one forward."""
        hidden = provider.hidden_states(state)
        rows = torch.zeros(len(positions), dtype=torch.long, device=hidden.device)
        logits = provider.logits_from_hidden(hidden, rows, positions.to(hidden.device))
        return _filtered_log_probs(logits, provider.excluded_token_ids).float().cpu()

    # -- proposer --------------------------------------------------------
    def propose(self, scoped, state, remaining, max_candidates, **kwargs):
        result = self.orig[0](scoped, state, remaining, max_candidates, **kwargs)
        full, offset = self.full(state)
        ref = self.single(self.proposer, full, remaining.cpu() + offset)
        top2 = torch.topk(ref, 2, dim=-1).values
        ref_logp, ref_tok = ref.max(-1)
        margin = (top2[:, 0] - top2[:, 1]).tolist()
        # At temperature 0 `confidences` is the argmax token's probability, not its log.
        chosen = {int(p): (int(t), math.log(max(float(c), 1e-45))) for p, t, c in zip(
            result.sequence_positions.tolist(), result.token_ids.tolist(), result.confidences.tolist())}
        self.counts["propose_positions"] = self.counts.get("propose_positions", 0)
        for p, t, lp, m in zip(remaining.tolist(), ref_tok.tolist(), ref_logp.tolist(), margin):
            if p in chosen:
                gap = abs(chosen[p][1] - lp)
                self.counts["propose_positions"] += 1
                self.counts["propose_token_mismatch"] += int(chosen[p][0] != t)
                # A mismatch the reference is not near-tied on (top-2 margin > 0.1 nats).
                self.counts["propose_hard_mismatch"] = (self.counts.get("propose_hard_mismatch", 0)
                                                        + int(chosen[p][0] != t and m > 0.1))
                self.counts["propose_gap_over_0p1"] = self.counts.get("propose_gap_over_0p1", 0) + int(gap > 0.1)
                self.worst["propose_logp"] = max(self.worst["propose_logp"], gap)
                if gap > 0.3:
                    self.keep("propose", gap, full=full.clone(), position=p + offset,
                              cached_token=chosen[p][0], cached_logp=chosen[p][1], ref_token=t, ref_logp=lp)
        self.counts["propose_calls"] += 1
        return result

    # -- verification ----------------------------------------------------
    def verify(self, verifier, base, positions, tokens, *, tau_nats, group_rows, veto):
        result = self.orig[1](verifier, base, positions, tokens, tau_nats=tau_nats,
                              group_rows=group_rows, veto=veto)
        if not isinstance(verifier, cross.BlockCacheProvider):
            return result
        self.counts["block_mode"] += 1
        full, offset = self.full(base)
        count = result.rows
        pos = positions[:count].cpu() + offset
        probes = full.repeat(count, 1)
        for row in range(1, count):
            probes[row, pos[:row]] = tokens[:row].cpu()
        ref = self.reference(self.verifier, probes, pos)
        conditional = ref.gather(1, tokens[:count].cpu().view(-1, 1)).squeeze(1)
        marginal = self.single(self.verifier, full, pos).gather(1, tokens[:count].cpu().view(-1, 1)).squeeze(1)
        gaps = (conditional - result.conditional.float().cpu()).abs()
        self.worst["verify_conditional"] = max(self.worst["verify_conditional"], float(gaps.max()))
        if float(gaps.max()) > 0.3:
            j = int(gaps.argmax())
            self.keep("verify", float(gaps.max()), full=probes[j].clone(), position=int(pos[j]),
                      token=int(tokens[j]), cached_logp=float(result.conditional[j]), ref_logp=float(conditional[j]))
        self.worst["verify_marginal"] = max(self.worst["verify_marginal"],
                                            float((marginal - result.marginal.float().cpu()).abs().max()))
        errors = (conditional.cumsum(0) - marginal.cumsum(0)).abs()
        passing = errors <= tau_nats
        passing[0] = True
        accepted = int(torch.cumprod(passing.long(), 0).sum())
        # The original stops scoring at the first failing group, so compare
        # only when the reference also accepts inside the scored rows.
        flipped = min(accepted, count) != min(result.accepted, count)
        self.counts["accept_mismatch"] += int(flipped)
        if flipped:
            # How far the deciding error sits from the threshold, in nats: bf16
            # noise can only flip decisions that are this close to tau.
            j = min(accepted, result.accepted, count - 1)
            near = abs(float(errors[j]) - tau_nats)
            self.counts["accept_flip_max_distance_to_tau"] = max(
                self.counts.get("accept_flip_max_distance_to_tau", 0.0), round(near, 4))
        self.counts["verify_calls"] += 1
        return result

    # -- remask ----------------------------------------------------------
    def remask(self, verifier, base, canvas_indices, positions, tokens, *, bottom, group_rows, stop_ids=frozenset()):
        result = self.orig[2](verifier, base, canvas_indices, positions, tokens, bottom=bottom,
                              group_rows=group_rows, stop_ids=stop_ids)
        if not isinstance(verifier, cross.BlockCacheProvider):
            return result
        full, offset = self.full(base)
        pos = positions.cpu() + offset
        probes = full.repeat(len(pos), 1)
        probes[torch.arange(len(pos)), pos] = self.verifier.mask_token_id
        ref = self.reference(self.verifier, probes, pos)
        toks = tokens.cpu()
        logp = ref.gather(1, toks.view(-1, 1)).squeeze(1)
        stops = torch.tensor(sorted(stop_ids), dtype=toks.dtype)
        if stops.numel():
            logp = torch.where(torch.isin(toks, stops), torch.logsumexp(ref[:, stops], 1), logp)
        argmax = ref.argmax(-1)
        self.worst["remask_logp"] = max(self.worst["remask_logp"],
                                        float((logp - torch.tensor(result.log_probs)).abs().max()))
        self.counts["remask_argmax_mismatch"] += int((argmax != torch.tensor(result.argmax)).sum())
        order = torch.argsort(logp, stable=True)[: min(bottom, len(pos))]
        ref_bottom = [int(canvas_indices[i]) for i in order.tolist()]
        self.counts["remask_choice_mismatch"] += int(sorted(ref_bottom) != sorted(result.bottom))
        self.counts["remask_passes"] += 1
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roles", choices=("1p7b_proposes", "8b_proposes"), required=True)
    parser.add_argument("--deltas", nargs="+", default=["0.10", "1.00"])
    args = parser.parse_args()
    pilot = PROJECT / "outputs" / ("cross8b_pilot" if args.roles == "1p7b_proposes" else "cross8b_swap_pilot")
    torch.backends.cuda.matmul.allow_tf32 = False
    proposer, verifier = load_models(args.roles)
    recorder = Recorder(proposer, verifier)

    tasks = []
    for bench in ("gsm8k", "mbpp_plus", "humaneval_plus"):
        rows = [json.loads(line) for line in (DATA / bench / "test.jsonl").open()]
        sample = select_sample(rows, 100, 271828)
        pilot_ids = select_sample(rows, 3, 271828)
        if bench == "humaneval_plus":
            longest = max(sample, key=lambda r: len(proposer.encode_prompt(proposer.render_prompt(r["prompt"]))[0]))
            tasks += [(bench, longest), (bench, pilot_ids[1])]
        else:
            tasks.append((bench, pilot_ids[-1] if bench == "gsm8k" else sample[0]))

    report = []
    for bench, row in tasks:
        for delta in args.deltas:
            config = parse_method(f"xadaptive-remask:{delta}", CANVAS[bench], 4, max_candidates=64,
                                  sampling=SAMPLING)
            eid = str(row["example_id"])
            prompt = proposer.encode_prompt(proposer.render_prompt(row["prompt"]))
            vprompt = verifier.encode_prompt(verifier.render_prompt(row["prompt"]))
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
            recorder.reset(f"{bench}/{eid}/δ{delta}")
            torch.manual_seed(task_seed(271828, config.label, eid))
            result = cross.generate_cross(proposer, verifier, prompt, vprompt, config)
            text = proposer.decode(result.continuation_token_ids)
            record_path = pilot / bench / config.label / "records.jsonl"
            reproduced = None
            if record_path.exists():
                for line in record_path.open():
                    rec = json.loads(line)
                    if rec["example_id"] == eid:
                        reproduced = rec["raw_completion"] == text
            peaks = [round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())]
            entry = {"benchmark": bench, "task": eid, "delta": delta, "prompt_tokens": int(prompt.shape[1]),
                     "edits": result.extras["tokens_changed_by_remask"], "reproduces_pilot": reproduced,
                     "peak_gib_per_card": peaks, "prompts_equal": bool(torch.equal(prompt, vprompt))}
            report.append(entry)
            print(json.dumps(entry), flush=True)
    print("worst |Δ log p| vs uncached reference:", json.dumps({k: round(v, 4) for k, v in recorder.worst.items()}))
    print("counts:", json.dumps(recorder.counts))
    dump = PROJECT / "logs" / f"check_cross8b_worst_{args.roles}.pt"
    torch.save(recorder.worst_cases, dump)
    for kind, cases in recorder.worst_cases.items():
        for c in cases:
            print(f"worst {kind}: gap={c['gap']:.3f} task={c['task']} pos={c['position']} "
                  f"cached={c['cached_logp']:.3f} ref={c['ref_logp']:.3f}")


if __name__ == "__main__":
    main()
