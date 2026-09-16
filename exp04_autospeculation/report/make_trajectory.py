#!/usr/bin/env python3
"""Full decoding trajectory of one MBPP+ task under adaptive delta=1.5.

The benchmark runs used --trace-limit 0, so no step traces exist in outputs/.
This task was re-run once with --trace-limit 1 and otherwise the launcher's
exact arguments; the run is greedy and CUBLAS-deterministic, and its completion,
EOS position, step count, verifier rows and commitment histogram all match the
recorded run byte for byte (checked below, and the build fails if they drift).
The re-run's artifacts are kept in report/data/trajectory_mbpp56/.

Two views are emitted into trajectory.tex:
  1. a per-step table: which positions were committed, with which tokens, and
     why the step stopped where it did;
  2. the answer region of the canvas after every step, masks shown as a dot and
     the tokens committed at that step highlighted.
"""
from __future__ import annotations

import json

from common import REPORT, records

DATA = REPORT / "data" / "trajectory_mbpp56"
BLOCK = 64
MASK_GLYPH = "·"
EOT = "<|endoftext|>"


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(REPORT.parent / "models" / "SDAR-1.7B-Chat-b64-local"), local_files_only=True
    )


def tex_token(text: str) -> str:
    """A token as visible typewriter text: spaces and newlines made explicit."""
    out = []
    for ch in text:
        if ch == " ":
            out.append(r"\textvisiblespace{}")
        elif ch == "\n":
            out.append(r"$\hookleftarrow$")
        elif ch in "\\":
            out.append(r"\textbackslash{}")
        elif ch in "{}$&#_%":
            out.append("\\" + ch)
        elif ch == "^":
            out.append(r"\^{}")
        elif ch == "~":
            out.append(r"\~{}")
        else:
            out.append(ch)
    return r"\texttt{" + "".join(out) + "}"


def verb_escape(text: str) -> str:
    """Escape for a Verbatim block that uses \\ { } as command characters."""
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def main() -> None:
    tok = tokenizer()
    rerun = json.loads((DATA / "records.jsonl").read_text().splitlines()[0])
    traces = [json.loads(line) for line in (DATA / "decision_traces.jsonl").read_text().splitlines()]
    example_id = rerun["example_id"]

    # The re-run must be the recorded run, or the trajectory describes nothing.
    original = next(
        r for r in records("3_adaptive_delta", "mbpp_plus", "adaptive_delta_1p50")
        if r["example_id"] == example_id
    )
    for key in ("raw_completion", "output_token_count", "eos_position",
                "denoising_steps", "verifier_rows", "commitment_histogram"):
        if rerun[key] != original[key]:
            raise SystemExit(f"re-run diverges from the recorded run on {key}")

    prompt_len = len(tok(rerun["rendered_prompt"], add_special_tokens=False)["input_ids"])
    canvas_len = int(rerun["canvas_token_count"])
    eos = int(rerun["eos_position"])
    tau = traces[0]["threshold_tau_nats"]

    def canvas_pos(block: int, local: int) -> int:
        return block * BLOCK + local - prompt_len

    # ------------------------------------------------------------ step table
    rows = []
    canvas: dict[int, str] = {}
    snapshots = []
    content_total = padding_total = 0
    for t in traces:
        k = t["accepted_prefix_length"]
        ids = t["candidate_token_ids"][:k]
        committed = sorted(
            (canvas_pos(t["block_index"], p), tok.decode([i]))
            for p, i in zip(t["committed_positions"], ids)
        )
        errs = t["error_by_prefix_nats"]
        if k < len(errs):
            stop = rf"err$_{{{k + 1}}}$={errs[k]:.2f} $>$ $\tau$"
        elif k == len(t["candidate_positions"]):
            stop = "block complete" if k < BLOCK else "all 64 (cap)"
        else:
            stop = "--"

        # Compress runs of consecutive-position EOT tokens; show everything else.
        parts, run = [], []
        n_content = n_pad = 0

        def flush():
            if run:
                lo, hi = run[0], run[-1]
                span = f"{lo}" if lo == hi else f"{lo}--{hi}"
                parts.append(rf"{span}: \texttt{{<|endoftext|>}}$\times${len(run)}")
                run.clear()

        for pos, text in committed:
            if text == EOT:
                if run and pos != run[-1] + 1:
                    flush()
                run.append(pos)
                n_pad += 1
            else:
                flush()
                parts.append(f"{pos}: {tex_token(text)}")
                n_content += 1
            canvas[pos] = text
        flush()
        content_total += n_content
        padding_total += n_pad
        beyond = sum(1 for pos, _ in committed if pos >= canvas_len)
        note = rf" \emph{{({beyond} beyond the canvas end)}}" if beyond else ""
        rows.append(
            f"{t['step_index']} & {t['block_index']} & {t['remaining_masks_before']} & "
            f"{len(t['candidate_positions'])} & {k} & {n_content} & {stop} & "
            + "; ".join(parts) + note + r" \\"
        )
        if any(pos < eos + 1 for pos, _ in committed):
            snapshots.append((t["step_index"], {pos for pos, _ in committed}, dict(canvas)))

    table = [
        r"\begin{longtable}{@{}rrrrrrl p{0.365\textwidth}@{}}",
        r"\caption{Every step of the decode. \emph{masked} is the number of masked "
        r"positions left in the block before the step; \emph{cand.}\ the candidates "
        r"proposed (all remaining positions, capped at 64, in confidence order); "
        r"\emph{acc.}\ the accepted prefix; \emph{content} the accepted tokens that "
        r"are not end-of-text padding. \emph{Stop} gives the verifier error of the "
        r"first rejected prefix against $\tau=\ln(1+\delta)=" + f"{tau:.3f}" + r"$ "
        r"nats. Positions are canvas offsets; tokens are listed in position order, "
        r"with \textvisiblespace{} a space and $\hookleftarrow$ a newline.}"
        r"\label{tab:trajectory}\\",
        r"\toprule",
        r"step & block & masked & cand. & acc. & content & stop & tokens committed \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"step & block & masked & cand. & acc. & content & stop & tokens committed \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        *rows,
        r"\end{longtable}",
    ]

    # -------------------------------------------------------- canvas snapshots
    # Answer region only: position 0 up to and including the first EOT.
    blocks = []
    for step, new, state in snapshots:
        text = []
        for pos in range(0, eos + 1):
            if pos not in state:
                text.append(MASK_GLYPH)
                continue
            piece = verb_escape(state[pos])
            text.append(r"\HL{" + piece + "}" if pos in new else piece)
        body = "".join(text)
        # minipage keeps each header with its snapshot across a column break
        blocks.append(
            r"\noindent\begin{minipage}{\linewidth}\textbf{after step " + str(step) + r"}" + "\n"
            r"\begin{Verbatim}[commandchars=\\\{\},breaklines=true,breakanywhere=true,fontsize=\footnotesize,"
            r"frame=leftline,framesep=4pt,rulecolor=\color{rulegrey}]" + "\n"
            + body + "\n" + r"\end{Verbatim}" + "\n" + r"\end{minipage}\par\medskip"
        )

    steps = len(traces)
    header = (
        rf"\textbf{{Task \texttt{{{example_id}}}}}, adaptive $\delta{{=}}1.5$ "
        rf"($\tau={tau:.3f}$ nats), canvas {canvas_len}, block 64. The prompt is "
        rf"{prompt_len} tokens, so block~0 holds only its last {BLOCK - prompt_len % BLOCK} "
        rf"canvas positions. The decode takes {steps} steps and commits "
        rf"{content_total} tokens that are not end-of-text padding and "
        rf"{padding_total} that are (the canvas is filled to its end, plus "
        rf"{(-(prompt_len + canvas_len)) % BLOCK} positions past it that complete the "
        rf"last block). The answer ends at canvas position {eos}; the re-run reproduces "
        r"the recorded completion byte for byte. It scores incorrect."
    )
    out = "\n\n".join(
        [
            header,
            r"{\small" + "\n" + "\n".join(table) + "\n}",
            r"\paragraph{The answer region after each step.} Positions 0 to "
            + str(eos)
            + r", with $\cdot$ a still-masked position and tokens committed at that "
            r"step in \HL{colour}. Steps that commit only padding are omitted.",
            r"\begin{multicols}{2}" + "\n" + "\n\n".join(blocks) + "\n" + r"\end{multicols}",
        ]
    )
    (REPORT / "trajectory.tex").write_text(out, encoding="utf-8")
    print(f"wrote trajectory.tex: {steps} steps, {content_total} content + {padding_total} padding tokens, "
          f"{len(blocks)} snapshots, prompt {prompt_len}, eos {eos}")


if __name__ == "__main__":
    main()
