#!/usr/bin/env python3
"""Three decoded examples per benchmark, for adaptive delta and autospec.

Selection rule, fixed in advance so nothing is cherry-picked: the first three
tasks in the order the benchmark file was run, and the same three task ids for
both schedulers, so the two completions on a page answer the same question.

Model output is typeset with fancyvrb, so nothing in it needs LaTeX escaping.
"""
from __future__ import annotations

from common import AUTOSPEC, BENCH_LABEL, BENCHMARKS, REPORT, records

ADAPTIVE_METHOD = "adaptive_delta_0p50"
PROMPT_CHARS = 420
COMPLETION_CHARS = 760
N_EXAMPLES = 3


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[... truncated for the report ...]"


def verbatim(text: str) -> str:
    # A literal \end{Verbatim} inside model output would close the block early.
    safe = text.replace(r"\end{Verbatim}", r"\end{Verbatim(escaped)}")
    return (
        "\\begin{Verbatim}[breaklines=true,breakanywhere=true,"
        "fontsize=\\scriptsize,frame=leftline,framesep=4pt,rulecolor=\\color{rulegrey}]\n"
        + safe
        + "\n\\end{Verbatim}"
    )


def verdict(rec: dict) -> str:
    ok = "correct" if rec.get("score") else "incorrect"
    fail = rec.get("failure_type")
    extra = ", " + str(fail).replace("_", r"\_") if fail else ""
    return (
        f"{ok}{extra}; {rec['output_token_count']} output tokens, "
        f"{rec['denoising_steps']} sequential rounds, {rec['wall_time_seconds']:.1f}\\,s"
    )


def main() -> None:
    out: list[str] = []
    for bench in BENCHMARKS:
        A = {r["example_id"]: r for r in records("3_adaptive_delta", bench, ADAPTIVE_METHOD)}
        S = {r["example_id"]: r for r in records("4_autospec", bench, AUTOSPEC)}
        order = [r["example_id"] for r in records("3_adaptive_delta", bench, ADAPTIVE_METHOD)]
        shared = [e for e in order if e in S][:N_EXAMPLES]

        out.append(f"\\subsection{{{BENCH_LABEL[bench]}}}")
        for eid in shared:
            a, s = A[eid], S[eid]
            safe_id = eid.replace("_", r"\_")
            out.append(f"\\paragraph{{\\texttt{{{safe_id}}}}}\\mbox{{}}")
            out.append("\\emph{Prompt.}")
            out.append(verbatim(clip(a["raw_prompt"], PROMPT_CHARS)))
            out.append(f"\\emph{{Adaptive $\\delta$=0.5}} --- {verdict(a)}.")
            out.append(verbatim(clip(a["raw_completion"], COMPLETION_CHARS)))
            out.append(f"\\emph{{Autospec $\\rho$=auto}} --- {verdict(s)}.")
            out.append(verbatim(clip(s["raw_completion"], COMPLETION_CHARS)))
        out.append("")

    path = REPORT / "examples.tex"
    path.write_text("\n\n".join(out), encoding="utf-8")
    print(f"wrote {path} ({len(out)} blocks)")


if __name__ == "__main__":
    main()
