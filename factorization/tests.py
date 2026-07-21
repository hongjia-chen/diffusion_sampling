
#%%
import json
from pathlib import Path

path = Path(
    "../outputs/factorization/"
    "uncond-greedy-cli-smoke/"
    "steps.jsonl"
)

with path.open() as handle:
    steps = [
        json.loads(line)
        for line in handle
        if line.strip()
    ]

for expected_step, row in enumerate(steps):
    assert row["step_index"] == expected_step

    before = row["remaining_mask_count_before"]
    after = row["remaining_mask_count_after"]

    assert after == before - 1

    committed = row["committed"]

    print(
        f"step={expected_step}: "
        f"masks={before}->{after}, "
        f"canvas_position={committed['canvas_position']}, "
        f"token_id={committed['token_id']}, "
        f"token={committed['token_text_repr']!r}, "
        f"confidence={committed['probability']:.6f}"
    )

print("PASS: exactly one mask was removed per step.")
# %%
