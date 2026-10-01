"""Would exp05's remask pass ever pick the token that actually makes a GSM8K answer wrong?

For each wrong no-remask GSM8K completion containing an arithmetic slip `a op b = c`
(c wrong), score with Dream -- exactly as remask_pass does, one row per token with that
token masked -- every token of the SDAR block holding the first wrong digit of c.

Views:
  as_run   : canvas visible up to the end of that block (or the stop), later masked
             -- what the real remask pass saw
  full     : the whole answer visible (hindsight upper bound)
  prefix   : only tokens before the wrong digit visible (what a causal check sees)
"""
import json, re, sys, collections
import torch
sys.path.insert(0, '.'); sys.path.insert(0, 'report')
from remask_diagnosis import paired, DELTAS
from adaptive_re.model import MaskedDiffusionProvider, ModelOptions
from adaptive_re.scheduler import _forward_slices, _filtered_log_probs

OUT = sys.argv[1]
BLOCK, CANVAS, STOP = 64, 256, 151645
dream = MaskedDiffusionProvider.from_pretrained(ModelOptions(
    model_id='Dream-org/Dream-v0-Instruct-7B', family='dream', device='cuda:0',
    dtype='bfloat16', local_files_only=True, require_lean_logits=True, vocab_limit=151665))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('../models/SDAR-1.7B-Chat-b64-local')
assert tok.is_fast

NUM = r'\d[\d,]*(?:\.\d+)?'
OP = r'(?:\\times|\\cdot|\\div|[x×*/+\-])'
EXPR = re.compile(rf'((?:\$?{NUM}\s*{OP}\s*)+\$?{NUM})\s*=\s*\$?\s*({NUM})')

def evaluate(lhs):
    e = lhs.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/').replace('×', '*')
    e = e.replace('$', '').replace(',', '')
    e = re.sub(r'(?<=\d)\s*x\s*(?=\d)', '*', e)
    return eval(e)

def first_slip(text):
    for m in EXPR.finditer(text):
        try: v = evaluate(m.group(1))
        except Exception: continue
        c = float(m.group(2).replace(',', ''))
        if abs(v - c) > 0.011 and abs(v - c) > 1e-6 * abs(v):
            return m, v
    return None

def loo(base, positions, tokens):
    """remask_pass scoring: one row per position, that position masked."""
    lps, args = [], []
    for s in range(0, len(positions), 4):
        pos = torch.tensor(positions[s:s + 4], device=dream.device)
        rows = base.repeat(len(pos), 1)
        rows[torch.arange(len(pos)), pos] = dream.mask_token_id
        (logits,) = _forward_slices(dream, rows, [(torch.arange(len(pos), device=dream.device), pos)])
        lp = _filtered_log_probs(logits, dream.excluded_token_ids)
        t = torch.tensor(tokens[s:s + 4], device=dream.device)
        lps += lp.gather(1, t[:, None]).squeeze(1).tolist(); args += lp.argmax(-1).tolist()
    return lps, args

seen, results = set(), []
for d in DELTAS:
    for p in paired('gsm8k', d):
        r = p['without']
        if r['score']: continue
        text = r['raw_completion']
        if (p['id'], text) in seen: continue
        hit = first_slip(text)
        if not hit: continue
        seen.add((p['id'], text))
        m, correct = hit
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc['input_ids'], enc['offset_mapping']
        written = m.group(2)
        want = (f'{correct:.2f}'.rstrip('0').rstrip('.'))
        # first wrong digit: first char where written (commas removed) departs from correct
        w_digits = written.replace(',', '')
        k = next((j for j in range(len(w_digits)) if j >= len(want) or w_digits[j] != want[j]), len(w_digits) - 1)
        char = m.start(2) + [j for j, ch in enumerate(written) if ch != ','][k]
        i = next(t for t, (a, b) in enumerate(offs) if a <= char < b)
        canvas = ids + [STOP]
        stop_at = len(ids)
        p_s = len(tok(r['rendered_prompt'], add_special_tokens=False)['input_ids'])
        blk = (p_s + i) // BLOCK
        lo = max(blk * BLOCK - p_s, 0); hi = min((blk + 1) * BLOCK - p_s, stop_at + 1)
        dprompt = dream.encode_prompt(dream.render_prompt(r['raw_prompt'])).to(dream.device)
        off = dprompt.shape[1]
        def base(visible):
            body = [t if j < visible else dream.mask_token_id for j, t in enumerate(canvas)]
            body += [dream.mask_token_id] * (CANVAS - len(body))
            return torch.cat([dprompt, torch.tensor([body[:CANVAS]], device=dream.device)], 1)
        block_pos = list(range(lo, hi)); block_tok = [canvas[j] for j in block_pos]
        dpos = [j + off for j in block_pos]
        res = dict(delta=d, id=p['id'], expr=m.group(0), correct=want, i=i, block=blk,
                   wrong_tok=tok.decode([ids[i]]), block_len=len(block_pos),
                   later_mentions=text[m.end():].replace(',', '').count(w_digits))
        for view, visible in (('as_run', hi), ('full', len(canvas))):
            lps, args = loo(base(visible), dpos, block_tok)
            j = block_pos.index(i)
            order = sorted(range(len(lps)), key=lambda q: lps[q])
            res[view] = dict(lp=lps[j], rank=order.index(j) + 1, argmax=tok.decode([args[j]]),
                             in_bottom4=order.index(j) < 4,
                             chosen_like=[tok.decode([block_tok[q]]) for q in order[:4]],
                             bottom4_lp=[round(lps[q], 2) for q in order[:4]])
        lps, args = loo(base(i), [i + off], [ids[i]])
        res['prefix'] = dict(lp=lps[0], argmax=tok.decode([args[0]]))
        results.append(res)
        print(json.dumps(res), flush=True)
json.dump(results, open(OUT, 'w'), indent=1)
