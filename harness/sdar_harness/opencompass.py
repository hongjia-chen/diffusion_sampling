"""OpenCompass, as SDAR's eval config uses it: prompts, answer extraction, graders.

Ported verbatim (quirks included) from the OpenCompass copy vendored in
JetAstra/SDAR, `evaluation/opencompass/opencompass/` (Apache-2.0), so that a
completion is graded exactly as SDAR graded its published numbers:

    datasets/math.py      last_boxed_only_string, remove_boxed, extract_boxed_answer,
                          normalize_final_answer, math_postprocess_v2,
                          MATHEvaluator(version="v2").is_equiv
    datasets/gsm8k.py     gsm8k_dataset_postprocess
    datasets/mbpp.py      MBPPEvaluator._process_answer
    datasets/humaneval.py humaneval_postprocess_v2
    evaluator/math_evaluator.py   MATHVerifyEvaluator (math_verify call)

and the four dataset configs `eval_sdar_hf.py` imports (PROMPTS below).
"""

from __future__ import annotations

import re

# ----------------------------------------------------------------------
# Prompts: the single HUMAN turn of each config, rendered through the chat
# template (BD3withChatTemplate: apply_chat_template(add_generation_prompt=True)).
# ----------------------------------------------------------------------

PROMPTS = {
    # gsm8k_0shot_v2_gen_17d799
    "gsm8k": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    # math_prm800k_500_0shot_cot_gen_11c4b5
    "math_500": "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    # humaneval_openai_sample_evals_gen_dcae0e
    "humaneval": (
        "Read the following function signature and docstring, and fully implement the "
        "function described. Your response should only contain the code for this function.\n{prompt}"
    ),
    # sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416 (its stray space before "You" kept)
    "mbpp_sanitized": (
        "You are an expert Python programmer, and here is your task:\n{text}\n"
        "Your code should pass these tests:\n\n{test_list}\n"
        " You should submit your final solution in the following format: ```python\n\n```"
    ),
}


# ----------------------------------------------------------------------
# datasets/math.py
# ----------------------------------------------------------------------

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\\{(.*)\\}$', answer)
        if match:
            answer = match.group(1)
    return answer


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                     (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                     (',\\text{and}', ','), ('\\text{and}', ','),
                     ('\\text{m}', '\\text{}'), ('\\le', '<')]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f'
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    assert '\n' not in final_answer
    assert '\r' not in final_answer
    assert '\f' not in final_answer
    if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
        final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

    if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
        final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]

    if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
        final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

    if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
        final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
    final_answer = final_answer.strip()
    if 'rac' in final_answer and '\\frac' not in final_answer:
        final_answer = final_answer.replace('rac', '\\frac')

    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def math_postprocess_v2(text: str) -> str:
    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans

    for maybe_ans in text.split('.'):
        if re.search('final answer|answer is', maybe_ans.lower()):
            return normalize_final_answer(maybe_ans)
    return normalize_final_answer(text.split('.')[0])


# MATHEvaluator(version="v2"): its helpers, as module functions.

def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except AssertionError:
        return string


def _fix_sqrt_v2(string):
    _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
    return _string


def _strip_string_v2(string):
    string = str(string).strip()
    string = string.replace('\n', '')
    string = string.rstrip('.')
    string = string.replace('\\!', '')
    string = string.replace('\\ ', '')
    string = string.replace('\\\\', '\\')
    string = string.replace('\\\\', '\\')
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    _string = re.sub(r'\\text{.*?}$', '', string).strip()
    if _string != '' and _string != string:
        string = _string
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')
    string = string.replace('\\$', '')
    string = string.replace('$', '')
    string = string.replace('\\text', '')
    string = string.replace('x\\in', '')
    string = string.replace('\\%', '')
    string = string.replace('\\%', '')
    string = string.replace('%', '')
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    string = string.replace('\\cdot', '')
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')
    string = string.replace('and', '')
    string = string.replace('\\mathbf', '')
    string = re.sub(r'\\mbox{.*?}', '', string)
    string.replace("'", '')   # no-op upstream too
    string.replace('"', '')
    if 'j' in string and 'i' not in string:
        string = string.replace('j', 'i')
    string = re.sub(r'(\d+)\.0+([^\d])', r'\1\2', string)
    string = re.sub(r'(\d+)\.0+$', r'\1', string)
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]
    string = _fix_sqrt_v2(string)
    string = string.replace(' ', '')
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def math_equiv_v2(str1, str2) -> bool:
    """`MATHEvaluator(version='v2').is_equiv`."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string_v2(str1)
        ss2 = _strip_string_v2(str2)
        if ss1 == ss2:
            return True
        ss1 = normalize_final_answer(ss1)
        ss2 = normalize_final_answer(ss2)
        if ss1 == ss2:
            return True
    except Exception:
        pass
    try:
        ss1 = normalize_final_answer(str1)
        ss2 = normalize_final_answer(str2)
        if ss1 == ss2:
            return True
    except Exception:
        pass
    return str1 == str2


# ----------------------------------------------------------------------
# datasets/gsm8k.py
# ----------------------------------------------------------------------

def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')


# ----------------------------------------------------------------------
# evaluator/math_evaluator.py (MATHVerifyEvaluator)
# ----------------------------------------------------------------------

def math_verify_agree(text: str, target: str) -> tuple[bool, str]:
    """`MATHVerifyEvaluator`, call for call: gold parsed as `$target$`, the
    completion parsed whole (boxed first, no anchor-free fallback), and the
    two passed to `verify` in OpenCompass's order. A grader error is wrong."""
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    try:
        gold = parse(
            f"${target}$", extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
        if not gold:
            return False, ""
        answer = parse(
            text,
            extraction_config=[
                LatexExtractionConfig(
                    # OpenCompass also passes equations=True, a no-op since
                    # latex2sympy2_extended 1.x that logs a warning per call.
                    normalization_config=NormalizationConfig(
                        nits=False, malformed_operators=False, basic_latex=True,
                        boxed="all", units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        return bool(verify(answer, gold)), str(answer)
    except Exception:
        return False, ""


# ----------------------------------------------------------------------
# datasets/mbpp.py, datasets/humaneval.py: code extraction
# ----------------------------------------------------------------------

_MBPP_PATTERNS = (
    r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]", r"BEGIN\s*'(.*)'\s*\[DONE\]",
    r"\[BEGIN\]\s*'(.*)'\s*DONE", r"BEGIN\s*'(.*)'\s*DONE",
    r"\[BEGIN\]\s*'(.*)\s*\[DONE\]", r"BEGIN\s*'(.*)\s*\[DONE\]",
    r"\[BEGIN\]\s*'(.*)\s*DONE", r"BEGIN\s*'(.*)\s*DONE",
    r"\[BEGIN\]\s*(.*)\s*\[DONE\]", r"BEGIN\s*(.*)\s*\[DONE\]",
    r"\[BEGIN\]\s*(.*)\s*DONE", r"BEGIN\s*(.*)\s*DONE",
    r"```python\s*(.*)\s*```", r"```\s*(.*)\s*```",
    r"```python\s*(.*)\s*$", r"```\s*(.*)\s*$",
    r"(.*)\s*```.*", r"\[BEGIN\]\s*'(.*)", r"\[BEGIN\](.*)", r"'(.*)'\s*\[DONE\]",
)


def mbpp_answer(text: str) -> str:
    """`MBPPEvaluator._process_answer`."""
    for pattern in _MBPP_PATTERNS:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
            break
    text = text.split("```")[0]
    text = re.split(r"'?\s*\[?DONE\]?", text)[0]
    text = text.replace("\\_", "_")
    return text.strip()


def humaneval_answer(text: str) -> str:
    """`humaneval_postprocess_v2`: the first fenced block, else the text."""
    blocks = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
    if blocks:
        text = blocks[0]
    return text.lstrip()
