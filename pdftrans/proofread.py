"""Automatic proofreading of translations: deterministic checks plus an LLM review pass.

Everything here is independent of BabelDOC and of the network, so it can be
unit-tested directly. The translator in ``translator.py`` wires it in.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

# Markers BabelDOC puts into the text sent for translation: formula placeholders
# ({v3}) and rich-text style tags (<style id='2'>…</style>, <b1>…</b1>).
PLACEHOLDER_RE = re.compile(
    r"\{\s*v\s*\d+\s*\}|<\s*style\s+id\s*=\s*'\s*\d+\s*'\s*>|<\s*/\s*style\s*>|<\s*/?\s*b\d+\s*>"
)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]+")
CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")

BATCH_MARKER = "## Here is the input:"
SINGLE_MARKER = "Now translate the following text:"


def normalize_placeholder(p: str) -> str:
    return re.sub(r"\s+", "", p)


def placeholders(text: str) -> Counter:
    return Counter(normalize_placeholder(p) for p in PLACEHOLDER_RE.findall(text))


def strip_placeholders(text: str) -> str:
    return PLACEHOLDER_RE.sub(" ", text)


def numbers(text: str) -> Counter:
    """Numbers outside placeholders, with thousands separators removed (1,024 == 1024)."""
    found = NUMBER_RE.findall(strip_placeholders(text))
    return Counter(re.sub(r",(?=\d{3}\b)", "", n) for n in found)


def english_runs(text: str, min_words: int = 4) -> list[str]:
    """Runs of at least min_words consecutive English words."""
    plain = strip_placeholders(text)
    runs = re.findall(r"(?:[A-Za-z][A-Za-z'’-]*[\s,;:]+){%d,}[A-Za-z][A-Za-z'’-]*" % (min_words - 1), plain)
    return [r.strip() for r in runs]


@dataclass
class Check:
    """Result of the deterministic checks for one paragraph."""

    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def broken_markup(self) -> bool:
        return any(i.startswith("占位符") for i in self.issues)


def check_translation(source: str, output: str, keep_terms: set[str] | None = None) -> Check:
    """Deterministic checks that catch the typical LLM translation failures."""
    check = Check()
    src = source.strip()
    out = (output or "").strip()
    if not out:
        check.issues.append("译文为空")
        return check

    if placeholders(src) != placeholders(out):
        check.issues.append("占位符不一致：公式或格式标记丢失、重复或被改动")

    missing = numbers(src) - numbers(out)
    if missing:
        check.issues.append("数字不一致，缺少：" + "、".join(sorted(missing)))

    src_words = ENGLISH_WORD_RE.findall(strip_placeholders(src))
    if len(src_words) >= 4:
        keep = {t.lower() for t in (keep_terms or set())}
        leftover = [
            r for r in english_runs(out)
            if r.lower() in strip_placeholders(src).lower() and r.lower() not in keep
        ]
        if leftover:
            check.issues.append("疑似未翻译的英文：" + "；".join(leftover[:3]))
        cjk = len(CJK_RE.findall(out))
        src_chars = len(re.sub(r"\s", "", strip_placeholders(src)))
        if cjk == 0:
            check.issues.append("译文中没有中文")
        elif src_chars >= 40:
            ratio = len(re.sub(r"\s", "", strip_placeholders(out))) / src_chars
            if ratio < 0.18:
                check.issues.append("译文明显过短，可能漏译")
            elif ratio > 1.3:
                check.issues.append("译文明显过长，可能有重复或多余内容")
    return check


# ---------------------------------------------------------------------------
# Prompt parsing (BabelDOC prompt formats)


@dataclass
class BatchPrompt:
    items: list[dict]  # [{"id": 0, "input": "...", ...}]
    glossary: str  # markdown glossary tables, may be empty
    context: str  # contextual hints (document / section titles), may be empty


def _section(prompt: str, heading: str) -> str:
    m = re.search(r"^## " + re.escape(heading) + r"\s*$(.*?)(?=^## |\Z)", prompt, re.M | re.S)
    return m.group(1).strip() if m else ""


def parse_batch_prompt(prompt: str) -> BatchPrompt | None:
    if not prompt or BATCH_MARKER not in prompt:
        return None
    try:
        items = json.loads(prompt.split(BATCH_MARKER, 1)[1])
    except ValueError:
        return None
    if not isinstance(items, list) or not all(isinstance(i, dict) and "id" in i for i in items):
        return None
    return BatchPrompt(
        items=items,
        glossary=_section(prompt, "Glossary Tables"),
        context=_section(prompt, "Contextual Hints for Better Translation"),
    )


def parse_single_prompt(prompt: str) -> str | None:
    """The paragraph text of BabelDOC's single-paragraph (fallback) prompt."""
    if not prompt or SINGLE_MARKER not in prompt:
        return None
    return prompt.split(SINGLE_MARKER, 1)[1].strip()


def glossary_targets(glossary_md: str) -> set[str]:
    """Target terms of a markdown glossary table; English targets are allowed to stay English."""
    terms = set()
    for line in glossary_md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] and not set(cells[0]) <= set("-: ") and cells[0] != "Source Term":
            terms.add(cells[1])
            if cells[0] == cells[1]:
                terms.add(cells[0])
    return terms


def clean_json(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^<json>|</json>$", "", text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return text


def parse_batch_output(text: str) -> dict[int, str]:
    data = json.loads(clean_json(text))
    if isinstance(data, dict):
        data = [data]
    out = {}
    for item in data:
        if isinstance(item, dict) and "id" in item:
            value = item.get("output", item.get("input"))
            if isinstance(value, str):
                out[int(item["id"])] = value
    return out


def dump_batch_output(results: dict[int, str]) -> str:
    return json.dumps([{"id": k, "output": v} for k, v in sorted(results.items())], ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLM review

REVIEW_SYSTEM = """你是资深的学术论文翻译审校专家，负责逐段核对英文原文与简体中文译文。

审校标准：
1. 准确：译文必须完整、准确地表达原文意思，不能漏译、错译、多译（增加原文没有的内容）。
2. 术语：专业术语要用该领域通行的中文译法；有术语表时必须使用术语表中的译法；同一术语前后一致。
3. 不翻译的内容保持原样：公式、变量、引用编号（如 [12]、(Smith et al., 2020)）、数字、网址、代码、人名。
4. 标记保持原样：{v1} 这类公式占位符和 <style id='1'>…</style>、<b1>…</b1> 这类格式标记必须原样保留，数量、顺序和位置与原文对应，标记内部的英文照常翻译。
5. 通顺：符合中文学术写作习惯，但不要为了润色而改动意思正确的译文。
6. 结构：每段原文是固定的一段（可能是一段话的片段），不要合并、拆分或在段落之间挪动内容。

“自动检查提示”是程序检测到的疑点，请核实：确实有问题就修正，误报则忽略。"""

REVIEW_USER = """{context}{glossary}请审校下面的译文。输出一个 JSON 数组，每项对应一个输入 id，格式：
{{"id": 编号, "verdict": "ok" 或 "fixed", "problems": ["问题简述", ...], "output": "修正后的完整译文"}}
- verdict 为 "ok" 时 problems 为空数组，output 与原译文完全相同。
- verdict 为 "fixed" 时 output 是修正后的完整译文。
- 只输出 JSON 数组，不要任何解释，不要 ``` 代码块。

待审校内容：
{payload}"""


def build_review_prompt(items: list[dict], glossary: str = "", context: str = "") -> str:
    payload = json.dumps(items, ensure_ascii=False, indent=1)
    glossary_block = f"术语表（必须遵守）：\n{glossary}\n\n" if glossary else ""
    context_block = f"上下文提示：\n{context}\n\n" if context else ""
    return REVIEW_USER.format(context=context_block, glossary=glossary_block, payload=payload)


@dataclass
class Review:
    verdict: str
    problems: list[str]
    output: str


def parse_review(text: str) -> dict[int, Review]:
    data = json.loads(clean_json(text))
    if isinstance(data, dict):
        data = [data]
    reviews = {}
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        problems = item.get("problems") or []
        if isinstance(problems, str):
            problems = [problems]
        output = item.get("output")
        reviews[int(item["id"])] = Review(
            verdict=str(item.get("verdict", "ok")).lower(),
            problems=[str(p) for p in problems if str(p).strip()],
            output=output if isinstance(output, str) else "",
        )
    return reviews


def choose(source: str, draft: str, review: Review | None, keep_terms: set[str]) -> tuple[str, list[str]]:
    """Pick the reviewer's correction unless it breaks something the draft got right.

    Returns the final text and the list of problems the reviewer reported.
    """
    if review is None or review.verdict != "fixed" or not review.output.strip():
        return draft, review.problems if review else []
    fixed_check = check_translation(source, review.output, keep_terms)
    draft_check = check_translation(source, draft, keep_terms)
    if fixed_check.broken_markup and not draft_check.broken_markup:
        return draft, review.problems + ["审校修改破坏了公式/格式标记，已保留原译文"]
    if len(fixed_check.issues) > len(draft_check.issues):
        return draft, review.problems + ["审校修改引入了新问题，已保留原译文"]
    return review.output.strip(), review.problems
