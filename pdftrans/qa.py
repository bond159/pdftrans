"""Quality report for a finished translation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import proofread as pr

# kind -> label shown to the user, in display order
KINDS = {
    "check": "需要检查",
    "english": "页面里的英文段落",
    "fixed": "已自动修正",
    "manual": "手动修改",
}

TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Item:
    kind: str
    page: int | None  # 1-based page in the Chinese PDF, None if not found
    source: str
    translation: str
    notes: list[str] = field(default_factory=list)
    draft: str = ""


@dataclass
class Report:
    paragraphs: int = 0
    items: list[Item] = field(default_factory=list)

    def count(self, kind: str) -> int:
        return sum(1 for i in self.items if i.kind == kind)

    def summary(self) -> str:
        parts = [f"共 {self.paragraphs} 段"]
        for kind, label in KINDS.items():
            n = self.count(kind)
            if n:
                parts.append(f"{label} {n} 处")
        return "，".join(parts)

    def save(self, path: Path) -> None:
        data = {"paragraphs": self.paragraphs, "items": [asdict(i) for i in self.items]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Report":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(paragraphs=data["paragraphs"], items=[Item(**i) for i in data["items"]])


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", TAG_RE.sub("", pr.PLACEHOLDER_RE.sub("", text)))


class PageFinder:
    """Find which page of the output PDF shows a given translation."""

    def __init__(self, pdf: Path | None):
        self.pages: list[str] = []
        if pdf is not None and Path(pdf).exists():
            import pymupdf

            with pymupdf.open(pdf) as doc:
                self.pages = [_squash(p.get_text()) for p in doc]

    def find(self, text: str) -> int | None:
        squashed = _squash(text)
        # Try a few probes: typesetting may split a paragraph across columns or pages.
        for start in (0, len(squashed) // 2):
            probe = squashed[start : start + 12]
            if len(probe) < 4:
                continue
            for i, page in enumerate(self.pages):
                if probe in page:
                    return i + 1
        return None


def english_paragraphs(pdf: Path | None, min_words: int = 12) -> list[tuple[int, str]]:
    """Text blocks in the output that are still long English passages."""
    if pdf is None or not Path(pdf).exists():
        return []
    import pymupdf

    found = []
    with pymupdf.open(pdf) as doc:
        for number, page in enumerate(doc, 1):
            for block in page.get_text("blocks"):
                text = block[4].strip()
                words = pr.ENGLISH_WORD_RE.findall(text)
                cjk = len(pr.CJK_RE.findall(text))
                if len(words) >= min_words and cjk < 0.1 * len(text) and pr.english_runs(text, 6):
                    found.append((number, re.sub(r"\s+", " ", text)))
    return found


def build_report(paragraphs, pdf: Path | None) -> Report:
    """paragraphs: translator.Paragraph records collected while translating."""
    finder = PageFinder(pdf)
    report = Report(paragraphs=len(paragraphs))
    for p in paragraphs:
        if p.overridden:
            kind, notes = "manual", ["使用了你手动修改的译文"]
        elif p.issues:
            kind, notes = "check", list(p.issues) + [f"审校：{x}" for x in p.problems]
        elif p.corrected:
            kind, notes = "fixed", list(p.problems) or ["审校修改了译文"]
        else:
            continue
        report.items.append(Item(kind, finder.find(p.final), p.source, p.final, notes, p.draft))
    translated = {_squash(p.source)[:40] for p in paragraphs}
    for page, text in english_paragraphs(pdf):
        if _squash(text)[:40] in translated:
            continue  # already reported through its paragraph record
        report.items.append(Item("english", page, text, "", ["输出页面中仍是英文（参考文献、代码、表格可忽略）"]))
    order = list(KINDS)
    report.items.sort(key=lambda i: (order.index(i.kind), i.page or 10**6))
    return report
