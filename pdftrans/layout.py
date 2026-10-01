"""Find the paragraphs on a PDF page and write translations back in their place."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

MATH_FONT_RE = re.compile(r"CMMI|CMSY|CMEX|CMBSY|MSBM|MSAM|Math|Symbol|STIX|rsfs|eufm|esint|wasy", re.I)
SANS_FONT_RE = re.compile(r"Sans|Arial|Helvetica|Calibri|Verdana|Tahoma|Segoe|Gothic|Hei", re.I)
SKIP_RE = re.compile(
    r"(https?://|www\.)\S+|[\w.+-]+@[\w-]+\.[\w.]+|doi:\s*\S+|arXiv:\S+", re.I
)
CJK_TARGETS = ("zh", "ja", "ko")
# Bullets and enumerations that start a list item: "• ", "- ", "1. ", "(a) ", "iv) "
LIST_MARKER_RE = re.compile(r"^\s*([•·▪‣◦●○■□►▸➢✓\-–—*]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,4}[.)])\s+(?=\S)")


def is_cjk(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0x3040 <= o <= 0x30FF
        or 0xAC00 <= o <= 0xD7AF
        or 0xF900 <= o <= 0xFAFF
        or 0x3000 <= o <= 0x303F
        or 0xFF00 <= o <= 0xFFEF
    )


@dataclass
class TextUnit:
    """One paragraph (or heading, caption, table cell) that is translated as a whole."""

    page: int
    bbox: tuple[float, float, float, float]
    text: str
    size: float
    color: int
    bold: bool
    sans: bool
    line_rects: list[tuple[float, float, float, float]] = field(default_factory=list)
    line_height: float = 1.25  # baseline distance / font size
    marker: str = ""  # list bullet / number kept out of the translated text
    math_ratio: float = 0.0
    centered: bool = False
    translation: str | None = None

    @property
    def rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.bbox)


def join_lines(lines: list[str]) -> str:
    """Join PDF lines into running text, undoing end-of-line hyphenation."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line
        elif is_cjk(out[-1]) or is_cjk(line[0]):
            out += line
        else:
            out += " " + line
    return re.sub(r"\s+", " ", out).strip()


def _line_info(line: dict) -> dict | None:
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return None
    chars = sum(len(s["text"].strip()) for s in spans)
    math_chars = sum(len(s["text"].strip()) for s in spans if MATH_FONT_RE.search(s["font"]))
    main = max(spans, key=lambda s: len(s["text"].strip()))
    bold_chars = sum(
        len(s["text"].strip()) for s in spans if s["flags"] & 16 or re.search(r"Bold|Black|Heavy|Semibold", s["font"], re.I)
    )
    return {
        "text": "".join(s["text"] for s in line["spans"]),
        "bbox": tuple(line["bbox"]),
        "size": main["size"],
        "color": main["color"],
        "font": main["font"],
        "chars": chars,
        "math_chars": math_chars,
        "bold_chars": bold_chars,
        "baseline": main["origin"][1],
        "dir": line.get("dir", (1, 0)),
    }


def _same_row(a: tuple, b: tuple) -> bool:
    """Lines that share a vertical band sit next to each other (table cells, columns)."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap > 0.5 * min(a[3] - a[1], b[3] - b[1])


SENTENCE_END = tuple(".!?:;。！？：；…")


def _continues(group: list[dict], info: dict) -> bool:
    """Whether a line continues the paragraph built so far."""
    prev = group[-1]
    size = prev["size"]
    px0, py0, px1, py1 = prev["bbox"]
    x0, y0, x1, y1 = info["bbox"]
    gx0 = min(g["bbox"][0] for g in group)
    gx1 = max(g["bbox"][2] for g in group)
    if abs(info["size"] - size) > max(0.5, 0.08 * size):
        return False  # different font size: heading vs body, caption vs body
    if (prev["bold_chars"] > 0.6 * prev["chars"]) != (info["bold_chars"] > 0.6 * info["chars"]):
        return False  # bold heading followed by regular text
    if y0 < py0 + 0.3 * size or _same_row(prev["bbox"], info["bbox"]):
        return False  # not below the previous line: next column, table cell, ...
    if y0 - py1 > 0.8 * size:
        return False  # vertical gap between paragraphs
    if LIST_MARKER_RE.match(info["text"]):
        return False  # next list item
    if min(x1, gx1) - max(x0, gx0) < 0.3 * min(x1 - x0, gx1 - gx0):
        return False  # no horizontal overlap: different column
    centered = abs((x0 + x1) - (px0 + px1)) < size
    if not centered:
        if len(group) >= 2 and x0 > gx0 + 0.8 * size:
            return False  # first-line indent of a new paragraph
        ended_early = px1 < gx1 - 0.15 * (gx1 - gx0)
        if len(group) >= 2 and ended_early and prev["text"].rstrip().endswith(SENTENCE_END):
            return False  # short last line of the previous paragraph
    return True


def _split_paragraphs(infos: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for info in infos:
        if groups and _continues(groups[-1], info):
            groups[-1].append(info)
        else:
            groups.append([info])
    return groups


def extract_units(page: pymupdf.Page, page_no: int) -> list[TextUnit]:
    """Return the text units of a page in reading order."""
    data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_PRESERVE_WHITESPACE)
    page_w = page.rect.width
    units: list[TextUnit] = []
    # Group lines ourselves instead of trusting MuPDF's blocks, which sometimes split a
    # paragraph (a wrapped title) or glue unrelated lines together.
    infos = [
        i
        for block in data["blocks"]
        if block.get("type") == 0
        for i in (_line_info(line) for line in block["lines"])
        if i
    ]
    # Rotated or vertical text is left alone: re-flowing it horizontally would break the page.
    infos = [i for i in infos if abs(i["dir"][0] - 1) < 1e-3 and abs(i["dir"][1]) < 1e-3]
    for group in _split_paragraphs(infos):
        text = join_lines([g["text"] for g in group])
        marker = ""
        m = LIST_MARKER_RE.match(text)
        if m and len(text) > m.end():
            marker, text = m.group(1), text[m.end():]
        if not text:
            continue
        x0 = min(g["bbox"][0] for g in group)
        y0 = min(g["bbox"][1] for g in group)
        x1 = max(g["bbox"][2] for g in group)
        y1 = max(g["bbox"][3] for g in group)
        chars = sum(g["chars"] for g in group) or 1
        main = max(group, key=lambda g: g["chars"])
        size = main["size"]
        if len(group) > 1:
            steps = [b["baseline"] - a["baseline"] for a, b in zip(group, group[1:])]
            line_height = sum(steps) / len(steps) / size
        else:
            line_height = 1.25
        centers = [(g["bbox"][0] + g["bbox"][2]) / 2 for g in group]
        centered = all(abs(c - page_w / 2) < 0.03 * page_w for c in centers) and (
            (len(group) == 1 and x1 - x0 < 0.7 * page_w)
            or (1 < len(group) <= 3 and max(g["bbox"][0] for g in group) - x0 > size)
        )
        units.append(
            TextUnit(
                page=page_no,
                bbox=(x0, y0, x1, y1),
                text=text,
                size=size,
                color=main["color"],
                bold=sum(g["bold_chars"] for g in group) > 0.6 * chars,
                sans=bool(SANS_FONT_RE.search(main["font"])),
                line_rects=[g["bbox"] for g in group],
                marker=marker,
                line_height=min(max(line_height, 1.1), 1.8),
                math_ratio=sum(g["math_chars"] for g in group) / chars,
                centered=centered,
            )
        )
    return units


def should_translate(unit: TextUnit, target_lang: str) -> bool:
    text = unit.text.strip()
    visible = [c for c in text if not c.isspace()]
    letters = [c for c in visible if c.isalpha()]
    if len(letters) < 2:
        return False  # page numbers, bullets, lone symbols
    if unit.math_ratio > 0.4:
        return False  # display formulas
    if len(letters) < 0.4 * len(visible):
        return False  # mostly digits / operators, e.g. table numbers or inline equations
    if not SKIP_RE.sub("", text).strip(" .,;:()[]"):
        return False  # only URLs / emails / DOIs
    if target_lang.split("-")[0] in CJK_TARGETS:
        cjk = sum(1 for c in letters if is_cjk(c))
        if cjk > 0.5 * len(letters):
            return False  # already in a CJK language
    return True


def _rgb(color: int) -> tuple[float, float, float]:
    return ((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255


def _free_space_below(rect: pymupdf.Rect, obstacles: list[pymupdf.Rect], page_rect: pymupdf.Rect) -> float:
    """How far a box can grow downwards without running into anything else on the page."""
    limit = page_rect.y1 - 18
    for o in obstacles:
        if o.y0 >= rect.y1 - 0.5 and o.x0 < rect.x1 - 1 and o.x1 > rect.x0 + 1:
            limit = min(limit, o.y0)
    return max(0.0, limit - rect.y1 - 2)


def _widen(unit: TextUnit, obstacles: list[pymupdf.Rect], left_margin: float, right_margin: float) -> pymupdf.Rect:
    """Let a one-line unit (heading, caption, label) use free room beside it, since its
    box is only as wide as the original text and a translation may be longer."""
    rect = unit.rect
    if len(unit.line_rects) != 1:
        return rect
    left, right = left_margin, right_margin
    for o in obstacles:
        if o.y1 <= rect.y0 + 1 or o.y0 >= rect.y1 - 1:
            continue  # not on the same row
        if o.x0 >= rect.x1 - 1:
            right = min(right, o.x0 - unit.size * 0.5)
        elif o.x1 <= rect.x0 + 1:
            left = max(left, o.x1 + unit.size * 0.5)
    right = max(right, rect.x1)
    left = min(left, rect.x0)
    if unit.centered:
        grow = min(rect.x0 - left, right - rect.x1)
        return pymupdf.Rect(rect.x0 - grow, rect.y0, rect.x1 + grow, rect.y1)
    return pymupdf.Rect(rect.x0, rect.y0, right, rect.y1)


# Characters that must not start a line, and ones that must not end one.
NO_LINE_START = set("，。、；：！？）」』】》〉〕］｝”’,.;:!?)]}%·…—～‰")
NO_LINE_END = set("（「『【《〈〔［｛“‘([{$")
ATOM_RE = re.compile(r"\s+|[^\s]")


class Fonts:
    """The fonts used for translated text, shared by every page of a document.

    MuPDF caches embedded fonts per document, so using the same Font objects for
    every paragraph embeds each font once instead of once per paragraph.
    """

    def __init__(self, font_file: str = ""):
        self.cjk = pymupdf.Font("cjk")
        self.user = None
        if font_file:
            if not Path(font_file).is_file():
                raise FileNotFoundError(f"字体文件不存在: {font_file}")
            self.user = pymupdf.Font(fontfile=font_file)
        self.latin = {
            (False, False): pymupdf.Font("tiro"),
            (False, True): pymupdf.Font("tibo"),
            (True, False): pymupdf.Font("helv"),
            (True, True): pymupdf.Font("hebo"),
        }
        self._widths: dict[tuple[int, str], float] = {}
        self._pick: dict[tuple[bool, bool, str], pymupdf.Font] = {}

    def font_for(self, ch: str, sans: bool, bold: bool) -> pymupdf.Font:
        key = (sans, bold, ch)
        font = self._pick.get(key)
        if font is None:
            o = ord(ch)
            if self.user is not None and self.user.has_glyph(o):
                font = self.user
            elif not is_cjk(ch) and o < 0x2000 and self.latin[(sans, bold)].has_glyph(o):
                font = self.latin[(sans, bold)]
            else:
                font = self.cjk
            self._pick[key] = font
        return font

    def width(self, font: pymupdf.Font, ch: str) -> float:
        """Advance width of ch at font size 1."""
        key = (id(font), ch)
        w = self._widths.get(key)
        if w is None:
            w = font.text_length(ch, fontsize=1)
            self._widths[key] = w
        return w


def _atoms(text: str) -> list[str]:
    """Split text into unbreakable pieces: CJK characters, words, and spaces."""
    atoms: list[str] = []
    for m in ATOM_RE.finditer(text):
        tok = m.group()
        if tok.isspace():
            atoms.append(" ")
        elif atoms and not is_cjk(tok) and not atoms[-1].isspace() and not is_cjk(atoms[-1][-1]) and tok not in NO_LINE_END:
            atoms[-1] += tok  # continue a Latin word
        else:
            atoms.append(tok)
    # Keep punctuation attached to its neighbour so it never dangles at a line edge.
    merged: list[str] = []
    for a in atoms:
        if merged and a[0] in NO_LINE_START and not merged[-1].isspace():
            merged[-1] += a
        elif merged and merged[-1][-1] in NO_LINE_END and not a.isspace():
            merged[-1] += a
        else:
            merged.append(a)
    return merged


def break_lines(text: str, width: float, size: float, measure) -> list[list[str]]:
    """Greedy line breaking; measure(atom) returns the atom width at font size 1."""
    lines: list[list[str]] = [[]]
    used = 0.0
    for atom in _atoms(text):
        w = measure(atom) * size
        if atom == " ":
            if lines[-1]:
                lines[-1].append(atom)
                used += w
            continue
        if used + w > width + 0.01 and lines[-1]:
            while lines[-1] and lines[-1][-1] == " ":
                lines[-1].pop()
            lines.append([])
            used = 0.0
        if w > width and len(atom) > 1:
            # A single word wider than the column: break it by characters.
            for ch in atom:
                cw = measure(ch) * size
                if used + cw > width + 0.01 and lines[-1]:
                    lines.append([])
                    used = 0.0
                lines[-1].append(ch)
                used += cw
            continue
        lines[-1].append(atom)
        used += w
    while lines and lines[-1] and lines[-1][-1] == " ":
        lines[-1].pop()
    return [line for line in lines if line]


def typeset(page: pymupdf.Page, unit: TextUnit, rect: pymupdf.Rect, extra_height: float, fonts: Fonts) -> None:
    """Draw unit.translation inside rect, using up to extra_height more room below it
    and shrinking the font only when that is not enough."""
    text = unicodedata.normalize("NFC", unit.translation or "").strip()
    if not text:
        return

    def measure(atom: str) -> float:
        return sum(fonts.width(fonts.font_for(c, unit.sans, unit.bold), c) for c in atom)

    width = rect.width + 1
    size = unit.size
    pitch_factor = unit.line_height

    def hang_for(size: float) -> float:
        return measure(unit.marker) * size + size * 0.5 if unit.marker else 0.0

    lines = break_lines(text, width - hang_for(size), size, measure)
    min_size = unit.size * 0.45
    while True:
        needed = (len(lines) - 1) * size * pitch_factor + size * 1.05
        if needed <= rect.height + extra_height or size <= min_size:
            break
        size *= 0.94
        lines = break_lines(text, width - hang_for(size), size, measure)
    hang = hang_for(size)

    justify = len(unit.line_rects) >= 3 and not unit.centered
    color = _rgb(unit.color)
    writer = pymupdf.TextWriter(page.rect, color=color)
    bold_writer = pymupdf.TextWriter(page.rect, color=color) if unit.bold else None
    overprinted = False
    baseline = rect.y0 + size * 0.88
    if unit.marker:
        mx = rect.x0
        for ch in unit.marker:
            font = fonts.font_for(ch, unit.sans, unit.bold)
            writer.append((mx, baseline), ch, font=font, fontsize=size)
            mx += fonts.width(font, ch) * size
    for i, line in enumerate(lines):
        widths = [measure(a) * size for a in line]
        total = sum(widths)
        x = rect.x0 + hang
        gap = 0.0
        if unit.centered:
            x += max(0.0, (rect.width - hang - total) / 2)
        elif justify and i < len(lines) - 1 and len(line) > 1:
            gap = max(0.0, (width - hang - 1 - total) / (len(line) - 1))
            if gap > size * 0.6:
                gap = 0.0  # a short line before a forced break: don't stretch it apart
        for atom, w in zip(line, widths):
            if atom != " ":
                cx = x
                for ch in atom:
                    font = fonts.font_for(ch, unit.sans, unit.bold)
                    writer.append((cx, baseline), ch, font=font, fontsize=size)
                    if bold_writer is not None and font is fonts.cjk:
                        # The CJK font has no bold face; overprint slightly offset instead.
                        bold_writer.append((cx + size * 0.04, baseline), ch, font=font, fontsize=size)
                        overprinted = True
                    cx += fonts.width(font, ch) * size
            x += w + gap
        baseline += size * pitch_factor
    writer.write_text(page)
    if overprinted:
        bold_writer.write_text(page)


def write_translations(page: pymupdf.Page, units: list[TextUnit], fonts: Fonts) -> None:
    """Erase the original text of the given units and draw their translations in place."""
    units = [u for u in units if u.translation]
    if not units:
        return
    for u in units:
        for x0, y0, x1, y1 in u.line_rects:
            # Shrink vertically so neighbouring lines (and their descenders) survive.
            pad = (y1 - y0) * 0.2
            page.add_redact_annot(pymupdf.Rect(x0, y0 + pad, x1, y1 - pad), fill=False)
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        text=pymupdf.PDF_REDACT_TEXT_REMOVE,
    )
    obstacles = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")]
    obstacles += [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    obstacles += [pymupdf.Rect(d["rect"]) for d in page.get_drawings() if d["rect"].width > 2 or d["rect"].height > 2]
    obstacles += [u.rect for u in units]
    text_boxes = [u.rect for u in units] + [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")]
    left_margin = min(r.x0 for r in text_boxes)
    right_margin = max(r.x1 for r in text_boxes)
    for u in units:
        others = [o for o in obstacles if o != u.rect and not u.rect.contains(o)]
        rect = _widen(u, others, left_margin, right_margin)
        extra = min(_free_space_below(rect, others, page.rect), rect.height * 0.5 + u.size)
        typeset(page, u, rect, extra, fonts)


def side_by_side(original: pymupdf.Document, translated: pymupdf.Document, pages: list[int]) -> pymupdf.Document:
    """Each output page holds the original page on the left and its translation on the right."""
    out = pymupdf.open()
    for out_index, src_index in enumerate(pages):
        r = original[src_index].rect
        page = out.new_page(width=r.width * 2, height=r.height)
        page.show_pdf_page(pymupdf.Rect(0, 0, r.width, r.height), original, src_index)
        page.show_pdf_page(pymupdf.Rect(r.width, 0, r.width * 2, r.height), translated, out_index)
    return out


def alternating(original: pymupdf.Document, translated: pymupdf.Document, pages: list[int]) -> pymupdf.Document:
    """Original page followed by its translated page."""
    out = pymupdf.open()
    for out_index, src_index in enumerate(pages):
        out.insert_pdf(original, from_page=src_index, to_page=src_index)
        out.insert_pdf(translated, from_page=out_index, to_page=out_index)
    return out
