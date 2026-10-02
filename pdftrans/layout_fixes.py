"""Corrections to BabelDOC's paragraph detection.

Table-of-contents pages: BabelDOC recognises an entry only by a dot leader
("Introduction ........ 3"). Many books instead put the page number in a column
on the right with blank space in between, and the layout model returns the
whole list as one text block, so every entry ends up in a single paragraph that
is translated and typeset as running text: line breaks and the page-number
column disappear and the text overflows the page.

When most lines of a block start with a detached section number ("1.2   Title")
or end with a detached page number ("Title      12"), the block is treated as a
list of entries: each entry (with its wrapped continuation lines) becomes its own
paragraph, and each page number a paragraph of its own so it stays in its column.
"""

from __future__ import annotations

import re
import threading

def _text(chars) -> str:
    return "".join(c.char_unicode or "" for c in chars)


def _font_size(chars) -> float:
    sizes = sorted(
        (c.pdf_style.font_size if c.pdf_style and c.pdf_style.font_size else c.visual_bbox.box.y2 - c.visual_bbox.box.y)
        for c in chars
    )
    return sizes[len(sizes) // 2] if sizes else 0.0


def _words(chars) -> list[tuple[int, int]]:
    """Runs of characters separated by spaces or by visible blank space (index ranges)."""
    size = _font_size(chars)
    words: list[tuple[int, int]] = []
    start = None
    for i, c in enumerate(chars):
        blank = not (c.char_unicode or "").strip()
        if blank:
            if start is not None:
                words.append((start, i))
                start = None
            continue
        if start is not None and c.visual_bbox.box.x - chars[i - 1].visual_bbox.box.x2 > 0.3 * size:
            words.append((start, i))
            start = None
        if start is None:
            start = i
    if start is not None:
        words.append((start, len(chars)))
    return words


def _gap(chars, left: tuple[int, int], right: tuple[int, int]) -> float:
    """Blank space between two words, in font sizes."""
    size = _font_size(chars)
    if size <= 0:
        return 0.0
    return (chars[right[0]].visual_bbox.box.x - chars[left[1] - 1].visual_bbox.box.x2) / size


NUMBER_RE = re.compile(r"\d{1,4}|[ivxlcdm]{1,7}|[IVXLCDM]{1,7}")
GAP_EMS = 2.0  # blank space before a page number; word spacing is well below 1
LEAD_RE = re.compile(r"\d{1,3}(\.\d{1,3}){0,4}\.?")
LEAD_GAP_EMS = 1.0  # blank space after a section number such as "1.2.3"


def trailing_page_number(chars) -> int | None:
    """Index where a detached page number at the end of a line starts ("Title      12"), or None."""
    words = _words(chars)
    if len(words) < 2 or not NUMBER_RE.fullmatch(_text(chars[slice(*words[-1])])):
        return None
    if not re.search(r"[A-Za-z]", _text(chars[: words[-2][1]])) or _gap(chars, words[-2], words[-1]) < GAP_EMS:
        return None
    return words[-2][1]


def leading_number(chars) -> int | None:
    """Index where the title after a detached section number starts ("1.2   Title"), or None."""
    words = _words(chars)
    if len(words) < 2 or not LEAD_RE.fullmatch(_text(chars[slice(*words[0])])):
        return None
    if not re.search(r"[A-Za-z]", _text(chars[words[1][0] :])) or _gap(chars, words[0], words[1]) < LEAD_GAP_EMS:
        return None
    return words[1][0]


def _visual_lines(chars) -> list[list]:
    """Split characters into the lines they are printed on.

    BabelDOC's lines sometimes hold several printed lines of a list; a new line starts
    where the text returns to the left and moves down.
    """
    size = _font_size(chars) or 1.0
    lines: list[list] = [[]]
    prev = None
    for c in chars:
        if (c.char_unicode or "").strip():
            if prev is not None:
                b, pb = c.visual_bbox.box, prev.visual_bbox.box
                if b.x < pb.x - 0.5 * size and abs((b.y + b.y2) - (pb.y + pb.y2)) / 2 > 0.6 * size:
                    lines.append([])
            prev = c
        lines[-1].append(c)
    return [line for line in lines if line]


def split_toc_entries(finder, paragraphs: list) -> set[int]:
    """Give every table-of-contents entry its own paragraph (see module doc).

    Returns the ids of the paragraphs that make up lists of entries."""
    from babeldoc.format.pdf.document_il.il_version_1 import Box, PdfLine, PdfParagraph, PdfParagraphComposition
    from babeldoc.format.pdf.document_il.midend.paragraph_finder import generate_base58_id

    def make(template, compositions):
        p = PdfParagraph(
            box=Box(0, 0, 0, 0),
            pdf_paragraph_composition=compositions,
            unicode="",
            debug_id=generate_base58_id(),
            layout_label=template.layout_label,
            layout_id=template.layout_id,
        )
        finder.update_paragraph_data(p)
        return p

    result, entries = [], set()
    for paragraph in paragraphs:
        old = paragraph.pdf_paragraph_composition
        if any(c.pdf_line is None or not c.pdf_line.pdf_character for c in old):
            result.append(paragraph)
            continue
        lines = []
        for c in old:
            printed = _visual_lines(c.pdf_line.pdf_character)
            if len(printed) == 1:
                lines.append(c.pdf_line)
                continue
            for chars in printed:
                line = PdfLine(pdf_character=chars, render_order=c.pdf_line.render_order)
                finder.update_line_data(line)
                lines.append(line)
        comps = old if len(lines) == len(old) else [PdfParagraphComposition(pdf_line=line) for line in lines]
        leads = [leading_number(line.pdf_character) for line in lines]
        trails = [trailing_page_number(line.pdf_character) for line in lines]
        marked = sum(1 for a, b in zip(leads, trails) if a is not None or b is not None)
        if (marked < 3 or marked < 0.4 * len(lines)) and marked < len(lines):
            result.append(paragraph)  # running text, not a list of entries
            continue

        groups: list[list] = []  # each: compositions of one entry
        numbers = {}  # group index -> page-number line
        text_x = None  # where the title text of the current entry starts
        for comp, line, lead, trail in zip(comps, lines, leads, trails):
            x = line.pdf_character[0].visual_bbox.box.x
            continuation = (
                groups
                and lead is None
                and (len(groups) - 1) not in numbers
                and text_x is not None
                and abs(x - text_x) < 0.5 * _font_size(line.pdf_character)
            )
            if not continuation:
                groups.append([])
                text_x = line.pdf_character[lead].visual_bbox.box.x if lead is not None else None
            groups[-1].append(comp)
            if trail is not None:
                chars = line.pdf_character
                number = PdfLine(pdf_character=[c for c in chars[trail:] if (c.char_unicode or "").strip()])
                finder.update_line_data(number)
                line.pdf_character = chars[:trail]
                finder.update_line_data(line)
                numbers[len(groups) - 1] = number
        for k, group in enumerate(groups):
            if k == 0:
                paragraph.pdf_paragraph_composition = group
                finder.update_paragraph_data(paragraph)
                result.append(paragraph)
            else:
                result.append(make(paragraph, group))
            if k in numbers:
                result.append(make(paragraph, [PdfParagraphComposition(pdf_line=numbers[k])]))
        entries.update(id(p) for p in result[-(len(groups) + len(numbers)) :])
    paragraphs[:] = result
    return entries


def uniform_line_box(paragraph) -> None:
    """Give a one-line entry the full height of its font instead of the height of its ink.

    An entry in capitals ("SMTP") has a shorter ink box than one with ascenders and
    descenders; the translation is placed from the top of the box, so it would sit
    lower than its neighbours and touch the next line.
    """
    comps = paragraph.pdf_paragraph_composition
    if paragraph.box is None or len(comps) != 1 or comps[0].pdf_line is None:
        return
    chars = [c for c in comps[0].pdf_line.pdf_character if c.box is not None and (c.char_unicode or "").strip()]
    if chars:
        paragraph.box.y = min(paragraph.box.y, min(c.box.y for c in chars))
        paragraph.box.y2 = max(paragraph.box.y2, max(c.box.y2 for c in chars))


_installed = False
_page = threading.local()  # pages are processed in parallel threads


def install() -> None:
    """Apply the corrections to BabelDOC (once per process)."""
    global _installed
    if _installed:
        return
    from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder

    independent = ParagraphFinder.process_independent_paragraphs
    merge_numbers = ParagraphFinder.merge_alternating_line_number_paragraphs

    def process_independent_paragraphs(self, paragraphs, median_width):
        _page.entries = split_toc_entries(self, paragraphs)
        return independent(self, paragraphs, median_width)

    def merge_alternating_line_number_paragraphs(self, paragraphs):
        # BabelDOC joins "text, line number, text" into one paragraph (for documents with
        # numbered lines). Entry, page number, entry in a contents list looks the same, so
        # that merging is applied only to the paragraphs between such lists.
        entries = getattr(_page, "entries", None) or set()
        if not entries:
            return merge_numbers(self, paragraphs)
        result, run = [], []
        for p in paragraphs + [None]:
            if p is None or id(p) in entries:
                merge_numbers(self, run)
                result.extend(run)
                run = []
                if p is not None:
                    result.append(p)
            else:
                run.append(p)
        paragraphs[:] = result

    fix_overlaps = ParagraphFinder.fix_overlapping_paragraphs

    def fix_overlapping_paragraphs(self, page):
        # Runs after BabelDOC recomputed the paragraph boxes from the ink of the glyphs.
        entries = getattr(_page, "entries", None) or set()
        _page.entries = set()
        for p in page.pdf_paragraph:
            if id(p) in entries:
                uniform_line_box(p)
        return fix_overlaps(self, page)

    ParagraphFinder.process_independent_paragraphs = process_independent_paragraphs
    ParagraphFinder.fix_overlapping_paragraphs = fix_overlapping_paragraphs
    ParagraphFinder.merge_alternating_line_number_paragraphs = merge_alternating_line_number_paragraphs
    _installed = True
