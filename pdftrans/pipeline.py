"""End-to-end job: read a PDF, translate its paragraphs, write the output PDFs."""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pymupdf

from .config import Settings, parse_pages
from .layout import Fonts, TextUnit, alternating, extract_units, should_translate, side_by_side, write_translations
from .llm import Translator

# progress(done, total, message)
ProgressFn = Callable[[int, int, str], None]

MODE_SUFFIX = {"mono": "mono", "dual": "dual", "alt": "alt"}


class Cancelled(Exception):
    pass


@dataclass
class JobResult:
    outputs: dict[str, Path]
    pages: list[int]
    units: int
    translated: int
    cached: int


def make_batches(texts: list[str], max_chars: int, max_items: int = 40) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for t in texts:
        if current and (size + len(t) > max_chars or len(current) >= max_items):
            batches.append(current)
            current, size = [], 0
        current.append(t)
        size += len(t)
    if current:
        batches.append(current)
    return batches


def output_paths(src: Path, out_dir: Path, lang: str, modes: list[str]) -> dict[str, Path]:
    return {m: out_dir / f"{src.stem}.{lang}.{MODE_SUFFIX[m]}.pdf" for m in modes}


def translate_pdf(
    src: str | Path,
    settings: Settings,
    translator: Translator,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> JobResult:
    src = Path(src)
    modes = [m for m in settings.modes if m in MODE_SUFFIX] or ["mono"]
    out_dir = Path(settings.output_dir) if settings.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda done, total, msg: None)

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    doc = pymupdf.open(src)
    if doc.needs_pass:
        raise ValueError("PDF 已加密，需要密码才能打开")
    pages = parse_pages(settings.pages, doc.page_count)
    if not pages:
        raise ValueError("没有要翻译的页面")
    fonts = Fonts(settings.font_file)

    # 1. Collect paragraphs.
    report(0, 1, f"解析 PDF：{len(pages)} 页")
    by_page: dict[int, list[TextUnit]] = {}
    for p in pages:
        page = doc[p]
        if page.rotation:
            page.remove_rotation()
        by_page[p] = [u for u in extract_units(page, p) if should_translate(u, settings.target_lang)]
    units = [u for p in pages for u in by_page[p]]

    # 2. Translate unique texts, reusing the cache where possible.
    unique = list(dict.fromkeys(u.text for u in units))
    results: dict[str, str] = {}
    for t in unique:
        hit = translator.cached(t)
        if hit is not None:
            results[t] = hit
    cached = len(results)
    todo = [t for t in unique if t not in results]
    batches = make_batches(todo, settings.batch_chars)
    total = len(unique)
    report(cached, total, f"共 {total} 段文本，缓存命中 {cached} 段，需翻译 {len(todo)} 段（{len(batches)} 批）")

    if batches:
        pool = ThreadPoolExecutor(max_workers=max(1, settings.concurrency))
        try:
            futures = {pool.submit(translator.translate_batch, b): b for b in batches}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=0.3, return_when=FIRST_COMPLETED)
                check_cancel()
                for fut in finished:
                    for text, translated in zip(futures[fut], fut.result()):
                        results[text] = translated
                if finished:
                    report(len(results), total, f"已翻译 {len(results)}/{total} 段")
        finally:
            # On error or cancel, drop queued batches instead of paying for them.
            pool.shutdown(wait=False, cancel_futures=True)

    check_cancel()

    # 3. Write translations into the pages.
    report(total, total, "正在排版译文…")
    for u in units:
        u.translation = results.get(u.text)
    for p in pages:
        check_cancel()
        write_translations(doc[p], by_page[p], fonts)

    # 4. Build the requested outputs.
    report(total, total, "正在生成 PDF…")
    translated = doc
    if len(pages) != doc.page_count:
        translated.select(pages)
    try:
        translated.subset_fonts()  # embed only the glyphs used; keeps output files small
    except Exception as e:  # a font MuPDF can't subset still renders fine, just larger
        report(total, total, f"字体子集化失败，输出文件会偏大：{e}")
    paths = output_paths(src, out_dir, settings.target_lang, modes)
    original = pymupdf.open(src)
    for mode, path in paths.items():
        if mode == "mono":
            out = translated
        elif mode == "dual":
            out = side_by_side(original, translated, pages)
        else:
            out = alternating(original, translated, pages)
        out.save(path, garbage=3, deflate=True)
    report(total, total, "完成")
    return JobResult(outputs=paths, pages=pages, units=len(units), translated=len(todo), cached=cached)
