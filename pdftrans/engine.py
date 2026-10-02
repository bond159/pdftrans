"""Run one translation job: BabelDOC does layout and typesetting, our translator proofreads."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import layout_fixes
from .config import Settings, output_paths
from .qa import Report, build_report
from .store import Store
from .translator import Journal, make_translator, role_prompt

logger = logging.getLogger(__name__)

# progress(percent 0-100, message)
ProgressFn = Callable[[float, str], None]

STAGE_NAMES = {
    "Parse PDF and Create Intermediate Representation": "解析 PDF",
    "DetectScannedFile": "检测扫描页",
    "Parse Page Layout": "识别版面（正文/图/表/公式）",
    "Parse Paragraphs": "识别段落",
    "Parse Formulas and Styles": "识别公式和样式",
    "Parse Table": "识别表格",
    "Remove Char Descent": "整理字符",
    "Automatic Term Extraction": "提取术语",
    "Translate Paragraphs": "翻译并校对",
    "Typesetting": "排版译文",
    "Add Fonts": "嵌入字体",
    "Generate drawing instructions": "生成页面",
    "Subset font": "精简字体",
    "Save PDF": "保存 PDF",
}

_layout_model = None
_layout_lock = threading.Lock()


class Cancelled(Exception):
    pass


@dataclass
class JobResult:
    mono: Path | None
    dual: Path | None
    report: Report
    glossary: Path | None = None
    seconds: float = 0.0
    tokens: int = 0
    extra: dict = field(default_factory=dict)


def bundled_assets() -> Path | None:
    """Offline asset package shipped next to the app (see packaging/), if any."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    found = sorted((base / "assets").glob("offline_assets_*.zip")) if (base / "assets").is_dir() else []
    return found[-1] if found else None


def prepare_assets(progress: ProgressFn) -> None:
    """Make sure BabelDOC's layout model and fonts are on disk.

    Restores them from the package bundled with the app when there is one;
    otherwise BabelDOC downloads them (once) from its mirrors.
    """
    from babeldoc.assets import assets

    package = bundled_assets()
    if package is not None:
        try:
            assets.restore_offline_assets_package(package)
            return
        except SystemExit:  # BabelDOC exits on a mismatched package; fall back to downloading
            logger.warning("bundled asset package %s does not match this BabelDOC version", package)
    progress(0, "首次运行：正在下载版面识别模型和字体（约 150 MB，只需一次）…")
    assets.warmup()


def layout_model():
    global _layout_model
    with _layout_lock:
        if _layout_model is None:
            from babeldoc.docvision.base_doclayout import DocLayoutModel

            _layout_model = DocLayoutModel.load_onnx()
        return _layout_model


def _load_glossaries(settings: Settings):
    from babeldoc.glossary import Glossary

    if not settings.glossary_file:
        return []
    path = Path(settings.glossary_file)
    if not path.is_file():
        raise FileNotFoundError(f"术语表文件不存在：{path}")
    return [Glossary.from_csv(path, "zh-CN")]


def _rename(path: Path | None, target: Path) -> Path | None:
    if path is None or not Path(path).exists():
        return None
    target.unlink(missing_ok=True)
    shutil.move(str(path), target)
    return target


# Batch sizes in pages: small at first so translated pages show up quickly, then larger
# because every batch has a fixed cost (parsing, embedding fonts, saving).
BATCH_SIZES = (3, 10, 20)
BATCH_PAGES = 30

# on_batch(original page indexes (0-based), PDF holding exactly those pages translated)
BatchFn = Callable[[list[int], Path], None]


def page_batches(pages: list[int]) -> list[list[int]]:
    batches, i = [], 0
    while i < len(pages):
        size = BATCH_SIZES[len(batches)] if len(batches) < len(BATCH_SIZES) else BATCH_PAGES
        batches.append(pages[i : i + size])
        i += size
    return batches


def page_spec(pages: list[int]) -> str:
    """0-based page indexes -> BabelDOC's 1-based spec, e.g. [0, 1, 2, 5] -> "1-3,6"."""
    parts, start, prev = [], None, None
    for p in sorted(pages):
        if start is None:
            start = prev = p
        elif p == prev + 1:
            prev = p
        else:
            parts.append(f"{start + 1}-{prev + 1}" if prev > start else f"{start + 1}")
            start = prev = p
    if start is not None:
        parts.append(f"{start + 1}-{prev + 1}" if prev > start else f"{start + 1}")
    return ",".join(parts)


def merge_pdfs(parts: list[Path], target: Path) -> Path:
    import pymupdf

    out = pymupdf.open()
    for part in parts:
        with pymupdf.open(part) as doc:
            out.insert_pdf(doc)
    tmp = target.with_name(target.name + ".tmp")
    out.save(tmp, garbage=3, deflate=True)
    out.close()
    target.unlink(missing_ok=True)
    shutil.move(str(tmp), target)
    return target


def merge_glossary(into: Path, new: Path | None) -> None:
    """Add terms from BabelDOC's auto-extracted glossary of one batch to the running glossary."""
    if new is None or not Path(new).exists():
        return
    rows: dict[str, str] = {}
    for path in (into, Path(new)):
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    src, tgt = (row.get("source") or "").strip(), (row.get("target") or "").strip()
                    if src and tgt and src not in rows:  # first translation of a term wins
                        rows[src] = tgt
    with into.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(sorted(rows.items()))


def job_signature(src: Path, st: Settings) -> str:
    """Everything that changes the translated pages; finished batches are reused only if it matches."""
    stat = src.stat()
    glossary = Path(st.glossary_file) if st.glossary_file else None
    parts = [
        src.name, stat.st_size, int(stat.st_mtime), st.engine, st.base_url, st.model, st.review_model,
        st.proofread, st.extra_prompt, st.auto_glossary, st.translate_tables, st.ocr_workaround,
        st.font_family, st.dual, st.proxy if st.engine != "llm" else "",
        glossary.stat().st_mtime if glossary and glossary.exists() else "",
    ]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, default=str).encode()).hexdigest()


class Job:
    """One PDF translation. Call run() from a worker thread; cancel() from anywhere.

    Pages are translated in batches. Each finished batch is reported through
    on_batch so it can be shown right away, and kept in a work folder so an
    interrupted job resumes where it stopped.
    """

    def __init__(
        self,
        src: str | Path,
        settings: Settings,
        progress: ProgressFn | None = None,
        store: Store | None = None,
        layout=None,
        skip_assets: bool = False,
        on_batch: BatchFn | None = None,
    ):
        self.src = Path(src)
        self.settings = settings
        self.progress = progress or (lambda pct, msg: None)
        self.on_batch = on_batch or (lambda pages, path: None)
        self.store = store if store is not None else Store()
        self.layout = layout
        self.skip_assets = skip_assets
        self.config = None
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()
        if self.config is not None:
            self.config.cancel_translation()

    def run(self) -> JobResult:
        import babeldoc.format.pdf.high_level as high_level
        import pymupdf

        from .config import parse_pages

        st = self.settings
        if not self.src.is_file():
            raise FileNotFoundError(f"文件不存在：{self.src}")
        with pymupdf.open(self.src) as doc:
            if doc.needs_pass:
                raise ValueError("PDF 已加密，需要先解除密码")
            pages = parse_pages(st.pages, doc.page_count)
        if not pages:
            raise ValueError("没有要翻译的页面")
        paths = output_paths(self.src, st.output_dir, st.engine)
        out_dir = paths["mono"].parent
        out_dir.mkdir(parents=True, exist_ok=True)

        journal = Journal()
        translator = make_translator(st, self.store, journal)
        self.progress(0, "检查翻译服务连接…")
        translator.preflight()
        if self._cancelled.is_set():
            raise Cancelled()

        high_level.init()
        layout_fixes.install()
        if not self.skip_assets:
            prepare_assets(self.progress)
        if self._cancelled.is_set():
            raise Cancelled()
        self.progress(0, "加载版面识别模型…")
        layout = self.layout or layout_model()

        # Work folder with finished batches; kept until the whole job succeeds.
        work = out_dir / f".{self.src.stem}.pdftrans-{st.engine}"
        signature = job_signature(self.src, st)
        sig_file = work / "signature.txt"
        if work.exists() and (not sig_file.exists() or sig_file.read_text() != signature):
            shutil.rmtree(work, ignore_errors=True)  # settings or file changed: start over
        work.mkdir(parents=True, exist_ok=True)
        sig_file.write_text(signature)
        terms = work / "glossary.csv"

        batches = page_batches(pages)
        started = time.time()
        translated_pages, translated_seconds = 0, 0.0
        monos, duals = [], []
        for k, batch in enumerate(batches):
            if self._cancelled.is_set():
                raise Cancelled()
            mono_k, dual_k, notes_k = (work / f"part-{k:04d}{ext}" for ext in (".pdf", ".dual.pdf", ".json"))
            label = f"第 {k + 1}/{len(batches)} 批（第 {batch[0] + 1}–{batch[-1] + 1} 页）"
            if mono_k.exists() and (not st.dual or dual_k.exists()):
                self._load_notes(journal, notes_k)
                self.progress((k + 1) / len(batches) * 100, f"{label}：沿用上次已完成的结果")
            else:
                before = {p.source for p in journal.all()}
                batch_started = time.time()
                result = self._run_batch(high_level, translator, layout, batch, work / f"run-{k:04d}", terms, k, len(batches), label,
                                         remaining=len(pages) - sum(len(b) for b in batches[:k]),
                                         pace=translated_seconds / translated_pages if translated_pages else 0.0)
                _rename(result.mono_pdf_path, mono_k)
                if st.dual:
                    _rename(result.dual_pdf_path, dual_k)
                merge_glossary(terms, getattr(result, "auto_extracted_glossary_path", None))
                shutil.rmtree(work / f"run-{k:04d}", ignore_errors=True)
                self._save_notes(journal, before, notes_k)
                translated_pages += len(batch)
                translated_seconds += time.time() - batch_started
            monos.append(mono_k)
            if st.dual:
                duals.append(dual_k)
            self.on_batch(batch, mono_k)

        self.progress(100, "合并各批结果…")
        mono = merge_pdfs(monos, paths["mono"])
        dual = merge_pdfs(duals, paths["dual"]) if duals else None
        glossary = None
        if terms.exists():
            shutil.copyfile(terms, paths["glossary"])
            glossary = paths["glossary"]

        self.progress(100, "生成质检报告…")
        report = build_report(journal.all(), mono)
        report.save(paths["report"])
        shutil.rmtree(work, ignore_errors=True)
        self.progress(100, "完成")
        return JobResult(
            mono=mono,
            dual=dual,
            report=report,
            glossary=glossary,
            seconds=time.time() - started,
            tokens=translator.token_count.value,
        )

    @staticmethod
    def _save_notes(journal: Journal, before: set[str], path: Path) -> None:
        records = [asdict(p) for p in journal.all() if p.source not in before]
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _load_notes(journal: Journal, path: Path) -> None:
        from .translator import Paragraph

        if path.exists():
            for record in json.loads(path.read_text(encoding="utf-8")):
                journal.add(Paragraph(**record))

    def _make_config(self, translator, layout, batch: list[int], out: Path, terms: Path):
        from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
        from babeldoc.glossary import Glossary

        st = self.settings
        llm = st.engine == "llm"
        table_model = None
        if st.translate_tables:
            from babeldoc.docvision.table_detection.rapidocr import RapidOCRModel

            table_model = RapidOCRModel()
        glossaries = _load_glossaries(st) if llm else []
        if llm and terms.exists():
            glossaries.append(Glossary.from_csv(terms, "zh-CN"))  # keep terms consistent across batches
        return TranslationConfig(
            translator=translator,
            input_file=str(self.src),
            lang_in="en",
            lang_out="zh-CN",
            doc_layout_model=layout,
            pages=page_spec(batch),
            output_dir=str(out),
            no_dual=not st.dual,
            no_mono=False,
            qps=max(1, st.qps),
            pool_max_workers=max(1, st.workers),
            # BabelDOC skips text shorter than 5 characters by default, which leaves short
            # figure labels such as "Host" or "node" in English next to translated ones.
            min_text_length=3,
            use_rich_pbar=False,
            report_interval=0.3,
            watermark_output_mode=WatermarkOutputMode.NoWatermark,
            table_model=table_model,
            custom_system_prompt=role_prompt(st),
            glossaries=glossaries,
            # Term extraction and style tags need an LLM; machine translation would mangle the tags.
            auto_extract_glossary=st.auto_glossary and llm,
            disable_rich_text_translate=not llm,
            ocr_workaround=st.ocr_workaround,
            primary_font_family=None if st.font_family == "auto" else st.font_family,
            only_include_translated_page=True,
            save_auto_extracted_glossary=True,
        )

    def _run_batch(self, high_level, translator, layout, batch, out, terms, k, n, label, remaining, pace):
        self.config = self._make_config(translator, layout, batch, out, terms)
        getattr(layout, "init_font_mapper", lambda _c: None)(self.config)
        if self._cancelled.is_set():
            raise Cancelled()

        def progress(pct: float, msg: str) -> None:
            eta = ""
            if pace:
                left = max(0.0, pace * (remaining - len(batch) * pct / 100)) / 60
                eta = f"，全部预计还需 {left:.0f} 分钟" if left >= 1 else "，全部预计不到 1 分钟"
            self.progress((k + pct / 100) / n * 100, f"{label} · {msg}{eta}")

        result = asyncio.run(self._translate(high_level, progress))
        if result is None:
            raise Cancelled()
        return result

    async def _translate(self, high_level, progress):
        result = None
        async for event in high_level.async_translate(self.config):
            kind = event["type"]
            if kind in ("progress_start", "progress_update", "progress_end"):
                stage = STAGE_NAMES.get(event["stage"], event["stage"])
                detail = ""
                if event.get("stage_total"):
                    detail = f"（{event.get('stage_current', 0)}/{event['stage_total']}）"
                progress(float(event.get("overall_progress", 0.0)), f"{stage}{detail}")
            elif kind == "error":
                if self._cancelled.is_set():
                    return None
                raise RuntimeError(str(event["error"]))
            elif kind == "finish":
                result = event["translate_result"]
                break
        return result
