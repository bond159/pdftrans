"""Run one translation job: BabelDOC does layout and typesetting, our translator proofreads."""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Settings
from .qa import Report, build_report
from .store import Store
from .translator import Journal, ProofreadingTranslator, role_prompt

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


class Job:
    """One PDF translation. Call run() from a worker thread; cancel() from anywhere."""

    def __init__(
        self,
        src: str | Path,
        settings: Settings,
        progress: ProgressFn | None = None,
        store: Store | None = None,
        layout=None,
        skip_assets: bool = False,
    ):
        self.src = Path(src)
        self.settings = settings
        self.progress = progress or (lambda pct, msg: None)
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
        from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode

        st = self.settings
        if not self.src.is_file():
            raise FileNotFoundError(f"文件不存在：{self.src}")
        out_dir = Path(st.output_dir) if st.output_dir else self.src.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        high_level.init()
        if not self.skip_assets:
            prepare_assets(self.progress)
        if self._cancelled.is_set():
            raise Cancelled()
        self.progress(0, "加载版面识别模型…")
        layout = self.layout or layout_model()

        journal = Journal()
        translator = ProofreadingTranslator(st, self.store, journal)
        table_model = None
        if st.translate_tables:
            from babeldoc.docvision.table_detection.rapidocr import RapidOCRModel

            table_model = RapidOCRModel()

        work = out_dir / f".{self.src.stem}.pdftrans"
        self.config = TranslationConfig(
            translator=translator,
            input_file=str(self.src),
            lang_in="en",
            lang_out="zh-CN",
            doc_layout_model=layout,
            pages=st.pages or None,
            output_dir=str(work),
            no_dual=not st.dual,
            no_mono=False,
            qps=max(1, st.qps),
            use_rich_pbar=False,
            report_interval=0.3,
            watermark_output_mode=WatermarkOutputMode.NoWatermark,
            table_model=table_model,
            custom_system_prompt=role_prompt(st),
            glossaries=_load_glossaries(st),
            auto_extract_glossary=st.auto_glossary,
            ocr_workaround=st.ocr_workaround,
            primary_font_family=None if st.font_family == "auto" else st.font_family,
            only_include_translated_page=bool(st.pages),
            save_auto_extracted_glossary=True,
        )
        getattr(layout, "init_font_mapper", lambda _c: None)(self.config)

        started = time.time()
        result = asyncio.run(self._translate(high_level))
        if result is None:
            raise Cancelled()

        stem = self.src.stem
        mono = _rename(result.mono_pdf_path, out_dir / f"{stem}.zh-CN.pdf")
        dual = _rename(result.dual_pdf_path, out_dir / f"{stem}.zh-CN.dual.pdf")
        glossary = _rename(getattr(result, "auto_extracted_glossary_path", None), out_dir / f"{stem}.zh-CN.glossary.csv")
        shutil.rmtree(work, ignore_errors=True)

        self.progress(100, "生成质检报告…")
        report = build_report(journal.all(), mono)
        report.save(out_dir / f"{stem}.zh-CN.report.json")
        self.progress(100, "完成")
        return JobResult(
            mono=mono,
            dual=dual,
            report=report,
            glossary=glossary,
            seconds=time.time() - started,
            tokens=translator.token_count.value,
        )

    async def _translate(self, high_level):
        result = None
        async for event in high_level.async_translate(self.config):
            kind = event["type"]
            if kind in ("progress_start", "progress_update", "progress_end"):
                stage = STAGE_NAMES.get(event["stage"], event["stage"])
                detail = ""
                if event.get("stage_total"):
                    detail = f"（{event.get('stage_current', 0)}/{event['stage_total']}）"
                self.progress(float(event.get("overall_progress", 0.0)), f"{stage}{detail}")
            elif kind == "error":
                if self._cancelled.is_set():
                    return None
                raise RuntimeError(str(event["error"]))
            elif kind == "finish":
                result = event["translate_result"]
                break
        return result
