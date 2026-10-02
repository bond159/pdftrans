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
        if not self.skip_assets:
            prepare_assets(self.progress)
        if self._cancelled.is_set():
            raise Cancelled()
        self.progress(0, "加载版面识别模型…")
        layout = self.layout or layout_model()

        table_model = None
        if st.translate_tables:
            from babeldoc.docvision.table_detection.rapidocr import RapidOCRModel

            table_model = RapidOCRModel()
        llm = st.engine == "llm"
        work = out_dir / f".{self.src.stem}.pdftrans-{st.engine}"
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
            pool_max_workers=max(1, st.workers),
            # BabelDOC skips text shorter than 5 characters by default, which leaves short
            # figure labels such as "Host" or "node" in English next to translated ones.
            min_text_length=3,
            use_rich_pbar=False,
            report_interval=0.3,
            watermark_output_mode=WatermarkOutputMode.NoWatermark,
            table_model=table_model,
            custom_system_prompt=role_prompt(st),
            glossaries=_load_glossaries(st) if llm else [],
            # Term extraction and style tags need an LLM; machine translation would mangle the tags.
            auto_extract_glossary=st.auto_glossary and llm,
            disable_rich_text_translate=not llm,
            ocr_workaround=st.ocr_workaround,
            primary_font_family=None if st.font_family == "auto" else st.font_family,
            only_include_translated_page=bool(st.pages),
            save_auto_extracted_glossary=True,
        )
        getattr(layout, "init_font_mapper", lambda _c: None)(self.config)

        started = time.time()
        try:
            result = asyncio.run(self._translate(high_level))
            if result is None:
                raise Cancelled()
            mono = _rename(result.mono_pdf_path, paths["mono"])
            dual = _rename(result.dual_pdf_path, paths["dual"])
            glossary = _rename(getattr(result, "auto_extracted_glossary_path", None), paths["glossary"])
        finally:
            shutil.rmtree(work, ignore_errors=True)

        self.progress(100, "生成质检报告…")
        report = build_report(journal.all(), mono)
        report.save(paths["report"])
        self.progress(100, "完成")
        return JobResult(
            mono=mono,
            dual=dual,
            report=report,
            glossary=glossary,
            seconds=time.time() - started,
            tokens=translator.token_count.value,
        )

    def _eta(self, stage: str, current: int, total: int) -> str:
        """Remaining time for the translation stage, from its pace so far."""
        now = time.time()
        start = self._stage_start.setdefault(stage, (now, current))
        elapsed, done = now - start[0], current - start[1]
        if stage != "Translate Paragraphs" or done <= 0 or elapsed < 10 or current >= total:
            return ""
        minutes = elapsed / done * (total - current) / 60
        return f"，预计还需 {minutes:.0f} 分钟" if minutes >= 1 else "，预计不到 1 分钟"

    async def _translate(self, high_level):
        result = None
        self._stage_start: dict[str, tuple[float, int]] = {}
        async for event in high_level.async_translate(self.config):
            kind = event["type"]
            if kind in ("progress_start", "progress_update", "progress_end"):
                stage = STAGE_NAMES.get(event["stage"], event["stage"])
                detail = ""
                if event.get("stage_total"):
                    current, total = event.get("stage_current", 0), event["stage_total"]
                    detail = f"（{current}/{total}{self._eta(event['stage'], current, total)}）"
                self.progress(float(event.get("overall_progress", 0.0)), f"{stage}{detail}")
            elif kind == "error":
                if self._cancelled.is_set():
                    return None
                raise RuntimeError(str(event["error"]))
            elif kind == "finish":
                result = event["translate_result"]
                break
        return result
