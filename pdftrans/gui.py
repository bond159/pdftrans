"""Desktop GUI (PySide6)."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pymupdf
from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .config import ENGINES, PRESETS, PROOFREAD_MODES, output_paths, QWEN_MODELS, Settings, load_settings, parse_pages, save_settings
from .qa import KINDS, Item, Report
from .store import Store

ZOOMS = ["适合宽度", "50%", "75%", "100%", "125%", "150%", "200%"]
FONT_FAMILIES = {"auto": "自动（跟随原文）", "serif": "宋体风格（衬线）", "sans-serif": "黑体风格（无衬线）"}


class Worker(QObject):
    """Runs a translation job, or a connection test, off the GUI thread."""

    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, src: str, settings: Settings, test_only: bool = False, list_models: bool = False):
        super().__init__()
        self.src = src
        self.settings = settings
        self.test_only = test_only
        self.list_models = list_models
        self.job = None
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()
        if self.job is not None:
            self.job.cancel()

    @Slot()
    def run(self) -> None:
        from .engine import Cancelled, Job

        try:
            if self.list_models:
                self.done.emit(fetch_models(self.settings))
                return
            if self.test_only:
                self.done.emit(test_connection(self.settings))
                return
            self.job = Job(self.src, self.settings, self.progress.emit)
            if self.cancelled.is_set():
                self.job.cancel()
            self.done.emit(self.job.run())
        except Cancelled:
            self.failed.emit("已取消")
        except Exception as e:  # show every failure in the window instead of crashing
            self.failed.emit(f"{type(e).__name__}: {e}")


def test_connection(settings: Settings) -> str:
    """Translate one sentence with the chosen engine (and the review model, if used)."""
    import openai

    from .mt import make_client
    from .translator import Proofreader, llm_check

    sentence = "Attention is all you need."
    lines = []
    if settings.engine != "llm":
        try:
            result = make_client(settings.engine, settings.proxy).translate(sentence)
        except Exception as e:
            hint = "\n谷歌翻译在中国大陆需要能访问谷歌的网络代理。" if settings.engine == "google" else ""
            raise RuntimeError(f"无法连接{ENGINES[settings.engine]}：{e}{hint}") from e
        lines.append(f"{ENGINES[settings.engine]}：{result}")
    models = []
    if settings.engine == "llm":
        models.append(("翻译模型", settings.model))
    if settings.proofread == "full" and settings.resolved_api_key():
        if settings.engine != "llm" or settings.resolved_review_model() != settings.model:
            models.append(("审校模型", settings.resolved_review_model()))
    if models:
        client = openai.OpenAI(base_url=settings.base_url or None, api_key=settings.resolved_api_key() or "EMPTY", timeout=60)
        proof = Proofreader(settings, None, None, client)
        for role, model in models:
            lines.append(f"{role} {model}：{llm_check(proof, model)}")
    return "\n".join(lines)


def latin_only(widget: QWidget) -> QWidget:
    """Let a field that only ever holds addresses, names or numbers bypass the input method.

    On macOS a Chinese input method can swallow typing in some Qt fields; password
    fields are unaffected because they never use the input method. Fields for
    URLs, model names and proxies behave the same way with this.
    """
    targets = [widget]
    if isinstance(widget, QComboBox) and widget.lineEdit() is not None:
        targets.append(widget.lineEdit())
    for w in targets:
        w.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, False)
        w.setInputMethodHints(Qt.InputMethodHint.ImhLatinOnly | Qt.InputMethodHint.ImhNoPredictiveText)
    return widget


def fetch_models(settings: Settings) -> list[str]:
    """Model names offered by the configured OpenAI-compatible endpoint."""
    import openai

    client = openai.OpenAI(base_url=settings.base_url or None, api_key=settings.resolved_api_key() or "EMPTY", timeout=30)
    try:
        models = sorted(m.id for m in client.models.list())
    except openai.AuthenticationError as e:
        raise RuntimeError("API Key 无效，无法读取模型列表") from e
    except openai.APIConnectionError as e:
        raise RuntimeError("无法连接接口，请检查 Base URL 和网络") from e
    except openai.APIStatusError as e:
        raise RuntimeError(f"该接口不提供模型列表（HTTP {e.status_code}），请手动填写模型名称") from e
    if not models:
        raise RuntimeError("接口返回的模型列表为空，请手动填写模型名称")
    return models


def render_page(doc: pymupdf.Document, index: int, zoom: float) -> QPixmap:
    pix = doc[index].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class Preview(QWidget):
    """Two pages side by side, scrolling together. Each side shows the original or one of
    the translations (large model, Google, Microsoft), so translations can be compared."""

    ORIGINAL = "原文"

    def __init__(self):
        super().__init__()
        self.original: pymupdf.Document | None = None
        # label -> (document, original page -> page in that document)
        self.docs: dict[str, tuple[pymupdf.Document, dict[int, int]]] = {}
        self.page = 0

        bar = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一页")
        self.next_btn = QPushButton("下一页 ▶")
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_total = QLabel("/ 0")
        self.zoom = QComboBox()
        self.zoom.addItems(ZOOMS)
        self.left_src = QComboBox()
        self.right_src = QComboBox()
        for box in (self.left_src, self.right_src):
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            box.setMinimumContentsLength(8)
        for w in (self.prev_btn, self.page_spin, self.page_total, self.next_btn):
            bar.addWidget(w)
        bar.addStretch()
        bar.addWidget(QLabel("左"))
        bar.addWidget(self.left_src)
        bar.addWidget(QLabel("右"))
        bar.addWidget(self.right_src)
        bar.addWidget(QLabel("缩放"))
        bar.addWidget(self.zoom)

        self.left = QLabel()
        self.right = QLabel()
        for label in (self.left, self.right):
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            label.setStyleSheet("background: #ffffff; border: 1px solid #c8c8c8; color: #666;")
        pages = QWidget()
        row = QHBoxLayout(pages)
        row.addWidget(self.left)
        row.addWidget(self.right)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(pages)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.scroll, 1)

        self.prev_btn.clicked.connect(lambda: self.go(self.page - 1))
        self.next_btn.clicked.connect(lambda: self.go(self.page + 1))
        self.page_spin.valueChanged.connect(lambda v: self.go(v - 1))
        self.zoom.currentIndexChanged.connect(lambda _: self.refresh())
        self.left_src.currentIndexChanged.connect(lambda _: self.refresh())
        self.right_src.currentIndexChanged.connect(lambda _: self.refresh())
        self.left.setText("把英文 PDF 拖到窗口里，或点「选择 PDF…」")
        self.right.setText("翻译完成后在这里显示中文版")

    def set_original(self, path: str) -> None:
        self.original = pymupdf.open(path)
        self.docs = {self.ORIGINAL: (self.original, {i: i for i in range(self.original.page_count)})}
        self.page_spin.blockSignals(True)
        self.page_spin.setMaximum(self.original.page_count)
        self.page_spin.blockSignals(False)
        self.page_total.setText(f"/ {self.original.page_count}")
        self._fill_sources(left=self.ORIGINAL, right=self.ORIGINAL)
        self.go(0)

    def add_translation(self, label: str, path: Path, pages: list[int], show: bool = True) -> None:
        doc = pymupdf.open(path)
        self.docs[label] = (doc, {p: i for i, p in enumerate(pages) if i < doc.page_count})
        right = label if show else self.right_src.currentText()
        self._fill_sources(left=self.left_src.currentText() or self.ORIGINAL, right=right)
        if show and self.page not in self.docs[label][1] and pages:
            self.go(pages[0])
        else:
            self.refresh()

    def _fill_sources(self, left: str, right: str) -> None:
        for box, choice in ((self.left_src, left), (self.right_src, right)):
            box.blockSignals(True)
            box.clear()
            box.addItems(list(self.docs))
            box.setCurrentText(choice if choice in self.docs else self.ORIGINAL)
            box.blockSignals(False)

    def go_translated(self, label: str, translated_page: int) -> None:
        """Show the original page whose translation (in document label) is translated_page."""
        if label in self.docs:
            self.right_src.setCurrentText(label)
            for original, page in self.docs[label][1].items():
                if page == translated_page:
                    self.go(original)
                    return

    def go(self, page: int) -> None:
        if self.original is None:
            return
        self.page = max(0, min(page, self.original.page_count - 1))
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(self.page + 1)
        self.page_spin.blockSignals(False)
        self.refresh()

    def _zoom(self) -> float:
        choice = self.zoom.currentText()
        if choice.endswith("%"):
            return int(choice[:-1]) / 100
        width = self.scroll.viewport().width() / 2 - 30
        return max(0.2, width / self.original[self.page].rect.width)

    def _show(self, label: QLabel, source: str, zoom: float) -> None:
        if source not in self.docs:
            label.setPixmap(QPixmap())
            label.setText("翻译完成后在这里显示中文版")
            return
        doc, page_map = self.docs[source]
        if self.page in page_map:
            label.setPixmap(render_page(doc, page_map[self.page], zoom))
        else:
            label.setPixmap(QPixmap())
            label.setText("此页不在翻译范围内")

    def refresh(self) -> None:
        if self.original is None:
            return
        zoom = self._zoom()
        self._show(self.left, self.left_src.currentText(), zoom)
        if len(self.docs) == 1:
            self.right.setPixmap(QPixmap())
            self.right.setText("翻译完成后在这里显示中文版")
        else:
            self._show(self.right, self.right_src.currentText(), zoom)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        if self.zoom.currentText() == ZOOMS[0]:
            self.refresh()


class ReportView(QWidget):
    """Quality report: paragraphs to look at, with an editor for manual corrections."""

    jump = Signal(str, int)  # translation label, 1-based page in that PDF
    regenerate = Signal()

    def __init__(self, store_factory):
        super().__init__()
        self.store_factory = store_factory
        self.items: list[Item] = []
        self.label = ""
        self.summary = QLabel("翻译完成后，这里列出自动校对修正过的段落和仍需人工检查的地方。")
        self.summary.setWordWrap(True)
        self.filter = QComboBox()
        self.filter.addItem("全部", "")
        for kind, label in KINDS.items():
            self.filter.addItem(label, kind)
        self.filter.currentIndexChanged.connect(lambda _: self._fill())
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["类型", "页", "说明 / 译文"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._show_selected)
        self.table.cellDoubleClicked.connect(self._jump_selected)

        self.source = QPlainTextEdit()
        self.source.setReadOnly(True)
        self.translation = QPlainTextEdit()
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.TextFormat.PlainText)
        self.save_btn = QPushButton("保存我的修改")
        self.revert_btn = QPushButton("取消手动修改")
        self.regen_btn = QPushButton("重新生成 PDF")
        self.save_btn.clicked.connect(self._save)
        self.revert_btn.clicked.connect(self._revert)
        self.regen_btn.clicked.connect(self.regenerate.emit)

        detail = QWidget()
        d = QFormLayout(detail)
        d.addRow("原文", self.source)
        d.addRow("译文", self.translation)
        d.addRow("", self.notes)
        hint = QLabel(
            "{v1} 是公式占位符，<style id='1'>…</style> 是格式标记，修改译文时请原样保留。"
            "保存后点「重新生成 PDF」：其余段落直接用缓存，不会重复计费。"
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.PlainText)
        hint.setStyleSheet("color: #777;")
        d.addRow("", hint)
        buttons = QHBoxLayout()
        for b in (self.save_btn, self.revert_btn, self.regen_btn):
            buttons.addWidget(b)
        d.addRow(buttons)

        top = QHBoxLayout()
        top.addWidget(self.summary, 1)
        top.addWidget(QLabel("显示"))
        top.addWidget(self.filter)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(detail)
        split.setSizes([320, 320])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        self._enable(False)

    def _enable(self, on: bool) -> None:
        for w in (self.translation, self.save_btn, self.revert_btn):
            w.setEnabled(on)

    def set_report(self, report: Report, label: str = "") -> None:
        self.items = report.items
        self.label = label
        prefix = f"【{label}】" if label else ""
        self.summary.setText(prefix + report.summary() + "。双击一行可在预览中跳到对应页面。")
        self._fill()

    def _visible(self) -> list[Item]:
        kind = self.filter.currentData()
        return [i for i in self.items if not kind or i.kind == kind]

    def _fill(self) -> None:
        rows = self._visible()
        self.table.setRowCount(len(rows))
        for r, item in enumerate(rows):
            text = "；".join(item.notes) + "  —  " + (item.translation or item.source)[:80]
            for c, value in enumerate([KINDS[item.kind], str(item.page or "?"), text]):
                self.table.setItem(r, c, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

    def _selected(self) -> Item | None:
        rows = self.table.selectionModel().selectedRows()
        visible = self._visible()
        return visible[rows[0].row()] if rows and rows[0].row() < len(visible) else None

    def _show_selected(self) -> None:
        item = self._selected()
        if item is None:
            return
        self.source.setPlainText(item.source)
        self.translation.setPlainText(item.translation)
        notes = "；".join(item.notes)
        if item.draft and item.draft != item.translation:
            notes += f"\n审校前的译文：{item.draft}"
        self.notes.setText(notes)
        self._enable(item.kind != "english")

    def _jump_selected(self, *_):
        item = self._selected()
        if item and item.page:
            self.jump.emit(self.label, item.page)

    def _save(self) -> None:
        item = self._selected()
        text = self.translation.toPlainText().strip()
        if item is None or not text:
            return
        self.store_factory().set_override(item.source, text)
        item.translation = text
        item.kind = "manual"
        item.notes = ["使用你手动修改的译文（重新生成 PDF 后生效）"]
        self._fill()

    def _revert(self) -> None:
        item = self._selected()
        if item is not None:
            self.store_factory().remove_override(item.source)
            self.notes.setText("已取消手动修改，重新生成 PDF 后恢复自动译文。")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PDF 英译中 pdftrans {__version__}")
        self.resize(1440, 920)
        self.setAcceptDrops(True)
        self.settings = load_settings()
        self.src = ""
        self.job_thread: QThread | None = None
        self.worker: Worker | None = None
        self.result = None
        self.running_engine = "llm"
        self._store: Store | None = None

        self.preview = Preview()
        self.report_view = ReportView(self.store)
        self.report_view.jump.connect(self.jump_to_page)
        self.report_view.regenerate.connect(self.start)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.preview, "预览")
        self.tabs.addTab(self.report_view, "质检报告")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._build_panel())
        scroll.setMinimumWidth(400)
        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1010])
        self.setCentralWidget(splitter)
        self._load_into_form()

    def store(self) -> Store:
        if self._store is None:
            self._store = Store()
        return self._store

    # ----- form -------------------------------------------------------------
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        files = QGroupBox("文件")
        f = QFormLayout(files)
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("把英文 PDF 拖到窗口中")
        pick = QPushButton("选择 PDF…")
        pick.clicked.connect(self.choose_file)
        f.addRow(self._row(self.file_edit, pick))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("默认与原文件相同的文件夹")
        pick_out = QPushButton("浏览…")
        pick_out.clicked.connect(lambda: self._pick_dir(self.out_edit))
        f.addRow("输出到", self._row(self.out_edit, pick_out))
        layout.addWidget(files)

        engine_box = QGroupBox("翻译引擎")
        e = QFormLayout(engine_box)
        self.engine = QComboBox()
        for key, label in ENGINES.items():
            self.engine.addItem(label, key)
        self.engine.currentIndexChanged.connect(lambda _: self._update_engine_fields())
        e.addRow("引擎", self.engine)
        self.proxy = latin_only(QLineEdit())
        self.proxy.setPlaceholderText("留空则使用系统代理，例如 http://127.0.0.1:7890")
        self.proxy_label = QLabel("网络代理")
        e.addRow(self.proxy_label, self.proxy)
        self.engine_hint = QLabel()
        self.engine_hint.setWordWrap(True)
        self.engine_hint.setStyleSheet("color: #777;")
        e.addRow(self.engine_hint)
        layout.addWidget(engine_box)

        service = QGroupBox("大模型（翻译 / 审校）")
        s = QFormLayout(service)
        self.preset = QComboBox()
        self.preset.addItems(PRESETS.keys())
        self.preset.currentTextChanged.connect(self.on_preset)
        s.addRow("服务", self.preset)
        self.base_url = latin_only(QLineEdit())
        self.base_url.setClearButtonEnabled(True)
        s.addRow("Base URL", self.base_url)
        self.api_key = latin_only(QLineEdit())
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("sk-…（也可用环境变量 DASHSCOPE_API_KEY）")
        show = QCheckBox("显示")
        show.toggled.connect(
            lambda on: (
                self.api_key.setEchoMode(QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password),
                latin_only(self.api_key),  # setEchoMode turns the input method back on
            )
        )
        s.addRow("API Key", self._row(self.api_key, show))
        self.model = self._model_box()
        self.models_btn = QPushButton("获取模型列表")
        self.models_btn.setToolTip("从当前接口读取可用的模型名称（不同服务、套餐的模型名称不同）")
        self.models_btn.clicked.connect(self.fetch_models)
        s.addRow("翻译模型", self._row(self.model, self.models_btn))
        self.review_model = self._model_box()
        s.addRow("审校模型", self.review_model)
        self.test_btn = QPushButton("测试连接")
        self.test_btn.clicked.connect(self.test_connection)
        s.addRow(self.test_btn)
        layout.addWidget(service)

        opts = QGroupBox("翻译选项")
        o = QFormLayout(opts)
        self.proofread = QComboBox()
        for key, label in PROOFREAD_MODES.items():
            self.proofread.addItem(label, key)
        o.addRow("自动校对", self.proofread)
        self.dual = QCheckBox("同时输出左右双语对照版")
        o.addRow("输出", self.dual)
        # Number boxes rather than free text: they take digits directly, whatever input method is active.
        self.all_pages = QCheckBox("全部")
        self.all_pages.setChecked(True)
        self.page_from = QSpinBox()
        self.page_to = QSpinBox()
        for box in (self.page_from, self.page_to):
            box.setRange(1, 1)
            box.setEnabled(False)
        self.all_pages.toggled.connect(lambda on: [b.setEnabled(not on) for b in (self.page_from, self.page_to)])
        self.page_from.valueChanged.connect(lambda v: self.page_to.setMinimum(v))
        o.addRow("页码范围", self._row(self.all_pages, QLabel("第"), self.page_from, QLabel("页 至 第"), self.page_to, QLabel("页")))
        self.glossary = QLineEdit()
        self.glossary.setPlaceholderText("可选：CSV 文件，两列 source,target")
        pick_glossary = QPushButton("…")
        pick_glossary.clicked.connect(self.choose_glossary)
        o.addRow("术语表", self._row(self.glossary, pick_glossary))
        self.auto_glossary = QCheckBox("自动提取全文术语，保证译法一致")
        o.addRow(self.auto_glossary)
        self.extra = QPlainTextEdit()
        self.extra.setPlaceholderText("可选：学科领域或风格要求，例如\n这是一篇医学影像论文\nattention 译为“注意力”")
        self.extra.setFixedHeight(80)
        o.addRow("附加要求", self.extra)
        layout.addWidget(opts)

        adv = QGroupBox("高级")
        a = QFormLayout(adv)
        self.font_family = QComboBox()
        for key, label in FONT_FAMILIES.items():
            self.font_family.addItem(label, key)
        a.addRow("中文字体", self.font_family)
        self.workers = QSpinBox()
        self.workers.setRange(1, 64)
        self.workers.setToolTip("同时发出的翻译请求数。越大越快，太大可能被服务限流（会自动重试）。")
        a.addRow("同时请求数", self.workers)
        self.qps = QSpinBox()
        self.qps.setRange(1, 50)
        self.qps.setSuffix(" 次/秒")
        self.qps.setToolTip("每秒最多发出的新请求数。")
        a.addRow("请求速率上限", self.qps)
        self.tables = QCheckBox("翻译表格内的文字（实验性）")
        a.addRow(self.tables)
        self.ocr = QCheckBox("扫描版 PDF 兼容模式（已有 OCR 文字层时）")
        a.addRow(self.ocr)
        layout.addWidget(adv)

        self.start_btn = QPushButton("开始翻译")
        self.start_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        layout.addWidget(self._row(self.start_btn, self.cancel_btn))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel("就绪")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.open_btn = QPushButton("打开中文 PDF")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_result)
        self.open_dir_btn = QPushButton("打开输出文件夹")
        self.open_dir_btn.setEnabled(False)
        self.open_dir_btn.clicked.connect(self.open_output_dir)
        layout.addWidget(self._row(self.open_btn, self.open_dir_btn))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(90)
        layout.addWidget(self.log, 1)
        return panel

    @staticmethod
    def _model_box() -> QComboBox:
        box = QComboBox()
        box.setEditable(True)
        box.addItems(QWEN_MODELS)
        return latin_only(box)

    @staticmethod
    def _row(*widgets: QWidget) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for i, x in enumerate(widgets):
            h.addWidget(x, 1 if i == 0 else 0)
        return w

    @staticmethod
    def _select(box: QComboBox, data: str) -> None:
        box.setCurrentIndex(max(0, box.findData(data)))

    def _load_into_form(self) -> None:
        st = self.settings
        self._select(self.engine, st.engine)
        self.proxy.setText(st.proxy)
        self.preset.blockSignals(True)
        self.preset.setCurrentText(st.preset)
        self.preset.blockSignals(False)
        self.base_url.setText(st.base_url)
        self.api_key.setText(st.api_key)
        self.model.setCurrentText(st.model)
        self.review_model.setCurrentText(st.review_model)
        self._select(self.proofread, st.proofread)
        self._update_engine_fields()
        self.dual.setChecked(st.dual)
        self.glossary.setText(st.glossary_file)
        self.auto_glossary.setChecked(st.auto_glossary)
        self.extra.setPlainText(st.extra_prompt)
        self._select(self.font_family, st.font_family)
        self.qps.setValue(st.qps)
        self.workers.setValue(st.workers)
        self.tables.setChecked(st.translate_tables)
        self.ocr.setChecked(st.ocr_workaround)
        self.out_edit.setText(st.output_dir)

    def _collect(self) -> Settings:
        st = self.settings
        st.preset = self.preset.currentText()
        st.base_url = self.base_url.text().strip()
        st.api_key = self.api_key.text().strip()
        st.model = self.model.currentText().strip()
        st.review_model = self.review_model.currentText().strip()
        st.proofread = self.proofread.currentData()
        st.dual = self.dual.isChecked()
        st.pages = self._page_spec()
        st.glossary_file = self.glossary.text().strip()
        st.auto_glossary = self.auto_glossary.isChecked()
        st.extra_prompt = self.extra.toPlainText()
        st.font_family = self.font_family.currentData()
        st.qps = self.qps.value()
        st.workers = self.workers.value()
        st.translate_tables = self.tables.isChecked()
        st.ocr_workaround = self.ocr.isChecked()
        st.output_dir = self.out_edit.text().strip()
        st.engine = self.engine.currentData()
        st.proxy = self.proxy.text().strip()
        save_settings(st)
        return st

    @Slot(str)
    def on_preset(self, name: str) -> None:
        if PRESETS[name]:
            self.base_url.setText(PRESETS[name])
        else:
            # Custom endpoint: empty the field and put the cursor there so it's obvious where to type.
            self.base_url.clear()
            self.base_url.setPlaceholderText("在这里填写接口地址，例如 https://api.example.com/v1")
            self.base_url.setFocus()

    def _update_engine_fields(self) -> None:
        engine = self.engine.currentData()
        machine = engine != "llm"
        self.proxy.setEnabled(machine)
        self.proxy_label.setEnabled(machine)
        if engine == "google":
            hint = "免费。中国大陆需要能访问谷歌的网络代理。不支持术语表；填写下方大模型 API Key 后可用大模型审校。"
        elif engine == "microsoft":
            hint = "免费，国内一般可直接使用。不支持术语表；填写下方大模型 API Key 后可用大模型审校。"
        else:
            hint = "用下方的大模型翻译，效果最好，按用量计费。"
        self.engine_hint.setText(hint)
        self.auto_glossary.setEnabled(not machine)
        self.glossary.setEnabled(not machine)

    # ----- files ------------------------------------------------------------
    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 PDF", "", "PDF 文件 (*.pdf)")
        if path:
            self.load_file(path)

    def choose_glossary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择术语表", "", "CSV 文件 (*.csv)")
        if path:
            self.glossary.setText(path)

    def _pick_dir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            edit.setText(path)

    def load_file(self, path: str) -> None:
        try:
            self.preview.set_original(path)
        except Exception as e:
            QMessageBox.warning(self, "无法打开", f"无法打开该 PDF：\n{e}")
            return
        self.src = path
        self.file_edit.setText(path)
        self._reset_pages(self.preview.original.page_count)
        self.result = None
        self.open_btn.setEnabled(False)
        self.open_dir_btn.setEnabled(False)
        self.tabs.setCurrentWidget(self.preview)
        self.status.setText(f"已载入：{Path(path).name}（{self.preview.original.page_count} 页）")
        # Show earlier results for this file, from every engine, so they can be compared.
        found = []
        current = self.engine.currentData()
        for engine in sorted(ENGINES, key=lambda e: e == current):  # current engine last = shown on the right
            paths = output_paths(path, self.out_edit.text().strip(), engine)
            if paths["mono"].exists():
                self.preview.add_translation(ENGINES[engine], paths["mono"], self._pages_of(paths["mono"]))
                found.append(ENGINES[engine])
                if engine == current and paths["report"].exists():
                    try:
                        self.report_view.set_report(Report.load(paths["report"]), ENGINES[engine])
                    except (OSError, ValueError, TypeError, KeyError):
                        pass
        if found:
            self.status.setText(self.status.text() + "，已载入上次的结果：" + "、".join(found))

    def _page_spec(self) -> str:
        if self.all_pages.isChecked() or self.preview.original is None:
            return ""
        return f"{self.page_from.value()}-{self.page_to.value()}"

    def _reset_pages(self, count: int) -> None:
        self.all_pages.setChecked(True)
        for box in (self.page_from, self.page_to):
            box.setMinimum(1)
            box.setMaximum(count)
        self.page_from.setValue(1)
        self.page_to.setValue(count)

    def _pages_of(self, translated: Path) -> list[int]:
        """Original pages contained in a translated PDF, in order."""
        total = self.preview.original.page_count
        try:
            with pymupdf.open(translated) as doc:
                if doc.page_count == total:
                    return list(range(total))
        except Exception:
            pass
        try:
            return parse_pages(self._page_spec(), total)
        except ValueError:
            return list(range(total))

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if any(u.toLocalFile().lower().endswith(".pdf") for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for u in event.mimeData().urls():
            if u.toLocalFile().lower().endswith(".pdf"):
                self.load_file(u.toLocalFile())
                break

    # ----- jobs -------------------------------------------------------------
    def _run(self, worker: Worker) -> None:
        self.job_thread = QThread(self)
        self.worker = worker
        worker.moveToThread(self.job_thread)
        self.job_thread.started.connect(worker.run)
        worker.progress.connect(self.on_progress)
        worker.done.connect(self.on_done)
        worker.failed.connect(self.on_failed)
        worker.done.connect(self.job_thread.quit)
        worker.failed.connect(self.job_thread.quit)
        self.job_thread.finished.connect(self._job_ended)
        self.job_thread.finished.connect(self.job_thread.deleteLater)
        self.set_busy(True)
        self.job_thread.start()

    def set_busy(self, busy: bool) -> None:
        self.start_btn.setEnabled(not busy)
        self.test_btn.setEnabled(not busy)
        self.models_btn.setEnabled(not busy)
        self.report_view.regen_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def fetch_models(self) -> None:
        settings = self._collect()
        self.status.setText("正在读取模型列表…")
        self._run(Worker("", settings, list_models=True))

    def test_connection(self) -> None:
        settings = self._collect()
        self.status.setText("正在测试连接…")
        self._run(Worker("", settings, test_only=True))

    def start(self) -> None:
        if self.worker is not None:
            return
        if not self.src:
            QMessageBox.information(self, "提示", "请先选择一个 PDF 文件")
            return
        settings = self._collect()
        try:
            parse_pages(settings.pages, self.preview.original.page_count)
        except ValueError as e:
            QMessageBox.information(self, "提示", str(e))
            return
        if settings.engine == "llm":
            if not settings.resolved_api_key():
                QMessageBox.information(self, "提示", "请填写大模型的 API Key")
                return
            what = f"翻译模型 {settings.model}"
        else:
            what = ENGINES[settings.engine]
            if settings.proofread == "full" and not settings.resolved_api_key():
                self.log.appendPlainText("未填写大模型 API Key：只做规则检查，不做大模型审校。")
        self.log.appendPlainText(f"开始翻译 {Path(self.src).name}，{what}")
        self.progress.setValue(0)
        self.running_engine = settings.engine
        self._run(Worker(self.src, settings))

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.status.setText("正在取消…")

    @Slot(float, str)
    def on_progress(self, pct: float, msg: str) -> None:
        self.progress.setValue(int(pct * 10))
        self.status.setText(f"{pct:.0f}%  {msg}")
        if not msg.endswith("）"):  # keep per-item counters out of the log
            self.log.appendPlainText(msg)

    @Slot(object)
    def on_done(self, result) -> None:
        if isinstance(result, list):  # model list
            for box in (self.model, self.review_model):
                current = box.currentText()
                box.clear()
                box.addItems(result)
                box.setCurrentText(current)
            self.status.setText(f"已读取 {len(result)} 个模型，可在「翻译模型」「审校模型」下拉框中选择")
            self.log.appendPlainText("可用模型：" + "、".join(result))
            return
        if isinstance(result, str):  # connection test
            self.status.setText("连接成功")
            self.log.appendPlainText(result)
            QMessageBox.information(self, "连接成功", result)
            return
        self.result = result
        self.progress.setValue(1000)
        self.status.setText(f"完成，用时 {result.seconds / 60:.1f} 分钟。{result.report.summary()}")
        self.log.appendPlainText(f"中文 PDF：{result.mono}")
        if result.dual:
            self.log.appendPlainText(f"双语对照 PDF：{result.dual}")
        label = ENGINES[self.running_engine]
        if result.mono:
            self.preview.add_translation(label, result.mono, self._pages_of(result.mono))
        self.report_view.set_report(result.report, label)
        self.open_btn.setEnabled(True)
        self.open_dir_btn.setEnabled(True)
        if result.report.count("check") or result.report.count("english"):
            self.log.appendPlainText("有需要检查的段落，见「质检报告」。")

    @Slot(str)
    def on_failed(self, msg: str) -> None:
        self.status.setText(msg if msg == "已取消" else f"失败：{msg}")
        self.log.appendPlainText(msg)
        if msg != "已取消":
            QMessageBox.warning(self, "出错了", msg)

    def _job_ended(self) -> None:
        self.set_busy(False)
        self.worker = None
        self.job_thread = None

    def jump_to_page(self, label: str, page: int) -> None:
        self.tabs.setCurrentWidget(self.preview)
        self.preview.go_translated(label, page - 1)

    def open_result(self) -> None:
        if self.result and self.result.mono:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.mono)))

    def open_output_dir(self) -> None:
        if self.result and self.result.mono:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.mono.parent)))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker:
            self.worker.cancel()
        if self.job_thread:
            self.job_thread.quit()
            self.job_thread.wait(5000)
        self._collect()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setStyle("Fusion")
    app.setApplicationName("pdftrans")
    win = MainWindow()
    win.show()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        win.load_file(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
