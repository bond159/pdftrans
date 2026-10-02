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
from .config import PRESETS, PROOFREAD_MODES, QWEN_MODELS, Settings, load_settings, parse_pages, save_settings
from .qa import KINDS, Item, Report
from .store import Store

ZOOMS = ["适合宽度", "50%", "75%", "100%", "125%", "150%", "200%"]
FONT_FAMILIES = {"auto": "自动（跟随原文）", "serif": "宋体风格（衬线）", "sans-serif": "黑体风格（无衬线）"}


class Worker(QObject):
    """Runs a translation job, or a connection test, off the GUI thread."""

    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, src: str, settings: Settings, test_only: bool = False):
        super().__init__()
        self.src = src
        self.settings = settings
        self.test_only = test_only
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
    import openai

    client = openai.OpenAI(base_url=settings.base_url or None, api_key=settings.resolved_api_key() or "EMPTY", timeout=60)
    extra = {"enable_thinking": False} if "dashscope" in settings.base_url or settings.model.startswith("qwen") else {}
    lines = []
    models = [("翻译模型", settings.model)]
    if settings.proofread == "full" and settings.resolved_review_model() != settings.model:
        models.append(("审校模型", settings.resolved_review_model()))
    for role, model in models:
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "把这句话翻译成中文，只输出译文：Attention is all you need."}],
            max_tokens=100,
            extra_body=extra,
        )
        lines.append(f"{role} {model}：{(reply.choices[0].message.content or '').strip()}")
    return "\n".join(lines)


def render_page(doc: pymupdf.Document, index: int, zoom: float) -> QPixmap:
    pix = doc[index].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class Preview(QWidget):
    """Original and translated page side by side, scrolling together."""

    def __init__(self):
        super().__init__()
        self.original: pymupdf.Document | None = None
        self.translated: pymupdf.Document | None = None
        self.page_map: dict[int, int] = {}  # original page -> translated page
        self.page = 0

        bar = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一页")
        self.next_btn = QPushButton("下一页 ▶")
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_total = QLabel("/ 0")
        self.zoom = QComboBox()
        self.zoom.addItems(ZOOMS)
        self.show_original = QCheckBox("显示原文")
        self.show_original.setChecked(True)
        for w in (self.prev_btn, self.page_spin, self.page_total, self.next_btn):
            bar.addWidget(w)
        bar.addStretch()
        bar.addWidget(self.show_original)
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
        self.show_original.toggled.connect(lambda _: self.refresh())
        self.left.setText("把英文 PDF 拖到窗口里，或点「选择 PDF…」")
        self.right.setText("翻译完成后在这里显示中文版")

    def set_original(self, path: str) -> None:
        self.original = pymupdf.open(path)
        self.translated = None
        self.page_map = {}
        self.page_spin.blockSignals(True)
        self.page_spin.setMaximum(self.original.page_count)
        self.page_spin.blockSignals(False)
        self.page_total.setText(f"/ {self.original.page_count}")
        self.go(0)

    def set_translated(self, path: Path, pages: list[int]) -> None:
        self.translated = pymupdf.open(path)
        self.page_map = {p: i for i, p in enumerate(pages) if i < self.translated.page_count}
        self.go(self.page if self.page in self.page_map else pages[0])

    def go_translated(self, translated_page: int) -> None:
        """Show the page whose translation is page translated_page (0-based)."""
        for original, translated in self.page_map.items():
            if translated == translated_page:
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

    def _zoom(self, panes: int) -> float:
        choice = self.zoom.currentText()
        if choice.endswith("%"):
            return int(choice[:-1]) / 100
        width = self.scroll.viewport().width() / panes - 30
        return max(0.2, width / self.original[self.page].rect.width)

    def refresh(self) -> None:
        if self.original is None:
            return
        both = self.show_original.isChecked() or self.translated is None
        self.left.setVisible(both)
        zoom = self._zoom(2 if both else 1)
        if both:
            self.left.setPixmap(render_page(self.original, self.page, zoom))
        if self.translated is not None and self.page in self.page_map:
            self.right.setPixmap(render_page(self.translated, self.page_map[self.page], zoom))
        else:
            self.right.setPixmap(QPixmap())
            self.right.setText("此页不在翻译范围内" if self.translated is not None else "翻译完成后在这里显示中文版")

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        if self.zoom.currentText() == ZOOMS[0]:
            self.refresh()


class ReportView(QWidget):
    """Quality report: paragraphs to look at, with an editor for manual corrections."""

    jump = Signal(int)  # 1-based page in the Chinese PDF
    regenerate = Signal()

    def __init__(self, store_factory):
        super().__init__()
        self.store_factory = store_factory
        self.items: list[Item] = []
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

    def set_report(self, report: Report) -> None:
        self.items = report.items
        self.summary.setText(report.summary() + "。双击一行可在预览中跳到对应页面。")
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
            self.jump.emit(item.page)

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

        service = QGroupBox("大模型服务")
        s = QFormLayout(service)
        self.preset = QComboBox()
        self.preset.addItems(PRESETS.keys())
        self.preset.currentTextChanged.connect(self.on_preset)
        s.addRow("服务", self.preset)
        self.base_url = QLineEdit()
        s.addRow("Base URL", self.base_url)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("sk-…（也可用环境变量 DASHSCOPE_API_KEY）")
        show = QCheckBox("显示")
        show.toggled.connect(
            lambda on: self.api_key.setEchoMode(QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password)
        )
        s.addRow("API Key", self._row(self.api_key, show))
        self.model = self._model_box()
        s.addRow("翻译模型", self.model)
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
        self.pages = QLineEdit()
        self.pages.setPlaceholderText("全部；或如 1-5,8")
        o.addRow("页码范围", self.pages)
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
        self.qps = QSpinBox()
        self.qps.setRange(1, 50)
        self.qps.setSuffix(" 次/秒")
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
        return box

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
        self.preset.blockSignals(True)
        self.preset.setCurrentText(st.preset)
        self.preset.blockSignals(False)
        self.base_url.setText(st.base_url)
        self.api_key.setText(st.api_key)
        self.model.setCurrentText(st.model)
        self.review_model.setCurrentText(st.review_model)
        self._select(self.proofread, st.proofread)
        self.dual.setChecked(st.dual)
        self.pages.setText(st.pages)
        self.glossary.setText(st.glossary_file)
        self.auto_glossary.setChecked(st.auto_glossary)
        self.extra.setPlainText(st.extra_prompt)
        self._select(self.font_family, st.font_family)
        self.qps.setValue(st.qps)
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
        st.pages = self.pages.text().strip()
        st.glossary_file = self.glossary.text().strip()
        st.auto_glossary = self.auto_glossary.isChecked()
        st.extra_prompt = self.extra.toPlainText()
        st.font_family = self.font_family.currentData()
        st.qps = self.qps.value()
        st.translate_tables = self.tables.isChecked()
        st.ocr_workaround = self.ocr.isChecked()
        st.output_dir = self.out_edit.text().strip()
        save_settings(st)
        return st

    @Slot(str)
    def on_preset(self, name: str) -> None:
        if PRESETS[name]:
            self.base_url.setText(PRESETS[name])

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
        self.result = None
        self.open_btn.setEnabled(False)
        self.open_dir_btn.setEnabled(False)
        self.tabs.setCurrentWidget(self.preview)
        self.status.setText(f"已载入：{Path(path).name}（{self.preview.original.page_count} 页）")
        # Show an earlier result for this file if there is one.
        out_dir = Path(self.out_edit.text().strip() or Path(path).parent)
        mono = out_dir / f"{Path(path).stem}.zh-CN.pdf"
        report = out_dir / f"{Path(path).stem}.zh-CN.report.json"
        if mono.exists():
            self.preview.set_translated(mono, self._pages())
            if report.exists():
                try:
                    self.report_view.set_report(Report.load(report))
                except (OSError, ValueError, TypeError, KeyError):
                    pass
            self.status.setText(self.status.text() + "，已显示上次的翻译结果")

    def _pages(self) -> list[int]:
        try:
            return parse_pages(self.pages.text(), self.preview.original.page_count)
        except ValueError:
            return list(range(self.preview.original.page_count))

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
        self.report_view.regen_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

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
        if not settings.resolved_api_key():
            QMessageBox.information(self, "提示", "请填写 API Key")
            return
        self.log.appendPlainText(f"开始翻译 {Path(self.src).name}，翻译模型 {settings.model}")
        self.progress.setValue(0)
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
        if result.mono:
            self.preview.set_translated(result.mono, self._pages())
        self.report_view.set_report(result.report)
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

    def jump_to_page(self, page: int) -> None:
        self.tabs.setCurrentWidget(self.preview)
        self.preview.go_translated(page - 1)

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
