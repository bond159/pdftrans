"""Desktop GUI (PySide6) for translating PDFs."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pymupdf
from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .config import LANGUAGES, PRESETS, Settings, load_settings, parse_pages, save_settings
from .llm import TranslationCache, Translator, make_backend
from .pipeline import Cancelled, JobResult, translate_pdf

MODE_LABELS = {
    "mono": "仅译文（保留原排版）",
    "dual": "双语对照（左原文 · 右译文）",
    "alt": "双语交替（原文页 + 译文页）",
}
ZOOMS = ["适合宽度", "50%", "75%", "100%", "125%", "150%", "200%"]


class Worker(QObject):
    """Runs one translation job off the GUI thread."""

    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, src: str, settings: Settings, test_only: bool = False):
        super().__init__()
        self.src = src
        self.settings = settings
        self.test_only = test_only
        self.cancel = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            backend = make_backend(self.settings)
            cache = TranslationCache() if self.settings.use_cache else None
            translator = Translator(backend, self.settings, cache)
            if self.test_only:
                self.done.emit(translator.test())
                return
            result = translate_pdf(self.src, self.settings, translator, self.progress.emit, self.cancel)
            self.done.emit(result)
        except Cancelled:
            self.failed.emit("已取消")
        except Exception as e:  # surface every failure in the GUI instead of crashing
            self.failed.emit(f"{type(e).__name__}: {e}")


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
            label.setStyleSheet("background: #ffffff; border: 1px solid #c8c8c8;")
        pages = QWidget()
        row = QHBoxLayout(pages)
        row.addWidget(self.left)
        row.addWidget(self.right)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(pages)
        self.scroll.setStyleSheet("QScrollArea { background: #e9e9ec; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.scroll, 1)

        self.prev_btn.clicked.connect(lambda: self.go(self.page - 1))
        self.next_btn.clicked.connect(lambda: self.go(self.page + 1))
        self.page_spin.valueChanged.connect(lambda v: self.go(v - 1))
        self.zoom.currentIndexChanged.connect(lambda _: self.refresh())
        self.show_original.toggled.connect(lambda _: self.refresh())
        self.clear()

    def clear(self) -> None:
        self.left.setText("拖入或选择一个 PDF 文件")
        self.right.setText("翻译完成后在这里显示译文")

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
        self.page_map = {p: i for i, p in enumerate(pages)}
        self.go(pages[0] if self.page not in self.page_map else self.page)

    def go(self, page: int) -> None:
        if self.original is None:
            return
        self.page = max(0, min(page, self.original.page_count - 1))
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(self.page + 1)
        self.page_spin.blockSignals(False)
        self.refresh()

    def _zoom_for(self, doc: pymupdf.Document, index: int, panes: int) -> float:
        choice = self.zoom.currentText()
        if choice.endswith("%"):
            return int(choice[:-1]) / 100
        width = self.scroll.viewport().width() / panes - 30
        return max(0.2, width / doc[index].rect.width)

    def refresh(self) -> None:
        if self.original is None:
            return
        both = self.show_original.isChecked() or self.translated is None
        self.left.setVisible(both)
        panes = 2 if both else 1
        zoom = self._zoom_for(self.original, self.page, panes)
        if both:
            self.left.setPixmap(render_page(self.original, self.page, zoom))
        if self.translated is not None and self.page in self.page_map:
            self.right.setPixmap(render_page(self.translated, self.page_map[self.page], zoom))
        elif self.translated is not None:
            self.right.setText("此页不在翻译范围内")
        else:
            self.right.setText("翻译完成后在这里显示译文")

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        if self.zoom.currentText() == ZOOMS[0]:
            self.refresh()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PDF 翻译器 pdftrans {__version__}")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        self.settings = load_settings()
        self.src = ""
        self.job_thread: QThread | None = None
        self.worker: Worker | None = None
        self.result: JobResult | None = None

        self.preview = Preview()
        panel = self._build_panel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(380)
        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 980])
        self.setCentralWidget(splitter)
        self._load_into_form()

    # ----- form -------------------------------------------------------------
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        files = QGroupBox("文件")
        f = QFormLayout(files)
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("把 PDF 拖到窗口中，或点击右侧按钮")
        pick = QPushButton("选择 PDF…")
        pick.clicked.connect(self.choose_file)
        f.addRow(self._row(self.file_edit, pick))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("默认与原文件相同的文件夹")
        pick_out = QPushButton("浏览…")
        pick_out.clicked.connect(self.choose_output_dir)
        f.addRow("输出到", self._row(self.out_edit, pick_out))
        layout.addWidget(files)

        service = QGroupBox("翻译服务（大模型 API）")
        s = QFormLayout(service)
        self.preset = QComboBox()
        self.preset.addItems(PRESETS.keys())
        self.preset.currentTextChanged.connect(self.on_preset)
        s.addRow("服务商", self.preset)
        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText("https://…/v1")
        s.addRow("Base URL", self.base_url)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("sk-…（也可用环境变量）")
        show = QCheckBox("显示")
        show.toggled.connect(
            lambda on: self.api_key.setEchoMode(QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password)
        )
        s.addRow("API Key", self._row(self.api_key, show))
        self.model = QLineEdit()
        s.addRow("模型", self.model)
        self.effort = QComboBox()
        self.effort.addItems(["low", "medium", "high"])
        self.effort_label = QLabel("推理强度")
        s.addRow(self.effort_label, self.effort)
        self.test_btn = QPushButton("测试连接")
        self.test_btn.clicked.connect(self.test_connection)
        s.addRow(self.test_btn)
        layout.addWidget(service)

        opts = QGroupBox("翻译选项")
        o = QFormLayout(opts)
        self.lang = QComboBox()
        for code, (name, _) in LANGUAGES.items():
            self.lang.addItem(name, code)
        o.addRow("目标语言", self.lang)
        self.pages = QLineEdit()
        self.pages.setPlaceholderText("全部；或如 1-5,8")
        o.addRow("页码范围", self.pages)
        self.mode_boxes = {m: QCheckBox(label) for m, label in MODE_LABELS.items()}
        modes = QWidget()
        mv = QVBoxLayout(modes)
        mv.setContentsMargins(0, 0, 0, 0)
        for box in self.mode_boxes.values():
            mv.addWidget(box)
        o.addRow("输出", modes)
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 32)
        o.addRow("并发请求数", self.concurrency)
        self.batch_chars = QSpinBox()
        self.batch_chars.setRange(200, 20000)
        self.batch_chars.setSingleStep(500)
        self.batch_chars.setSuffix(" 字符")
        o.addRow("每批文本量", self.batch_chars)
        self.font_edit = QLineEdit()
        self.font_edit.setPlaceholderText("可选：译文字体 .ttf/.otf")
        pick_font = QPushButton("…")
        pick_font.clicked.connect(self.choose_font)
        o.addRow("字体", self._row(self.font_edit, pick_font))
        self.use_cache = QCheckBox("缓存译文（重复翻译不再计费）")
        o.addRow(self.use_cache)
        self.extra = QPlainTextEdit()
        self.extra.setPlaceholderText("可选：术语表或风格要求，例如\nattention → 注意力\nLLM → 大语言模型\n译文风格：学术、简洁")
        self.extra.setFixedHeight(90)
        o.addRow("附加要求", self.extra)
        layout.addWidget(opts)

        self.start_btn = QPushButton("开始翻译")
        self.start_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        layout.addWidget(self._row(self.start_btn, self.cancel_btn))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel("就绪")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.open_dual_btn = QPushButton("打开结果 PDF")
        self.open_dual_btn.setEnabled(False)
        self.open_dual_btn.clicked.connect(self.open_result)
        self.open_dir_btn = QPushButton("打开输出文件夹")
        self.open_dir_btn.setEnabled(False)
        self.open_dir_btn.clicked.connect(self.open_output_dir)
        layout.addWidget(self._row(self.open_dual_btn, self.open_dir_btn))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(100)
        layout.addWidget(self.log, 1)
        return panel

    @staticmethod
    def _row(*widgets: QWidget) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for i, x in enumerate(widgets):
            h.addWidget(x, 1 if i == 0 else 0)
        return w

    def _load_into_form(self) -> None:
        st = self.settings
        self.preset.blockSignals(True)
        if st.preset in PRESETS:
            self.preset.setCurrentText(st.preset)
        self.preset.blockSignals(False)
        self._current_preset = self.preset.currentText()
        self.base_url.setText(st.base_url)
        self.api_key.setText(st.api_key)
        self.model.setText(st.model)
        self.effort.setCurrentText(st.effort)
        self._update_provider_fields()
        idx = self.lang.findData(st.target_lang)
        self.lang.setCurrentIndex(max(0, idx))
        self.pages.setText(st.pages)
        for m, box in self.mode_boxes.items():
            box.setChecked(m in st.modes)
        self.concurrency.setValue(st.concurrency)
        self.batch_chars.setValue(st.batch_chars)
        self.font_edit.setText(st.font_file)
        self.use_cache.setChecked(st.use_cache)
        self.extra.setPlainText(st.extra_prompt)
        self.out_edit.setText(st.output_dir)

    def _collect(self) -> Settings:
        st = self.settings
        st.preset = self.preset.currentText()
        st.provider = PRESETS[st.preset][0]
        st.base_url = self.base_url.text().strip()
        st.api_key = self.api_key.text().strip()
        st.saved_keys[st.preset] = st.api_key
        st.model = self.model.text().strip()
        st.effort = self.effort.currentText()
        st.target_lang = self.lang.currentData()
        st.pages = self.pages.text().strip()
        st.modes = [m for m, box in self.mode_boxes.items() if box.isChecked()]
        st.concurrency = self.concurrency.value()
        st.batch_chars = self.batch_chars.value()
        st.font_file = self.font_edit.text().strip()
        st.use_cache = self.use_cache.isChecked()
        st.extra_prompt = self.extra.toPlainText()
        st.output_dir = self.out_edit.text().strip()
        save_settings(st)
        return st

    def _update_provider_fields(self) -> None:
        is_claude = PRESETS[self.preset.currentText()][0] == "anthropic"
        self.effort.setVisible(is_claude)
        self.effort_label.setVisible(is_claude)
        self.base_url.setPlaceholderText("留空使用官方接口" if is_claude else "https://…/v1")

    @Slot(str)
    def on_preset(self, name: str) -> None:
        # Remember the key typed for the previous service, restore the one for this service.
        self.settings.saved_keys[self._current_preset] = self.api_key.text().strip()
        self._current_preset = name
        _, url, model = PRESETS[name]
        self.base_url.setText(url)
        self.model.setText(model)
        self.api_key.setText(self.settings.saved_keys.get(name, ""))
        self._update_provider_fields()

    # ----- files ------------------------------------------------------------
    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 PDF", "", "PDF 文件 (*.pdf)")
        if path:
            self.load_file(path)

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.out_edit.setText(path)

    def choose_font(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择字体", "", "字体 (*.ttf *.otf *.ttc)")
        if path:
            self.font_edit.setText(path)

    def load_file(self, path: str) -> None:
        try:
            self.preview.set_original(path)
        except Exception as e:
            QMessageBox.warning(self, "无法打开", f"无法打开该 PDF：\n{e}")
            return
        self.src = path
        self.file_edit.setText(path)
        self.result = None
        self.open_dual_btn.setEnabled(False)
        self.open_dir_btn.setEnabled(False)
        self.status.setText(f"已载入：{Path(path).name}（{self.preview.original.page_count} 页）")

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
        self.cancel_btn.setEnabled(busy)

    def test_connection(self) -> None:
        settings = self._collect()
        self.status.setText("正在测试连接…")
        self._run(Worker("", settings, test_only=True))

    def start(self) -> None:
        if not self.src:
            QMessageBox.information(self, "提示", "请先选择一个 PDF 文件")
            return
        settings = self._collect()
        if not settings.modes:
            QMessageBox.information(self, "提示", "请至少选择一种输出方式")
            return
        try:
            parse_pages(settings.pages, self.preview.original.page_count)
        except ValueError as e:
            QMessageBox.information(self, "提示", str(e))
            return
        if not settings.resolved_api_key() and "localhost" not in settings.base_url:
            reply = QMessageBox.question(self, "未填写 API Key", "没有填写 API Key，仍然继续吗？")
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.log.appendPlainText(f"开始翻译 {Path(self.src).name} → {settings.target_lang}，模型 {settings.model}")
        self.progress.setRange(0, 0)
        self._run(Worker(self.src, settings))

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel.set()
            self.status.setText("正在取消…")

    @Slot(int, int, str)
    def on_progress(self, done: int, total: int, msg: str) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.status.setText(msg)
        self.log.appendPlainText(msg)

    @Slot(object)
    def on_done(self, result) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if isinstance(result, str):  # connection test
            self.status.setText("连接成功")
            self.log.appendPlainText(f"连接成功，测试译文：{result}")
            QMessageBox.information(self, "连接成功", f"测试译文：\n{result}")
            return
        self.result = result
        lines = [f"完成：{result.units} 段，新翻译 {result.translated} 段，缓存 {result.cached} 段"]
        lines += [f"  {MODE_LABELS[m]}：{p}" for m, p in result.outputs.items()]
        self.status.setText(lines[0])
        self.log.appendPlainText("\n".join(lines))
        if "mono" in result.outputs:
            self.preview.set_translated(result.outputs["mono"], result.pages)
        self.open_dual_btn.setEnabled(True)
        self.open_dir_btn.setEnabled(True)

    @Slot(str)
    def on_failed(self, msg: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText(f"失败：{msg}" if msg != "已取消" else msg)
        self.log.appendPlainText(msg)
        if msg != "已取消":
            QMessageBox.warning(self, "出错了", msg)

    def _job_ended(self) -> None:
        self.set_busy(False)
        self.worker = None
        self.job_thread = None

    def open_result(self) -> None:
        if self.result:
            outputs = self.result.outputs
            path = outputs.get("dual") or outputs.get("mono") or next(iter(outputs.values()))
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_output_dir(self) -> None:
        if self.result:
            folder = next(iter(self.result.outputs.values())).parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker:
            self.worker.cancel.set()
        if self.job_thread:
            self.job_thread.quit()
            self.job_thread.wait(3000)
        self._collect()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        win.load_file(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
