"""Offscreen tests of the reader: live batches, page navigation, sharp rendering."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pymupdf  # noqa: E402

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QApplication = None


def make_pdf(path: Path, pages: int) -> None:
    doc = pymupdf.open()
    for n in range(pages):
        doc.new_page(width=612, height=792).insert_text((72, 100), f"Page {n + 1}", fontsize=20)
    doc.save(path)


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.src = Path(cls.tmp.name) / "book.pdf"
        make_pdf(cls.src, 40)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pump(self):
        for _ in range(50):
            self.app.processEvents()

    def preview(self):
        from pdftrans.gui import Preview

        view = Preview()
        view.resize(1000, 700)
        view.show()
        view.set_original(str(self.src))
        self.pump()
        return view

    def test_batches_appear_while_translating(self):
        view = self.preview()
        view.begin_live("大模型", list(range(10, 20)))
        self.assertEqual(view.right_src.currentText(), "大模型")
        self.assertIn("正在翻译", view.source(1, 12))
        self.assertIn("不在", view.source(1, 30))

        part = pymupdf.open()
        part.insert_pdf(pymupdf.open(self.src), from_page=10, to_page=12)
        view.add_batch("大模型", part.tobytes(), [10, 11, 12])
        self.pump()
        self.assertEqual(view.page, 10)  # jumps to the first translated page
        doc, _, index = view.source(1, 11)
        self.assertEqual(index, 1)
        self.assertIn("正在翻译", view.source(1, 13))

        view.end_live("大模型")
        self.assertNotIn("正在翻译", view.source(1, 13))

    def test_navigation_and_zoom_keep_the_page(self):
        view = self.preview()
        view.go(25)
        self.pump()
        self.assertEqual(view.page_spin.value(), 26)
        view.zoom.setCurrentText("150%")
        self.pump()
        self.assertEqual(view.page, 25)
        self.assertAlmostEqual(view.canvas.zoom, 1.5)

    def test_pages_render_at_screen_density(self):
        view = self.preview()
        self.pump()
        self.assertTrue(view.canvas.cache)
        scale, pixmap = next(iter(view.canvas.cache.values()))
        self.assertAlmostEqual(scale, view.canvas.zoom * view.canvas.devicePixelRatioF())
        self.assertFalse(pixmap.isNull())


if __name__ == "__main__":
    unittest.main()
