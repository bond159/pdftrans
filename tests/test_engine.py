"""Integration tests: the proofreading translator against a mock API, and full BabelDOC runs.

The full runs use a heuristic stand-in for BabelDOC's layout model, so they don't
need the 75 MB ONNX model, but BabelDOC still downloads its fonts on first use.
Set PDFTRANS_SKIP_E2E=1 to skip them when offline.
"""

import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

import pymupdf

from tests.support.mock_llm import MockLLM, fake_chinese
from tests.support.sandbox import HeuristicLayoutModel, patch_tiktoken

patch_tiktoken()

from pdftrans.config import Settings  # noqa: E402
from pdftrans.store import Store  # noqa: E402
from pdftrans.translator import Journal, ProofreadingTranslator  # noqa: E402

PARAGRAPH = (
    "Large language models have shown remarkable ability in 12 natural language tasks. "
    "However, translating documents while preserving their layout remains challenging [3]."
)


def batch_prompt(texts):
    items = [{"id": i, "input": t, "layout_label": "text"} for i, t in enumerate(texts)]
    return "You translate.\n\n## Here is the input:\n\n" + json.dumps(items)


def make_pdf(path: Path, pages: int = 2) -> None:
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(pymupdf.Rect(72, 60, 540, 100), f"Chapter {n + 1} Overview", fontname="tibo", fontsize=16)
        for k in range(3):
            page.insert_textbox(pymupdf.Rect(72, 120 + k * 120, 540, 230 + k * 120), PARAGRAPH, fontname="tiro", fontsize=11)
    doc.save(path)


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = Store(Path(self.tmp.name) / "s.db")

    def tearDown(self):
        self.tmp.cleanup()

    def translator(self, llm, mode="full"):
        st = Settings(base_url=llm.url, api_key="k", model="qwen-plus", review_model="qwen-max", proofread=mode)
        tr = ProofreadingTranslator(st, self.store, Journal())
        tr.ignore_cache = True
        return tr

    def test_review_fixes_dropped_numbers(self):
        with MockLLM() as llm:
            tr = self.translator(llm)
            out = json.loads(tr.llm_translate(batch_prompt([PARAGRAPH])))
            self.assertEqual(out[0]["output"], fake_chinese(PARAGRAPH))
            self.assertEqual([r["model"] for r in llm.requests], ["qwen-plus", "qwen-max"])
            self.assertIs(llm.requests[0]["enable_thinking"], False)
        record = tr.journal.all()[0]
        self.assertTrue(record.corrected)
        self.assertEqual(record.issues, [])
        self.assertIn("数字遗漏", record.problems)

    def test_rules_mode_retranslates_failing_paragraphs(self):
        with MockLLM() as llm:
            tr = self.translator(llm, mode="rules")
            out = json.loads(tr.llm_translate(batch_prompt([PARAGRAPH])))
            self.assertIn("12", out[0]["output"])
            self.assertEqual(len(llm.requests), 2)  # batch + one re-translation

    def test_off_mode_passes_through(self):
        with MockLLM() as llm:
            tr = self.translator(llm, mode="off")
            out = json.loads(tr.llm_translate(batch_prompt([PARAGRAPH])))
            self.assertNotIn("12", out[0]["output"])
            self.assertEqual(len(llm.requests), 1)
        self.assertTrue(tr.journal.all()[0].issues)  # the report still flags it

    def test_override_wins_and_skips_the_model(self):
        self.store.set_override(PARAGRAPH, "手动译文")
        with MockLLM() as llm:
            tr = self.translator(llm)
            single = "Rules...\n\nNow translate the following text:\n\n" + PARAGRAPH
            self.assertEqual(tr.llm_translate(single), "手动译文")
            self.assertEqual(llm.requests, [])
            out = json.loads(tr.llm_translate(batch_prompt([PARAGRAPH])))
            self.assertEqual(out[0]["output"], "手动译文")
        self.assertTrue(tr.journal.all()[0].overridden)

    def test_term_extraction_prompts_pass_through(self):
        with MockLLM() as llm:
            tr = self.translator(llm)
            self.assertEqual(tr.llm_translate("Extract terms as a JSON list"), "[]")
            self.assertEqual(len(llm.requests), 1)


@unittest.skipIf(os.environ.get("PDFTRANS_SKIP_E2E"), "end-to-end tests disabled")
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dir = Path(cls.tmp.name)
        cls.src = cls.dir / "paper.pdf"
        make_pdf(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def job(self, llm, src=None, on_batch=None, model=None, **kw):
        from pdftrans.engine import Job

        # A unique model name keeps BabelDOC's on-disk translation cache from leaking between runs.
        st = Settings(base_url=llm.url, api_key="k", model=model or f"qwen-test-{uuid.uuid4().hex[:8]}",
                      output_dir=str(self.dir / "out"), auto_glossary=False, **kw)
        return Job(src or self.src, st, store=Store(self.dir / "s.db"), layout=HeuristicLayoutModel(), skip_assets=True,
                   on_batch=on_batch)

    def test_translate_with_dual_output(self):
        with MockLLM() as llm:
            result = self.job(llm, dual=True).run()
        self.assertEqual(result.mono.name, "paper.zh-CN.pdf")
        self.assertEqual(result.dual.name, "paper.zh-CN.dual.pdf")
        mono = pymupdf.open(result.mono)
        self.assertEqual(mono.page_count, 2)
        text = mono[0].get_text()
        self.assertNotIn("remarkable", text)
        self.assertIn(fake_chinese("Overview"), text)
        dual = pymupdf.open(result.dual)
        self.assertAlmostEqual(dual[0].rect.width, 2 * 612, delta=1)
        self.assertTrue(result.report.count("fixed") > 0)
        self.assertTrue((self.dir / "out" / "paper.zh-CN.report.json").exists())

    def test_mono_only_by_default(self):
        with MockLLM() as llm:
            result = self.job(llm).run()
        self.assertIsNotNone(result.mono)
        self.assertIsNone(result.dual)

    def test_cancel(self):
        from pdftrans.engine import Cancelled

        with MockLLM() as llm:
            job = self.job(llm, proofread="off")
            threading.Timer(0.5, job.cancel).start()
            started = time.time()
            with self.assertRaises(Cancelled):
                job.run()
            self.assertLess(time.time() - started, 60)

    def test_batches_are_reported_and_resumed(self):
        from pdftrans.engine import Cancelled

        src = self.dir / "book.pdf"
        make_pdf(src, pages=6)  # batches of 3 and 3 pages
        model = f"qwen-test-{uuid.uuid4().hex[:8]}"
        work = self.dir / "out" / ".book.pdftrans-llm"
        with MockLLM() as llm:
            seen = []
            job = self.job(llm, src=src, model=model, on_batch=lambda pages, path: (seen.append((pages, path.exists())), job.cancel()))
            with self.assertRaises(Cancelled):
                job.run()
            self.assertEqual(seen, [([0, 1, 2], True)])
            self.assertTrue((work / "part-0000.pdf").exists())  # kept for the next run

            seen = []
            requests = len(llm.requests)
            result = self.job(llm, src=src, model=model, on_batch=lambda pages, path: seen.append(pages)).run()
            self.assertEqual(seen, [[0, 1, 2], [3, 4, 5]])
            self.assertEqual(pymupdf.open(result.mono).page_count, 6)
            self.assertFalse(work.exists())
            self.assertGreater(len(llm.requests), requests)

    def test_contents_entries_keep_their_lines(self):
        src = self.dir / "toc.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 80), "Table of Contents", fontname="tibo", fontsize=24)
        entries = []
        y = 130
        for chapter in (1, 2):
            page.insert_text((72, y), f"Chapter {chapter}   Networks and the Internet", fontname="tibo", fontsize=12)
            page.insert_text((520, y), str(chapter * 40), fontname="tibo", fontsize=12)
            entries.append(str(chapter * 40))
            y += 18
            for k in range(1, 8):
                number = str(chapter * 40 + k * 3)
                page.insert_text((100, y), f"{chapter}.{k}", fontname="tiro", fontsize=10)
                page.insert_text((130, y), f"Packet Switching Delay and Loss Part {k}", fontname="tiro", fontsize=10)
                page.insert_text((520, y), number, fontname="tiro", fontsize=10)
                entries.append(number)
                y += 14
        doc.save(src)
        with MockLLM() as llm:
            result = self.job(llm, src=src).run()
        words = pymupdf.open(result.mono)[0].get_text("words")
        right = sorted((w[4], round(w[1])) for w in words if w[0] > 480)
        self.assertEqual(sorted(n for n, _ in right), sorted(entries))  # page numbers stay in their column
        lines = {round(w[1]) for w in words if 90 < w[0] < 480 and w[1] > 110}
        self.assertGreaterEqual(len(lines), len(entries))  # one line per entry, not one run-on paragraph
        self.assertNotIn("Switching", " ".join(w[4] for w in words))


if __name__ == "__main__":
    unittest.main()
