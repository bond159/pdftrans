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

    def job(self, llm, **kw):
        from pdftrans.engine import Job

        # A unique model name keeps BabelDOC's on-disk translation cache from leaking between runs.
        st = Settings(base_url=llm.url, api_key="k", model=f"qwen-test-{uuid.uuid4().hex[:8]}",
                      output_dir=str(self.dir / "out"), auto_glossary=False, **kw)
        return Job(self.src, st, store=Store(self.dir / "s.db"), layout=HeuristicLayoutModel(), skip_assets=True)

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


if __name__ == "__main__":
    unittest.main()
