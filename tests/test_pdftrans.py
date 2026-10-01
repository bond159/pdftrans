import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pymupdf

from pdftrans.config import Settings, parse_pages
from pdftrans.layout import TextUnit, break_lines, extract_units, join_lines, should_translate
from pdftrans.llm import (
    Backend,
    OpenAICompatibleBackend,
    TranslationCache,
    TranslationError,
    Translator,
    parse_json_reply,
)
from pdftrans.pipeline import Cancelled, make_batches, translate_pdf


PARAGRAPH = (
    "Large language models have shown remarkable ability in a wide range of natural "
    "language tasks. However, translating documents while preserving their layout remains "
    "challenging, because the text in a PDF is stored as positioned glyphs."
)


class FakeBackend(Backend):
    """Translates by tagging the text, answering batches in the JSON format the prompt asks for."""

    def __init__(self, drop_key=None):
        self.calls = []
        self.drop_key = drop_key

    def complete(self, system, user):
        self.calls.append(user)
        if user.startswith("Translate the values"):
            data = json.loads(user[user.index("{") :])
            out = {k: "译文" + v[:20] for k, v in data.items() if k != self.drop_key}
            return "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```"
        return "译文" + user[:20]


def make_pdf(path, pages=2):
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(pymupdf.Rect(72, 60, 540, 100), f"Chapter {n + 1} Overview", fontname="tibo", fontsize=16)
        page.insert_textbox(pymupdf.Rect(72, 110, 540, 220), PARAGRAPH, fontname="tiro", fontsize=11)
        page.insert_textbox(pymupdf.Rect(72, 240, 540, 280), "x = 3.14 + 2", fontname="tiro", fontsize=11)
        page.insert_text((300, 760), str(n + 1), fontname="tiro", fontsize=9)
    doc.save(path)


class TextTests(unittest.TestCase):
    def test_parse_pages(self):
        self.assertEqual(parse_pages("", 3), [0, 1, 2])
        self.assertEqual(parse_pages("1-2, 5", 10), [0, 1, 4])
        self.assertEqual(parse_pages("8-", 9), [7, 8])
        self.assertEqual(parse_pages("2-99", 3), [1, 2])
        with self.assertRaises(ValueError):
            parse_pages("3-1", 5)

    def test_join_lines_dehyphenates_and_handles_cjk(self):
        self.assertEqual(join_lines(["the trans-", "lation works"]), "the translation works")
        self.assertEqual(join_lines(["state-of-", "The art"]), "state-of- The art")
        self.assertEqual(join_lines(["中文第一行", "第二行"]), "中文第一行第二行")

    def test_parse_json_reply_tolerates_fences_and_chatter(self):
        self.assertEqual(parse_json_reply('Sure!\n```json\n{"1": "a"}\n```'), {"1": "a"})
        with self.assertRaises(ValueError):
            parse_json_reply("no json here")

    def test_break_lines_keeps_punctuation_off_line_start(self):
        text = "这是一个测试，用于检查标点。" * 6
        lines = break_lines(text, 60, 10, lambda atom: len(atom))
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(sum(len(a) for a in line) * 10, 60 + 10)
            self.assertNotIn(line[0][0], "，。")
        self.assertEqual("".join("".join(line) for line in lines), text)

    def test_break_lines_wraps_latin_on_spaces(self):
        lines = break_lines("alpha beta gamma delta", 11, 1, lambda atom: len(atom))
        self.assertEqual(["".join(line) for line in lines], ["alpha beta", "gamma delta"])

    def test_should_translate(self):
        def unit(text, math=0.0):
            return TextUnit(page=0, bbox=(0, 0, 10, 10), text=text, size=10, color=0, bold=False, sans=False, math_ratio=math)

        self.assertTrue(should_translate(unit("Introduction"), "zh-CN"))
        self.assertFalse(should_translate(unit("12"), "zh-CN"))
        self.assertFalse(should_translate(unit("x = 3.14 + 2.71 * 10"), "zh-CN"))
        self.assertFalse(should_translate(unit("https://example.org/a"), "zh-CN"))
        self.assertFalse(should_translate(unit("这已经是中文了"), "zh-CN"))
        self.assertTrue(should_translate(unit("这已经是中文了"), "en"))
        self.assertFalse(should_translate(unit("f(x) dx", math=0.9), "zh-CN"))

    def test_make_batches(self):
        batches = make_batches(["a" * 10] * 5, max_chars=25)
        self.assertEqual([len(b) for b in batches], [2, 2, 1])
        self.assertEqual(make_batches(["a" * 100], max_chars=10), [["a" * 100]])


class TranslatorTests(unittest.TestCase):
    def test_batch_falls_back_per_item_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend(drop_key="2")
            cache = TranslationCache(Path(tmp) / "c.db")
            tr = Translator(backend, Settings(), cache)
            out = tr.translate_batch(["one", "two", "three"])
            self.assertEqual(out, ["译文one", "译文two", "译文three"])
            self.assertEqual(len(backend.calls), 2)  # the batch, then "two" on its own
            self.assertEqual(tr.cached("two"), "译文two")
            other = Translator(backend, Settings(target_lang="ja"), cache)
            self.assertIsNone(other.cached("two"))  # cache keys include the target language


class _Handler(BaseHTTPRequestHandler):
    requests = []
    fail_first = 0
    status = 200

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, self.headers.get("Authorization"), body))
        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            return
        if type(self).fail_first > 0:
            type(self).fail_first -= 1
            self.send_response(503)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        reply = {"choices": [{"message": {"content": "你好，世界"}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class OpenAIBackendTests(unittest.TestCase):
    def setUp(self):
        _Handler.requests = []
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.env = {k: os.environ.pop(k) for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY") if k in os.environ}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        os.environ.update(self.env)

    def test_request_shape_and_retry(self):
        _Handler.fail_first = 1
        backend = OpenAICompatibleBackend(self.url, "sk-test", "demo-model")
        import pdftrans.llm as llm

        sleep, llm.time.sleep = llm.time.sleep, lambda s: None
        try:
            self.assertEqual(backend.complete("sys", "hello"), "你好，世界")
        finally:
            llm.time.sleep = sleep
        self.assertEqual(len(_Handler.requests), 2)
        path, auth, body = _Handler.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(auth, "Bearer sk-test")
        self.assertEqual(body["model"], "demo-model")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])

    def test_client_error_is_not_retried(self):
        _Handler.fail_first = 0
        _Handler.status = 401
        try:
            with self.assertRaises(TranslationError):
                OpenAICompatibleBackend(self.url, "bad-key", "m").complete("s", "u")
        finally:
            _Handler.status = 200
        self.assertEqual(len(_Handler.requests), 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "paper.pdf"
        make_pdf(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def run_job(self, **kw):
        settings = Settings(output_dir=str(self.dir / "out"), **kw)
        return translate_pdf(self.src, settings, Translator(FakeBackend(), settings))

    def test_extract_units(self):
        doc = pymupdf.open(self.src)
        texts = [u.text for u in extract_units(doc[0], 0)]
        self.assertIn("Chapter 1 Overview", texts)
        self.assertIn(PARAGRAPH, texts)

    def test_outputs_replace_text_and_keep_layout(self):
        result = self.run_job(modes=["mono", "dual", "alt"])
        mono = pymupdf.open(result.outputs["mono"])
        text = mono[0].get_text()
        self.assertIn("译文Chapter 1 Overview", text)
        self.assertNotIn("remarkable ability", text)
        self.assertIn("3.14", text)  # formula left untouched
        self.assertEqual(mono.page_count, 2)
        dual = pymupdf.open(result.outputs["dual"])
        self.assertEqual(dual.page_count, 2)
        self.assertAlmostEqual(dual[0].rect.width, 2 * 612, places=0)
        self.assertEqual(pymupdf.open(result.outputs["alt"]).page_count, 4)

    def test_page_range(self):
        result = self.run_job(modes=["mono"], pages="2")
        mono = pymupdf.open(result.outputs["mono"])
        self.assertEqual(mono.page_count, 1)
        self.assertIn("译文Chapter 2 Overview", mono[0].get_text())

    def test_cancel(self):
        settings = Settings(output_dir=str(self.dir / "out"))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            translate_pdf(self.src, settings, Translator(FakeBackend(), settings), cancel=cancel)


if __name__ == "__main__":
    unittest.main()
