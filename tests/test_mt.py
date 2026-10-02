import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pdftrans.mt as mt
from pdftrans.config import Settings, output_paths
from pdftrans.store import Store
from pdftrans.translator import Journal, MachineTranslator
from tests.support.mock_llm import MockLLM, fake_chinese
from tests.support.mock_mt import MockMT

TEXT = "We trained the model for 300 epochs on 8 GPUs and reached 76.5% accuracy."


def no_proxy_env():
    return mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})


class SplitTests(unittest.TestCase):
    def test_short_text_is_one_piece(self):
        self.assertEqual(mt.split_text("Hello.", 100), ["Hello."])

    def test_splits_on_sentences_under_the_limit(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        pieces = mt.split_text(text, 120)
        self.assertTrue(all(len(p) <= 120 for p in pieces))
        self.assertEqual(" ".join(pieces), text)

    def test_hard_cut_for_huge_sentences(self):
        pieces = mt.split_text("x" * 250, 100)
        self.assertEqual([len(p) for p in pieces], [100, 100, 50])

    def test_explicit_proxy_wins(self):
        self.assertEqual(mt.resolve_proxy(" http://127.0.0.1:7890 "), "http://127.0.0.1:7890")


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.env = no_proxy_env()
        self.env.start()
        self.sleep = mock.patch("pdftrans.mt.time.sleep")
        self.sleep.start()

    def tearDown(self):
        self.env.stop()
        self.sleep.stop()

    def test_google(self):
        with MockMT() as server:
            client = mt.GoogleClient(url=server.google_url)
            self.assertEqual(client.translate(TEXT), fake_chinese(TEXT))

    def test_google_retries_after_429(self):
        with MockMT() as server:
            server.fail_429 = 2
            self.assertEqual(mt.GoogleClient(url=server.google_url).translate(TEXT), fake_chinese(TEXT))

    def test_google_gives_up_with_a_clear_error(self):
        with MockMT() as server:
            server.fail_429 = 10
            with self.assertRaises(mt.MTError) as ctx:
                mt.GoogleClient(url=server.google_url).translate(TEXT)
            self.assertIn("谷歌翻译", str(ctx.exception))

    def test_microsoft_refreshes_expired_token(self):
        with MockMT() as server:
            client = mt.MicrosoftClient(url=server.microsoft_url, auth_url=server.microsoft_auth_url)
            self.assertEqual(client.translate(TEXT), fake_chinese(TEXT))
            server.token = "token-2"  # the old token is now rejected
            self.assertEqual(client.translate("Hello there"), fake_chinese("Hello there"))
            self.assertEqual([r for r in server.requests if r[1] == "/translate/auth"].__len__(), 2)

    def test_long_text_is_split(self):
        with MockMT() as server:
            text = " ".join(["This is a sentence about translation."] * 120)
            out = mt.GoogleClient(url=server.google_url).translate(text)
            self.assertGreater(len([r for r in server.requests if r[1] == "/translate_a/single"]), 1)
            self.assertNotIn("sentence", out)


class MachineTranslatorTests(unittest.TestCase):
    def setUp(self):
        self.env = no_proxy_env()
        self.env.start()
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = Store(Path(self.tmp.name) / "s.db")

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def make(self, server, **kw):
        st = Settings(engine="google", **kw)
        tr = MachineTranslator(st, self.store, Journal(), client=mt.GoogleClient(url=server.google_url))
        tr.ignore_cache = True
        return tr

    def test_babeldoc_sees_a_non_llm_translator(self):
        from babeldoc.format.pdf.high_level import translator_supports_llm

        with MockMT() as server:
            self.assertFalse(translator_supports_llm(self.make(server, api_key="")))

    def test_rule_problems_are_reported_without_an_llm(self):
        with MockMT() as server:
            server.drop_numbers = True
            tr = self.make(server, api_key="")
            out = tr.translate(TEXT)
        self.assertNotIn("300", out)
        record = tr.journal.all()[0]
        self.assertTrue(any("数字" in i for i in record.issues))

    def test_llm_reviews_machine_translation_when_configured(self):
        with MockMT() as server, MockLLM() as llm:
            server.drop_numbers = True
            tr = self.make(server, api_key="k", base_url=llm.url, review_model="qwen-max")
            out = tr.translate(TEXT)
        self.assertEqual(out, fake_chinese(TEXT))
        record = tr.journal.all()[0]
        self.assertTrue(record.corrected)
        self.assertEqual(record.issues, [])

    def test_override(self):
        self.store.set_override(TEXT, "手动译文")
        with MockMT() as server:
            tr = self.make(server, api_key="")
            self.assertEqual(tr.translate(TEXT), "手动译文")
            self.assertEqual(server.requests, [])

    def test_preflight_explains_failures(self):
        with MockMT() as server:
            server.fail_429 = 100
            tr = self.make(server, api_key="")
            with mock.patch("pdftrans.mt.time.sleep"), self.assertRaises(RuntimeError) as ctx:
                tr.preflight()
            self.assertIn("代理", str(ctx.exception))


class OutputNameTests(unittest.TestCase):
    def test_engines_do_not_overwrite_each_other(self):
        names = {e: output_paths("/x/paper.pdf", "", e)["mono"].name for e in ("llm", "google", "microsoft")}
        self.assertEqual(names, {"llm": "paper.zh-CN.pdf", "google": "paper.zh-CN.google.pdf", "microsoft": "paper.zh-CN.microsoft.pdf"})
        self.assertEqual(output_paths("/x/paper.pdf", "/out", "google")["dual"], Path("/out/paper.zh-CN.google.dual.pdf"))


@unittest.skipIf(os.environ.get("PDFTRANS_SKIP_E2E"), "end-to-end tests disabled")
class MachineEngineTests(unittest.TestCase):
    def test_google_end_to_end(self):
        import pymupdf

        from pdftrans.engine import Job
        from tests.support.sandbox import HeuristicLayoutModel, patch_tiktoken
        from tests.test_engine import make_pdf

        patch_tiktoken()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, no_proxy_env(), MockMT() as server:
            undo = server.patch()
            try:
                src = Path(tmp) / "paper.pdf"
                make_pdf(src)
                st = Settings(engine="google", api_key="", output_dir=str(Path(tmp) / "out"))
                job = Job(src, st, store=Store(Path(tmp) / "s.db"), layout=HeuristicLayoutModel(), skip_assets=True)
                result = job.run()
            finally:
                undo()
            self.assertEqual(result.mono.name, "paper.zh-CN.google.pdf")
            text = pymupdf.open(result.mono)[0].get_text()
            self.assertIn(fake_chinese("Overview"), text)
            self.assertNotIn("remarkable", text)


if __name__ == "__main__":
    unittest.main()
