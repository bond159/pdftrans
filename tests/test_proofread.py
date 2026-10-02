import json
import unittest

from pdftrans import proofread as pr

SRC = "We train {v1} on <style id='2'>ImageNet</style> for 300 epochs, reaching 76.5% accuracy [12]."
GOOD = "我们在<style id='2'>ImageNet</style>上训练{v1}共 300 个周期，准确率达到 76.5% [12]。"


class CheckTests(unittest.TestCase):
    def test_good_translation_passes(self):
        self.assertTrue(pr.check_translation(SRC, GOOD).ok)

    def test_lost_placeholder(self):
        check = pr.check_translation(SRC, GOOD.replace("{v1}", ""))
        self.assertTrue(check.broken_markup)

    def test_placeholder_spacing_is_tolerated(self):
        self.assertTrue(pr.check_translation(SRC, GOOD.replace("{v1}", "{ v1 }")).ok)

    def test_missing_number(self):
        check = pr.check_translation(SRC, GOOD.replace("300", "三百"))
        self.assertTrue(any("300" in i for i in check.issues))

    def test_thousands_separator_is_the_same_number(self):
        self.assertTrue(pr.check_translation("about 1,024 samples here", "约 1024 个样本").ok)

    def test_untranslated_english(self):
        out = "我们提出 a simple pipeline that extracts paragraphs 的方法。"
        src = "We propose a simple pipeline that extracts paragraphs from documents."
        check = pr.check_translation(src, out)
        self.assertTrue(any("未翻译" in i for i in check.issues))

    def test_glossary_terms_may_stay_english(self):
        src = "The Large Language Model Meta AI family is strong."
        out = "Large Language Model Meta AI 系列很强。"
        self.assertTrue(pr.check_translation(src, out, {"Large Language Model Meta AI"}).ok)

    def test_too_short(self):
        src = "This sentence is long enough to need a translation that keeps all of its meaning intact."
        self.assertTrue(any("过短" in i for i in pr.check_translation(src, "这个。").issues))

    def test_empty_and_no_chinese(self):
        self.assertIn("译文为空", pr.check_translation(SRC, "").issues)
        self.assertTrue(any("没有中文" in i for i in pr.check_translation("Hello there my good friend", "Hallo").issues))

    def test_short_labels_are_not_checked_for_english(self):
        self.assertTrue(pr.check_translation("Figure 3", "图 3").ok)


class PromptTests(unittest.TestCase):
    PROMPT = (
        "You are a translator.\n\n## Glossary Tables\n\n### Glossary: auto\n\n"
        "| Source Term | Target Term |\n|-------------|-------------|\n| LLM | 大语言模型 |\n| GPT | GPT |\n\n"
        "## Contextual Hints for Better Translation\n1. First title in full text: Attention\n\n"
        "## Here is the input:\n\n" + json.dumps([{"id": 0, "input": "Hello", "layout_label": "text"}])
    )

    def test_parse_batch_prompt(self):
        batch = pr.parse_batch_prompt(self.PROMPT)
        self.assertEqual(batch.items[0]["input"], "Hello")
        self.assertIn("LLM", batch.glossary)
        self.assertIn("Attention", batch.context)
        self.assertEqual(pr.glossary_targets(batch.glossary), {"大语言模型", "GPT"})

    def test_non_batch_prompts(self):
        self.assertIsNone(pr.parse_batch_prompt("extract terms please"))
        self.assertEqual(pr.parse_single_prompt("rules...\nNow translate the following text:\n\nHi"), "Hi")

    def test_batch_output_roundtrip(self):
        text = '```json\n[{"id": 1, "output": "乙"}, {"id": 0, "output": "甲"}]\n```'
        self.assertEqual(pr.parse_batch_output(text), {0: "甲", 1: "乙"})
        self.assertEqual(pr.parse_batch_output(pr.dump_batch_output({0: "甲"})), {0: "甲"})


class ReviewTests(unittest.TestCase):
    def test_parse_review(self):
        text = json.dumps([
            {"id": 0, "verdict": "ok", "problems": [], "output": "甲"},
            {"id": 1, "verdict": "fixed", "problems": "漏译", "output": "乙"},
        ], ensure_ascii=False)
        reviews = pr.parse_review(text)
        self.assertEqual(reviews[1].problems, ["漏译"])
        self.assertEqual(reviews[1].verdict, "fixed")

    def test_choose_accepts_good_fix(self):
        bad = GOOD.replace("300", "三百")
        final, problems = pr.choose(SRC, bad, pr.Review("fixed", ["数字错误"], GOOD), set())
        self.assertEqual(final, GOOD)
        self.assertEqual(problems, ["数字错误"])

    def test_choose_rejects_fix_that_breaks_markup(self):
        broken = GOOD.replace("{v1}", "")
        final, problems = pr.choose(SRC, GOOD, pr.Review("fixed", ["润色"], broken), set())
        self.assertEqual(final, GOOD)
        self.assertTrue(any("保留原译文" in p for p in problems))

    def test_choose_keeps_draft_when_ok(self):
        self.assertEqual(pr.choose(SRC, GOOD, pr.Review("ok", [], GOOD), set()), (GOOD, []))
        self.assertEqual(pr.choose(SRC, GOOD, None, set()), (GOOD, []))

    def test_review_prompt_contains_payload(self):
        prompt = pr.build_review_prompt([{"id": 0, "source": "a", "translation": "甲"}], "| a | 甲 |", "")
        self.assertIn("术语表", prompt)
        self.assertIn('"translation": "甲"', prompt)


if __name__ == "__main__":
    unittest.main()
