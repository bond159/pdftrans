"""Unit tests for the table-of-contents detection in pdftrans.layout_fixes."""

import unittest
from types import SimpleNamespace as NS

from pdftrans import layout_fixes as lf


def line(*pieces, size=10.0, y=100.0):
    """Characters for a printed line; pieces are (x, text), each character 5 pt wide."""
    chars = []
    for x, text in pieces:
        for i, ch in enumerate(text):
            x0 = x + 5 * i
            box = NS(x=x0, x2=x0 + 5, y=y, y2=y + size * 0.7)
            chars.append(NS(char_unicode=ch, visual_bbox=NS(box=box), pdf_style=NS(font_size=size), vertical=False,
                            box=NS(x=x0, x2=x0 + 5, y=y, y2=y + size)))
    return chars


class DetectionTests(unittest.TestCase):
    def test_detached_page_number(self):
        chars = line((100, "1.1"), (130, "What Is the Internet?"), (450, "12"))
        cut = lf.trailing_page_number(chars)
        self.assertEqual(lf._text(chars[:cut]), "1.1What Is the Internet?")
        self.assertEqual(lf._text(chars[lf.leading_number(chars):cut]), "What Is the Internet?")

    def test_roman_page_number(self):
        self.assertIsNotNone(lf.trailing_page_number(line((100, "Preface"), (450, "xiii"))))

    def test_running_text_is_left_alone(self):
        chars = line((100, "In 2012 the network carried 12"))
        self.assertIsNone(lf.trailing_page_number(chars))
        self.assertIsNone(lf.leading_number(line((100, "2012 was a year of growth"))))

    def test_number_alone_is_not_an_entry(self):
        self.assertIsNone(lf.trailing_page_number(line((100, "1.1"), (450, "12"))))

    def test_printed_lines_are_separated(self):
        chars = line((100, "1.1"), (130, "Title"), (450, "2"), y=100) + line((100, "1.2"), (130, "Next"), (450, "5"), y=88)
        printed = lf._visual_lines(chars)
        self.assertEqual([lf._text(p) for p in printed], ["1.1Title2", "1.2Next5"])


if __name__ == "__main__":
    unittest.main()
