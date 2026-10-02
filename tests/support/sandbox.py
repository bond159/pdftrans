"""Test-only stand-ins that let BabelDOC run without downloading its assets.

BabelDOC normally downloads a DocLayout-YOLO ONNX model and a tiktoken
vocabulary. Tests (and CI sandboxes without access to HuggingFace) use these
stand-ins instead: a heuristic layout detector built on PyMuPDF, and a crude
tokenizer that only has to produce plausible token counts.
"""

from __future__ import annotations

import re

import numpy as np
import pymupdf
from babeldoc.docvision.base_doclayout import DocLayoutModel, YoloBox, YoloResult

# Class names used by the real DocLayout-YOLO DocStructBench model.
NAMES = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}


class HeuristicLayoutModel(DocLayoutModel):
    """Labels text blocks, images and drawings with PyMuPDF instead of a neural network."""

    @property
    def stride(self) -> int:
        return 32

    def _boxes(self, page: pymupdf.Page) -> list[YoloBox]:
        boxes = []
        for b in page.get_text("dict")["blocks"]:
            x0, y0, x1, y1 = b["bbox"]
            if b["type"] == 1:
                cls = 3
            else:
                sizes = [s["size"] for line in b["lines"] for s in line["spans"]]
                text = "".join(s["text"] for line in b["lines"] for s in line["spans"]).strip()
                if not sizes or not text:
                    continue
                if text.lower().startswith(("figure", "fig.")):
                    cls = 4
                elif max(sizes) >= 13:
                    cls = 0
                else:
                    cls = 1
            boxes.append(YoloBox(xyxy=np.array([x0, y0, x1, y1]), conf=np.float32(0.9), cls=cls))
        for d in page.get_drawings():
            r = d["rect"]
            if r.width > 40 and r.height > 40:
                boxes.append(YoloBox(xyxy=np.array([r.x0, r.y0, r.x1, r.y1]), conf=np.float32(0.8), cls=3))
        return boxes

    def handle_document(self, pages, mupdf_doc, translate_config, save_debug_image):
        for page in pages:
            translate_config.raise_if_cancelled()
            yield page, YoloResult(names=NAMES, boxes=self._boxes(mupdf_doc[page.page_number]))


class _Encoding:
    def encode(self, text, *args, **kwargs):
        # One token per CJK character, per Latin word, per number, per symbol.
        return re.findall(r"[\u3000-\u9fff\uff00-\uffef]|[A-Za-z]+|\d+|[^\w\s]", text)

    def decode(self, tokens, *args, **kwargs):
        return " ".join(tokens)


def patch_tiktoken() -> None:
    import tiktoken

    tiktoken.encoding_for_model = lambda *a, **k: _Encoding()
    tiktoken.get_encoding = lambda *a, **k: _Encoding()
