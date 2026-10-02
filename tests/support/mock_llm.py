"""A local stand-in for an OpenAI-compatible chat API (DashScope compatible mode).

It "translates" by replacing English words with Chinese characters while keeping
placeholders, tags and numbers, which is enough to exercise BabelDOC and the
proofreading pipeline end to end without a real model.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from pdftrans import proofread as pr

POOL = "模型方法数据结果实验研究分析性能训练系统文本翻译版面公式图像网络学习任务"


def fake_chinese(text: str, drop_numbers: bool = False) -> str:
    parts = re.split(f"({pr.PLACEHOLDER_RE.pattern})", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)  # placeholder or tag, keep as is
            continue
        part = re.sub(r"[A-Za-z][A-Za-z'’-]*", lambda m: POOL[len(m.group()) % len(POOL)] + POOL[(len(m.group()) * 7) % len(POOL)], part)
        if drop_numbers:
            part = re.sub(r"\d+", "", part)
        part = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", part)
        out.append(part)
    return "".join(out).strip()


class MockLLM:
    """Start with ``with MockLLM() as llm:``; ``llm.url`` is the base URL, ``llm.requests`` the bodies."""

    def __init__(self, sloppy_numbers: bool = True):
        self.requests: list[dict] = []
        self.sloppy_numbers = sloppy_numbers  # the "translation model" drops numbers; the reviewer fixes them
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append(body)
                content = mock.answer(body)
                reply = {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", ""),
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
                data = json.dumps(reply, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def answer(self, body: dict) -> str:
        user = body["messages"][-1]["content"]
        if "待审校内容" in user:
            items = json.loads(user.split("待审校内容：", 1)[1])
            out = []
            for item in items:
                fixed = fake_chinese(item["source"])
                if fixed != item["translation"]:
                    out.append({"id": item["id"], "verdict": "fixed", "problems": ["数字遗漏"], "output": fixed})
                else:
                    out.append({"id": item["id"], "verdict": "ok", "problems": [], "output": item["translation"]})
            return json.dumps(out, ensure_ascii=False)
        batch = pr.parse_batch_prompt(user)
        if batch is not None:
            return json.dumps(
                [{"id": i["id"], "output": fake_chinese(i["input"], self.sloppy_numbers)} for i in batch.items],
                ensure_ascii=False,
            )
        single = pr.parse_single_prompt(user)
        if single is not None:
            return fake_chinese(single, self.sloppy_numbers)
        if "Translate the following text into Simplified Chinese" in user:
            return fake_chinese(user.rsplit("\n\n", 1)[-1])
        return "[]"  # term extraction: no terms

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
