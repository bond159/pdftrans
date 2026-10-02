"""Local stand-ins for the free Google Translate and Microsoft Translator endpoints."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from tests.support.mock_llm import fake_chinese


class MockMT:
    """``with MockMT() as mt:``; ``mt.patch()`` points pdftrans.mt at it."""

    def __init__(self):
        self.requests: list[tuple[str, str]] = []
        self.fail_429 = 0  # answer this many requests with 429 first
        self.token = "token-1"
        self.drop_numbers = False
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, body, content_type="application/json"):
                data = body.encode() if isinstance(body, str) else body
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _throttled(self):
                if mock.fail_429 > 0:
                    mock.fail_429 -= 1
                    self._send(429, "slow down", "text/plain")
                    return True
                return False

            def do_GET(self):  # noqa: N802
                url = urlparse(self.path)
                mock.requests.append(("GET", url.path))
                if url.path == "/translator":
                    page = (
                        '<html><script>var _G={IG:"IGVALUE"}; _G["ig":"IGVALUE"];</script>'
                        '<div data-iid="translator.5023"></div><div data-iid="translator.5028"></div>'
                        f'<script>var params_AbusePreventionHelper = [1700000000000,"{mock.token}",3600000];</script></html>'
                    )
                    page = page.replace('_G["ig":"IGVALUE"]', '{"ig":"IGVALUE"}')
                    self._send(200, page, "text/html")
                elif url.path == "/translate_a/single":
                    if self._throttled():
                        return
                    q = parse_qs(url.query)["q"][0]
                    zh = fake_chinese(q, mock.drop_numbers)
                    half = len(zh) // 2  # Google answers in segments
                    self._send(200, json.dumps([[[zh[:half], q[:10], None, None], [zh[half:], q[10:], None, None]], None, "en"], ensure_ascii=False))
                else:
                    self._send(404, "not found", "text/plain")

            def do_POST(self):  # noqa: N802
                url = urlparse(self.path)
                mock.requests.append(("POST", url.path))
                form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
                query = parse_qs(url.query)
                if url.path != "/ttranslatev3" or query.get("IG") != ["IGVALUE"] or query.get("IID") != ["translator.5028"]:
                    self._send(404, "not found", "text/plain")
                    return
                if form.get("token") != [mock.token]:
                    self._send(200, '{"statusCode": 205}')  # what Bing answers to a stale token
                    return
                if self._throttled():
                    return
                text = form["text"][0]
                if len(text) > 1000:
                    self._send(400, "too long", "text/plain")
                    return
                out = [{"translations": [{"text": fake_chinese(text, mock.drop_numbers), "to": "zh-Hans"}]}]
                self._send(200, json.dumps(out, ensure_ascii=False))

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        base = f"http://127.0.0.1:{self.server.server_port}"
        self.google_url = base + "/translate_a/single"
        self.bing_url = base + "/translator"

    def patch(self):
        """Point pdftrans.mt at this server; returns a function that undoes it."""
        import pdftrans.mt as mt

        old = (mt.GOOGLE_URL, mt.BING_URL)
        mt.GOOGLE_URL, mt.BING_URL = self.google_url, self.bing_url

        def undo():
            mt.GOOGLE_URL, mt.BING_URL = old

        return undo

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
