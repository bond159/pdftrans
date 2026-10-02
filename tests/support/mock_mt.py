"""Local stand-ins for the free Google Translate and Microsoft Translator endpoints."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from tests.support.mock_llm import fake_chinese


class MockMT:
    """``with MockMT() as mt:`` then point pdftrans.mt.GOOGLE_URL etc. at ``mt.google_url`` ..."""

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
                if url.path == "/translate/auth":
                    self._send(200, mock.token, "text/plain")
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
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.headers.get("Authorization") != f"Bearer {mock.token}":
                    self._send(401, "expired", "text/plain")
                    return
                if self._throttled():
                    return
                out = [{"translations": [{"text": fake_chinese(i["Text"], mock.drop_numbers), "to": "zh-Hans"}]} for i in body]
                self._send(200, json.dumps(out, ensure_ascii=False))

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        base = f"http://127.0.0.1:{self.server.server_port}"
        self.google_url = base + "/translate_a/single"
        self.microsoft_auth_url = base + "/translate/auth"
        self.microsoft_url = base + "/translate"

    def patch(self):
        """Point pdftrans.mt at this server; returns a function that undoes it."""
        import pdftrans.mt as mt

        old = (mt.GOOGLE_URL, mt.MICROSOFT_URL, mt.MICROSOFT_AUTH_URL)
        mt.GOOGLE_URL, mt.MICROSOFT_URL, mt.MICROSOFT_AUTH_URL = self.google_url, self.microsoft_url, self.microsoft_auth_url

        def undo():
            mt.GOOGLE_URL, mt.MICROSOFT_URL, mt.MICROSOFT_AUTH_URL = old

        return undo

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
