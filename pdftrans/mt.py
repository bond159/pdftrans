"""Free machine translation services: Google Translate and Microsoft Translator (Bing).

Both use the public endpoints that browser extensions use, need no API key, and
may rate-limit heavy use. Google is not reachable from mainland China without a
proxy; Microsoft usually is.
"""

from __future__ import annotations

import os
import re
import threading
import time
import urllib.request

import httpx

GOOGLE_URL = "https://translate.googleapis.com/translate_a/single"
BING_URL = "https://www.bing.com/translator"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
)


class MTError(RuntimeError):
    pass


def _env_proxy_set() -> bool:
    return any(os.environ.get(k) for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"))


def resolve_proxy(proxy: str = "") -> str | None:
    """The proxy to use: the one typed in the settings, else the system proxy.

    *_PROXY environment variables (with NO_PROXY) are applied by httpx itself, so
    this only looks up the Windows / macOS proxy settings when none are set.
    """
    if proxy.strip():
        return proxy.strip()
    if _env_proxy_set():
        return None
    found = urllib.request.getproxies()
    return found.get("https") or found.get("http") or None


def split_text(text: str, limit: int) -> list[str]:
    """Split long text at sentence ends so every piece stays under the service's size limit."""
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        while len(sentence) > limit:  # one enormous "sentence": cut it hard
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


class Client:
    name = ""
    limit = 4000

    def __init__(self, proxy: str = "", timeout: float = 30):
        self.http = httpx.Client(
            proxy=resolve_proxy(proxy),
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        parts = [self._with_retries(p) for p in split_text(text, self.limit)]
        return "".join(parts) if self.joins_without_space else " ".join(parts)

    joins_without_space = True  # target is Chinese

    def _with_retries(self, text: str) -> str:
        delay = 1.0
        for attempt in range(4):
            try:
                return self._translate(text)
            except MTError:
                raise
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
                if attempt == 3:
                    raise MTError(f"{self.name}请求失败：{e}") from e
            time.sleep(delay)
            delay *= 2
        raise MTError("unreachable")

    def _check(self, resp: httpx.Response) -> None:
        if resp.status_code == 429:
            raise httpx.HTTPStatusError("429 Too Many Requests", request=resp.request, response=resp)
        resp.raise_for_status()

    def _translate(self, text: str) -> str:
        raise NotImplementedError


class GoogleClient(Client):
    name = "谷歌翻译"
    limit = 2000  # text goes in the URL

    def __init__(self, proxy: str = "", url: str = GOOGLE_URL, **kw):
        super().__init__(proxy, **kw)
        self.url = url

    def _translate(self, text: str) -> str:
        resp = self.http.get(
            self.url,
            params={"client": "gtx", "sl": "en", "tl": "zh-CN", "dt": "t", "ie": "UTF-8", "oe": "UTF-8", "q": text},
        )
        self._check(resp)
        data = resp.json()
        return "".join(seg[0] for seg in data[0] if seg and seg[0])


class MicrosoftClient(Client):
    """Microsoft Translator through the Bing Translator web page (bing.com/translator).

    The page hands out a short-lived token; it is fetched once and renewed when it
    expires or a request is refused. In mainland China bing.com redirects to
    cn.bing.com, which works the same way.
    """

    name = "微软翻译"
    limit = 1000  # the web endpoint accepts at most 1000 characters

    def __init__(self, proxy: str = "", url: str = BING_URL, **kw):
        super().__init__(proxy, **kw)
        self.url = url
        self.lock = threading.Lock()
        self.session: dict | None = None
        self.session_time = 0.0

    def _session(self, refresh: bool = False) -> dict:
        with self.lock:
            if refresh or self.session is None or time.time() - self.session_time > 600:
                resp = self.http.get(self.url)
                resp.raise_for_status()
                page = resp.text
                ig = re.search(r'"ig"\s*:\s*"([^"]+)"', page, re.I) or re.search(r"IG:\s*\"([^\"]+)\"", page)
                iid = re.findall(r'data-iid="([^"]+)"', page)
                abuse = re.search(r"params_AbusePreventionHelper\s*=\s*\[(\d+),\s*\"([^\"]+)\"", page)
                if not (ig and iid and abuse):
                    raise MTError("微软翻译页面格式已变化，无法获取令牌")
                base = str(resp.url).split("/translator")[0]  # www.bing.com or cn.bing.com
                self.session = {
                    "endpoint": f"{base}/ttranslatev3?isVertical=1&IG={ig.group(1)}&IID={iid[-1]}",
                    "key": abuse.group(1),
                    "token": abuse.group(2),
                }
                self.session_time = time.time()
            return self.session

    def _translate(self, text: str) -> str:
        for refresh in (False, True):
            sess = self._session(refresh)
            resp = self.http.post(
                sess["endpoint"],
                data={"fromLang": "en", "to": "zh-Hans", "text": text, "token": sess["token"], "key": sess["key"]},
                headers={"Referer": self.url},
            )
            if resp.status_code in (400, 401, 403) or not resp.text.strip().startswith("["):
                if not refresh:
                    continue  # stale token or session: get a new one and retry once
                if resp.status_code == 429:
                    self._check(resp)
                raise MTError(f"微软翻译拒绝了请求（HTTP {resp.status_code}）")
            self._check(resp)
            return resp.json()[0]["translations"][0]["text"]
        raise MTError("微软翻译认证失败")


def make_client(engine: str, proxy: str = "") -> Client:
    if engine == "google":
        return GoogleClient(proxy, url=GOOGLE_URL)
    if engine == "microsoft":
        return MicrosoftClient(proxy, url=BING_URL)
    raise ValueError(f"unknown engine {engine}")
