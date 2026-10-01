"""LLM backends that turn a batch of paragraphs into translations."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

import httpx

from .config import CONFIG_DIR, LANGUAGES, Settings

PROMPT_VERSION = "1"

SYSTEM_PROMPT = """You are a professional translator working on text extracted from a PDF document.
Translate every item into {lang}.

Rules:
- Translate faithfully and fluently, using the terminology of the document's field.
- Keep math, formulas, variables, citations like [12] or (Smith et al., 2020), numbers, URLs, emails and code exactly as they are.
- The text was extracted from a PDF, so it may contain broken line breaks or hyphenation; produce clean, continuous text.
- If an item is already in {lang} or should not be translated (a name, an identifier), return it unchanged.
- Output only the translation, no notes or explanations.{extra}"""

BATCH_INSTRUCTIONS = """Translate the values of this JSON object. Reply with a JSON object that has exactly the same keys, each mapped to its translation, and nothing else.

{payload}"""


class TranslationError(RuntimeError):
    pass


def build_system_prompt(settings: Settings) -> str:
    lang = LANGUAGES.get(settings.target_lang, (settings.target_lang, settings.target_lang))[1]
    extra = ""
    if settings.extra_prompt.strip():
        extra = "\n\nAdditional instructions from the user (glossary, style):\n" + settings.extra_prompt.strip()
    return SYSTEM_PROMPT.format(lang=lang, extra=extra)


def parse_json_reply(reply: str) -> dict:
    """Pull the JSON object out of a model reply, tolerating code fences and chatter."""
    text = reply.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("reply contains no JSON object")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data


class Backend:
    """A chat model that answers one system + user prompt with text."""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAICompatibleBackend(Backend):
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.3, timeout: float = 180):
        if not base_url:
            raise TranslationError("请填写 Base URL")
        if not model:
            raise TranslationError("请填写模型名称")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(headers=headers, timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        delay = 2.0
        for attempt in range(4):
            try:
                resp = self.client.post(self.url, json=body)
            except httpx.HTTPError as e:
                if attempt == 3:
                    raise TranslationError(f"网络错误: {e}") from e
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()["choices"][0]["message"]["content"] or ""
                    except (ValueError, KeyError, IndexError, TypeError) as e:
                        raise TranslationError(f"无法解析接口返回: {resp.text[:300]}") from e
                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable or attempt == 3:
                    raise TranslationError(f"接口错误 HTTP {resp.status_code}: {resp.text[:300]}")
                retry_after = resp.headers.get("retry-after", "")
                if retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            time.sleep(delay)
            delay *= 2
        raise TranslationError("unreachable")


class AnthropicBackend(Backend):
    def __init__(self, api_key: str, model: str, effort: str = "low", base_url: str = ""):
        import anthropic

        self.anthropic = anthropic
        kwargs = {"max_retries": 4}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model or "claude-opus-5-5"
        self.effort = effort
        # Server-side refusal fallback is only offered by Anthropic's own API,
        # not by relays configured through a custom base URL.
        self.use_fallbacks = not base_url

    def complete(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            if self.use_fallbacks:
                resp = self.client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = self.client.messages.create(**kwargs)
        except self.anthropic.AuthenticationError as e:
            raise TranslationError("API Key 无效") from e
        except self.anthropic.NotFoundError as e:
            raise TranslationError(f"模型或接口不存在: {e.message}") from e
        except self.anthropic.APIStatusError as e:
            raise TranslationError(f"接口错误 HTTP {e.status_code}: {e.message}") from e
        except self.anthropic.APIConnectionError as e:
            raise TranslationError(f"网络错误: {e}") from e
        if resp.stop_reason == "refusal":
            raise TranslationError("模型拒绝了这段内容的翻译请求")
        return "".join(block.text for block in resp.content if block.type == "text")


def make_backend(settings: Settings) -> Backend:
    key = settings.resolved_api_key()
    if settings.provider == "anthropic":
        return AnthropicBackend(key, settings.model, settings.effort, settings.base_url)
    return OpenAICompatibleBackend(settings.base_url, key, settings.model, settings.temperature)


class TranslationCache:
    """A small sqlite cache so re-running a document doesn't pay for the same text twice."""

    def __init__(self, path: Path = CONFIG_DIR / "cache.sqlite3"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self.db.commit()

    def get(self, key: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT v FROM t WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO t (k, v) VALUES (?, ?)", (key, value))
            self.db.commit()


class Translator:
    """Translates batches of paragraphs through a backend, with caching and per-item fallback."""

    def __init__(self, backend: Backend, settings: Settings, cache: TranslationCache | None = None):
        self.backend = backend
        self.system = build_system_prompt(settings)
        self.cache = cache
        ident = "|".join([PROMPT_VERSION, settings.provider, settings.base_url, settings.model, self.system])
        self.cache_ns = hashlib.sha256(ident.encode("utf-8")).hexdigest()[:16]

    def _key(self, text: str) -> str:
        return self.cache_ns + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def cached(self, text: str) -> str | None:
        return self.cache.get(self._key(text)) if self.cache else None

    def translate_one(self, text: str) -> str:
        reply = self.backend.complete(self.system, text).strip()
        return reply or text

    def translate_batch(self, texts: list[str]) -> list[str]:
        if len(texts) == 1:
            results = [self.translate_one(texts[0])]
        else:
            payload = json.dumps({str(i + 1): t for i, t in enumerate(texts)}, ensure_ascii=False, indent=1)
            reply = self.backend.complete(self.system, BATCH_INSTRUCTIONS.format(payload=payload))
            try:
                data = parse_json_reply(reply)
            except ValueError:
                data = {}
            results = []
            for i, text in enumerate(texts):
                value = data.get(str(i + 1))
                if not isinstance(value, str) or not value.strip():
                    # The model dropped or mangled this item; ask for it on its own.
                    value = self.translate_one(text)
                results.append(value.strip())
        if self.cache:
            for text, result in zip(texts, results):
                self.cache.put(self._key(text), result)
        return results

    def test(self) -> str:
        return self.translate_one("Hello, world! This is a connection test.")
