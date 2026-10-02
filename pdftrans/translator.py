"""BabelDOC translators with automatic proofreading and manual corrections.

Two translators plug into BabelDOC:

- ProofreadingTranslator: an LLM (Qwen or any OpenAI-compatible model). BabelDOC
  sends batches of paragraphs as one JSON prompt; the answer is checked by rules
  and reviewed by a second model before BabelDOC typesets it.
- MachineTranslator: Google Translate or Microsoft Translator. BabelDOC sends one
  paragraph at a time; the result is checked by rules, and reviewed by the LLM
  too when an API key is configured.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import openai
from babeldoc.translator.translator import BaseTranslator, OpenAITranslator

from . import proofread as pr
from .config import ENGINES, Settings
from .mt import Client, make_client
from .store import Store

logger = logging.getLogger(__name__)

PROOFREAD_VERSION = "1"

TRANSLATOR_ROLE = (
    "You are a professional translator of English academic papers and technical textbooks into "
    "Simplified Chinese (简体中文). Translate faithfully: never add, omit or summarize information. "
    "Use the standard Chinese terminology of the field and a formal academic register. "
    "Keep citations such as [12] or (Smith et al., 2020), numbers, variables, URLs, code and person "
    "names exactly as they are."
)

RETRANSLATE_USER = """Translate the following text into Simplified Chinese (简体中文).
Keep every placeholder such as {{v1}} and every tag such as <style id='1'>…</style> or <b1>…</b1> exactly unchanged, in the same order; translate the text inside tags.
Keep numbers, citations, formulas, URLs and names unchanged. Output only the translation.

{source}"""


def role_prompt(settings: Settings) -> str:
    extra = settings.extra_prompt.strip()
    return TRANSLATOR_ROLE + ("\n\nAdditional requirements from the user:\n" + extra if extra else "")


def uses_dashscope(settings: Settings) -> bool:
    return "dashscope" in settings.base_url or settings.model.lower().startswith("qwen")


def llm_extra_body(settings: Settings) -> dict:
    # Qwen3 models think by default; translation doesn't need it and non-streaming
    # calls to the open-weight Qwen3 models reject thinking outright.
    return {"enable_thinking": False} if uses_dashscope(settings) else {}


@dataclass
class Paragraph:
    """One translated paragraph as it ends up in the output, for the quality report."""

    source: str
    final: str
    draft: str = ""
    layout: str = ""
    problems: list[str] = field(default_factory=list)  # found (and fixed) by the reviewer
    issues: list[str] = field(default_factory=list)  # still failing rule checks on the final text
    corrected: bool = False  # the reviewer changed the draft
    overridden: bool = False  # a manual correction was applied


class Journal:
    """Thread-safe collection of Paragraph records for one job."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items: dict[str, Paragraph] = {}

    def add(self, p: Paragraph) -> None:
        with self.lock:
            self.items[p.source] = p

    def all(self) -> list[Paragraph]:
        with self.lock:
            return list(self.items.values())


class Proofreader:
    """Rule checks, LLM review and the per-paragraph bookkeeping shared by both translators."""

    def __init__(self, settings: Settings, store: Store | None, journal: Journal | None, client=None):
        self.mode = settings.proofread
        self.model = settings.model
        self.review_model = settings.resolved_review_model()
        self.role = role_prompt(settings)
        self.extra_body = llm_extra_body(settings)
        self.store = store
        self.journal = journal if journal is not None else Journal()
        self.client = client  # an openai.OpenAI, or None when no LLM is available
        self.on_usage = lambda resp: None

    @property
    def can_review(self) -> bool:
        return self.mode == "full" and self.client is not None

    def chat(self, model: str, system: str, user: str, max_tokens: int = 8192) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body=self.extra_body,
        )
        self.on_usage(resp)
        return (resp.choices[0].message.content or "").strip()

    def improve(self, sources: dict[int, str], drafts: dict[int, str], glossary: str = "", context: str = "", retranslate=None) -> dict[int, str]:
        """Check every draft; let the reviewer correct them, or re-translate failures."""
        keep = pr.glossary_targets(glossary)
        checks = {i: pr.check_translation(sources[i], drafts.get(i, ""), keep) for i in sources}
        finals = dict(drafts)
        problems: dict[int, list[str]] = {i: [] for i in sources}

        reviews = None
        if self.can_review:
            payload = [
                {"id": i, "source": sources[i], "translation": drafts.get(i, ""), "自动检查提示": checks[i].issues}
                for i in sources
            ]
            try:
                reply = self.chat(self.review_model, pr.REVIEW_SYSTEM, pr.build_review_prompt(payload, glossary, context))
                reviews = pr.parse_review(reply)
            except Exception as e:
                logger.warning("review call failed, falling back to rule checks: %s", e)
        if reviews is not None:
            for i in sources:
                finals[i], problems[i] = pr.choose(sources[i], drafts.get(i, ""), reviews.get(i), keep)
        elif retranslate is not None:
            for i, check in checks.items():
                if check.ok:
                    continue
                try:
                    again = retranslate(sources[i])
                except Exception as e:
                    logger.warning("re-translation failed: %s", e)
                    continue
                if len(pr.check_translation(sources[i], again, keep).issues) < len(check.issues):
                    finals[i] = again
                    problems[i] = check.issues
        if self.store is not None:
            for i in sources:
                self.store.save_note(sources[i], drafts.get(i, ""), finals.get(i, ""), problems[i])
        return finals

    def override(self, source: str) -> str | None:
        return self.store.override(source) if self.store is not None else None

    def finish(self, source: str, text: str, layout: str = "", keep: set[str] | None = None) -> str:
        """Apply a manual correction if there is one and record the paragraph for the report."""
        record = Paragraph(source=source, final=text, layout=layout)
        override = self.override(source)
        if override:
            record.final = override
            record.overridden = True
        note = self.store.note(source) if self.store is not None else None
        if note:
            record.draft = note["draft"]
            record.problems = note["problems"]
            record.corrected = bool(note["draft"]) and note["draft"].strip() != note["final"].strip()
        if not record.overridden:
            record.issues = pr.check_translation(source, record.final, keep or set()).issues
        self.journal.add(record)
        return record.final


class ProofreadingTranslator(OpenAITranslator):
    """LLM translation (Qwen or any OpenAI-compatible model) with proofreading."""

    # BabelDOC's cache stores this name in a 20-character column.
    name = "pdftrans"

    def __init__(self, settings: Settings, store: Store | None = None, journal: Journal | None = None):
        super().__init__(
            lang_in="en",
            lang_out="zh-CN",
            model=settings.model,
            base_url=settings.base_url or None,
            api_key=settings.resolved_api_key() or "EMPTY",
            ignore_cache=False,
        )
        self.extra_body.update(llm_extra_body(settings))
        if self.extra_body:
            self.add_cache_impact_parameters("extra_body", str(sorted(self.extra_body.items())))
        self.proof = Proofreader(settings, store, journal, self.client)
        self.proof.on_usage = self.update_token_count
        if self.proof.mode != "off":
            # Cached results are proofread results, so they depend on how proofreading ran.
            self.add_cache_impact_parameters(
                "proofread", f"{PROOFREAD_VERSION}:{self.proof.mode}:{self.proof.review_model}"
            )

    @property
    def journal(self) -> Journal:
        return self.proof.journal

    def preflight(self) -> None:
        """Fail fast on a wrong key, URL or model instead of producing an untranslated PDF."""
        models = [self.model]
        if self.proof.can_review and self.proof.review_model != self.model:
            models.append(self.proof.review_model)
        for model in models:
            llm_check(self.proof, model)

    def _retranslate(self, source: str) -> str:
        return self.proof.chat(self.model, self.proof.role, RETRANSLATE_USER.format(source=source))

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None  # BabelDOC probes LLM support this way
        draft = super().do_llm_translate(text, rate_limit_params or {})
        if self.proof.mode == "off":
            return draft
        try:
            batch = pr.parse_batch_prompt(text)
            if batch is not None:
                drafts = pr.parse_batch_output(draft)
                sources = {int(i["id"]): i.get("input", "") for i in batch.items if int(i["id"]) in drafts}
                finals = self.proof.improve(sources, drafts, batch.glossary, batch.context, self._retranslate)
                return pr.dump_batch_output(finals)
            single = pr.parse_single_prompt(text)
            if single is not None:
                return self.proof.improve({0: single}, {0: draft}, retranslate=self._retranslate)[0]
        except Exception as e:  # proofreading must never lose the draft translation
            logger.warning("proofreading failed, keeping the draft: %s", e)
        return draft  # term extraction and other prompts pass through untouched

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        # Runs after BabelDOC's cache lookup too, so manual corrections and the report
        # also cover paragraphs whose translation came from the cache.
        single = pr.parse_single_prompt(text)
        if single is not None and self.proof.override(single):
            return self.proof.finish(single, "")  # no need to ask the model
        out = super().llm_translate(text, ignore_cache, rate_limit_params)
        try:
            batch = pr.parse_batch_prompt(text)
            if batch is not None:
                results = pr.parse_batch_output(out)
                keep = pr.glossary_targets(batch.glossary)
                for item in batch.items:
                    i = int(item["id"])
                    if i in results:
                        results[i] = self.proof.finish(item.get("input", ""), results[i], item.get("layout_label", ""), keep)
                return pr.dump_batch_output(results)
            if single is not None:
                return self.proof.finish(single, out)
        except Exception as e:
            logger.debug("could not post-process translation: %s", e)
        return out


class MachineTranslator(BaseTranslator):
    """Google Translate / Microsoft Translator, proofread by rules (and the LLM when configured)."""

    model = ""

    def __init__(self, settings: Settings, store: Store | None = None, journal: Journal | None = None, client: Client | None = None):
        self.name = f"pdftrans-{settings.engine}"[:20]
        super().__init__("en", "zh-CN", ignore_cache=False)
        self.model = settings.engine
        self.mt = client or make_client(settings.engine, settings.proxy)
        llm = None
        if settings.proofread == "full" and settings.resolved_api_key():
            llm = openai.OpenAI(base_url=settings.base_url or None, api_key=settings.resolved_api_key(), timeout=600)
        self.proof = Proofreader(settings, store, journal, llm)
        if self.proof.can_review:
            self.add_cache_impact_parameters("proofread", f"{PROOFREAD_VERSION}:full:{self.proof.review_model}")
        self.token_count = _Counter()

    @property
    def journal(self) -> Journal:
        return self.proof.journal

    def preflight(self) -> None:
        try:
            self.mt.translate("Hello")
        except Exception as e:
            hint = "（谷歌翻译在中国大陆需要设置网络代理）" if self.model == "google" else ""
            raise RuntimeError(f"无法连接{ENGINES[self.model]}：{e}{hint}") from e
        if self.proof.can_review:
            llm_check(self.proof, self.proof.review_model)

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        raise NotImplementedError  # tells BabelDOC to send plain paragraphs to translate()

    def do_translate(self, text, rate_limit_params: dict = None):
        draft = self.mt.translate(text)
        if self.proof.mode == "off":
            return draft
        try:
            return self.proof.improve({0: text}, {0: draft})[0]
        except Exception as e:
            logger.warning("proofreading failed, keeping the draft: %s", e)
            return draft

    def translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        if self.proof.override(text):
            return self.proof.finish(text, "")
        out = super().translate(text, ignore_cache, rate_limit_params)
        return self.proof.finish(text, out)


def llm_check(proof: Proofreader, model: str) -> str:
    """One tiny request, with errors explained in Chinese."""
    try:
        return proof.chat(model, "You are a translator.", "Translate into Chinese: Hello", max_tokens=20)
    except openai.AuthenticationError as e:
        raise RuntimeError("API Key 无效或已过期，请检查 API Key") from e
    except openai.PermissionDeniedError as e:
        raise RuntimeError(f"没有权限调用模型 {model}（可能未开通该模型，或 Key 不能用于此接口）") from e
    except openai.NotFoundError as e:
        raise RuntimeError(f"找不到模型 {model}，或 Base URL 填写错误") from e
    except openai.RateLimitError as e:
        raise RuntimeError("请求太频繁或额度已用完，请稍后再试或检查账户余额") from e
    except openai.APIConnectionError as e:
        raise RuntimeError("无法连接大模型服务，请检查 Base URL 和网络") from e
    except openai.APIStatusError as e:
        raise RuntimeError(f"大模型服务返回错误 HTTP {e.status_code}：{e.message}") from e


class _Counter:
    value = 0


def make_translator(settings: Settings, store: Store | None = None, journal: Journal | None = None):
    if settings.engine == "llm":
        return ProofreadingTranslator(settings, store, journal)
    return MachineTranslator(settings, store, journal)
