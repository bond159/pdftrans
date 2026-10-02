"""BabelDOC translator with automatic proofreading and manual corrections.

BabelDOC sends batches of paragraphs as one JSON prompt (and single paragraphs
when it falls back). This translator lets the translation model answer, then
checks every paragraph with deterministic rules and has a review model correct
it, before handing the result back to BabelDOC for typesetting.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from babeldoc.translator.translator import OpenAITranslator

from . import proofread as pr
from .config import Settings
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


def _uses_dashscope(settings: Settings) -> bool:
    return "dashscope" in settings.base_url or settings.model.lower().startswith("qwen")


class ProofreadingTranslator(OpenAITranslator):
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
        if _uses_dashscope(settings):
            # Qwen3 models think by default; translation doesn't need it and non-streaming
            # calls to the open-weight Qwen3 models reject thinking outright.
            self.extra_body["enable_thinking"] = False
            self.add_cache_impact_parameters("enable_thinking", False)
        self.mode = settings.proofread
        self.review_model = settings.resolved_review_model()
        self.role = role_prompt(settings)
        self.store = store
        self.journal = journal if journal is not None else Journal()
        if self.mode != "off":
            # Cached results are proofread results, so they depend on how proofreading ran.
            self.add_cache_impact_parameters("proofread", f"{PROOFREAD_VERSION}:{self.mode}:{self.review_model}")

    # ----- low-level -------------------------------------------------------
    def chat(self, model: str, system: str, user: str, max_tokens: int = 8192) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body=self.extra_body,
        )
        self.update_token_count(resp)
        return (resp.choices[0].message.content or "").strip()

    # ----- BabelDOC entry points -------------------------------------------
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None  # BabelDOC probes LLM support this way
        draft = super().do_llm_translate(text, rate_limit_params or {})
        if self.mode == "off":
            return draft
        try:
            batch = pr.parse_batch_prompt(text)
            if batch is not None:
                return self._proofread_batch(batch, draft)
            single = pr.parse_single_prompt(text)
            if single is not None:
                return self._proofread_single(single, draft)
        except Exception as e:  # proofreading must never lose the draft translation
            logger.warning("proofreading failed, keeping the draft: %s", e)
        return draft  # term extraction and other prompts pass through untouched

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        # Runs after BabelDOC's cache lookup too, so manual corrections and the report
        # also cover paragraphs whose translation came from the cache.
        single = pr.parse_single_prompt(text)
        if single is not None and self.store is not None and self.store.override(single):
            return self._finish(single, "", "", set())  # no need to ask the model
        out = super().llm_translate(text, ignore_cache, rate_limit_params)
        try:
            batch = pr.parse_batch_prompt(text)
            if batch is not None:
                results = pr.parse_batch_output(out)
                keep = pr.glossary_targets(batch.glossary)
                for item in batch.items:
                    i = int(item["id"])
                    if i in results:
                        results[i] = self._finish(item.get("input", ""), results[i], item.get("layout_label", ""), keep)
                return pr.dump_batch_output(results)
            single = pr.parse_single_prompt(text)
            if single is not None:
                return self._finish(single, out, "", set())
        except Exception as e:
            logger.debug("could not post-process translation: %s", e)
        return out

    # ----- proofreading ----------------------------------------------------
    def _retranslate(self, source: str) -> str:
        return self.chat(self.model, self.role, RETRANSLATE_USER.format(source=source))

    def _review(self, items: list[dict], glossary: str, context: str) -> dict[int, pr.Review]:
        reply = self.chat(self.review_model, pr.REVIEW_SYSTEM, pr.build_review_prompt(items, glossary, context))
        return pr.parse_review(reply)

    def _improve(self, sources: dict[int, str], drafts: dict[int, str], glossary: str, context: str) -> dict[int, str]:
        keep = pr.glossary_targets(glossary)
        checks = {i: pr.check_translation(sources[i], drafts.get(i, ""), keep) for i in sources}
        finals = dict(drafts)
        problems: dict[int, list[str]] = {i: [] for i in sources}

        reviews: dict[int, pr.Review] | None = None
        if self.mode == "full":
            payload = [
                {"id": i, "source": sources[i], "translation": drafts.get(i, ""), "自动检查提示": checks[i].issues}
                for i in sources
            ]
            try:
                reviews = self._review(payload, glossary, context)
            except Exception as e:
                logger.warning("review call failed, falling back to rule checks: %s", e)
        if reviews is not None:
            for i in sources:
                finals[i], problems[i] = pr.choose(sources[i], drafts.get(i, ""), reviews.get(i), keep)
        else:
            # Rules only (or the review failed): translate failing paragraphs again on their own.
            for i, check in checks.items():
                if check.ok:
                    continue
                try:
                    again = self._retranslate(sources[i])
                except Exception as e:
                    logger.warning("re-translation failed: %s", e)
                    continue
                if len(pr.check_translation(sources[i], again, keep).issues) < len(check.issues):
                    finals[i] = again
                    problems[i] = check.issues
        for i in sources:
            if self.store is not None:
                self.store.save_note(sources[i], drafts.get(i, ""), finals.get(i, ""), problems[i])
        return finals

    def _proofread_batch(self, batch: pr.BatchPrompt, draft: str) -> str:
        try:
            drafts = pr.parse_batch_output(draft)
        except ValueError:
            return draft  # unusable reply; BabelDOC retries these paragraphs one by one
        sources = {int(item["id"]): item.get("input", "") for item in batch.items if int(item["id"]) in drafts}
        finals = self._improve(sources, drafts, batch.glossary, batch.context)
        return pr.dump_batch_output(finals)

    def _proofread_single(self, source: str, draft: str) -> str:
        return self._improve({0: source}, {0: draft}, "", "")[0]

    def _finish(self, source: str, text: str, layout: str, keep: set[str]) -> str:
        """Apply a manual correction if there is one and record the paragraph for the report."""
        record = Paragraph(source=source, final=text, layout=layout)
        override = self.store.override(source) if self.store is not None else None
        if override:
            record.final = override
            record.overridden = True
        note = self.store.note(source) if self.store is not None else None
        if note:
            record.draft = note["draft"]
            record.problems = note["problems"]
            record.corrected = bool(note["draft"]) and note["draft"].strip() != note["final"].strip()
        if not record.overridden:
            record.issues = pr.check_translation(source, record.final, keep).issues
        self.journal.add(record)
        return record.final
