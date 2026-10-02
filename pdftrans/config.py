"""Settings, service presets and their persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("PDFTRANS_HOME", Path.home() / ".pdftrans"))
CONFIG_FILE = CONFIG_DIR / "config.json"

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Preset name -> base URL. Every preset speaks the OpenAI chat-completions protocol.
PRESETS = {
    "通义千问（阿里云百炼）": DASHSCOPE_URL,
    "通义千问（百炼国际站）": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "自定义 OpenAI 兼容接口": "",
}

# Suggestions shown in the model boxes; any model name the service accepts can be typed in.
QWEN_MODELS = ["qwen-plus", "qwen-max", "qwen-flash", "qwen-plus-latest", "qwen-max-latest", "qwen-turbo"]

PROOFREAD_MODES = {
    "full": "规则检查 + 大模型审校（推荐）",
    "rules": "仅规则检查（不合格的段落重新翻译）",
    "off": "关闭",
}


@dataclass
class Settings:
    preset: str = "通义千问（阿里云百炼）"
    base_url: str = DASHSCOPE_URL
    api_key: str = ""
    model: str = "qwen-plus"
    review_model: str = "qwen-max"  # empty: same as model
    proofread: str = "full"
    dual: bool = False  # also write the side-by-side bilingual PDF
    pages: str = ""  # e.g. "1-3,7"; empty means all pages
    output_dir: str = ""
    glossary_file: str = ""  # CSV with columns source,target
    auto_glossary: bool = True  # let BabelDOC extract terms and keep them consistent
    extra_prompt: str = ""  # extra translation instructions (style, field)
    translate_tables: bool = False  # experimental in BabelDOC
    ocr_workaround: bool = False  # for scanned PDFs with an OCR text layer
    font_family: str = "auto"  # auto / serif / sans-serif
    qps: int = 4  # requests per second sent to the API

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("PDFTRANS_API_KEY", "")

    def resolved_review_model(self) -> str:
        return self.review_model or self.model


def load_settings(path: Path = CONFIG_FILE) -> Settings:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    known = {f.name for f in fields(Settings)}
    settings = Settings(**{k: v for k, v in data.items() if k in known})
    if settings.preset not in PRESETS:
        settings.preset = "自定义 OpenAI 兼容接口"
    if settings.proofread not in PROOFREAD_MODES:
        settings.proofread = "full"
    return settings


def save_settings(settings: Settings, path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # the file may hold an API key
    except OSError:
        pass


def parse_pages(spec: str, page_count: int) -> list[int]:
    """Turn "1-3,7" into zero-based page indexes. Empty spec means every page."""
    spec = spec.strip().replace("，", ",")
    if not spec:
        return list(range(page_count))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s) if start_s.strip() else 1
            end = int(end_s) if end_s.strip() else page_count
        else:
            start = end = int(part)
        if start < 1 or end < start:
            raise ValueError(f"无效的页码范围: {part}")
        pages.update(range(start - 1, min(end, page_count)))
    return sorted(pages)
