"""Settings, provider presets and their persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("PDFTRANS_HOME", Path.home() / ".pdftrans"))
CONFIG_FILE = CONFIG_DIR / "config.json"

# Language code -> (display name, name used in the prompt)
LANGUAGES = {
    "zh-CN": ("简体中文", "Simplified Chinese"),
    "zh-TW": ("繁體中文", "Traditional Chinese"),
    "en": ("English", "English"),
    "ja": ("日本語", "Japanese"),
    "ko": ("한국어", "Korean"),
    "fr": ("Français", "French"),
    "de": ("Deutsch", "German"),
    "es": ("Español", "Spanish"),
    "ru": ("Русский", "Russian"),
}

# Preset name -> (provider kind, base url, default model)
# "openai" covers every service speaking the OpenAI chat-completions protocol.
PRESETS = {
    "OpenAI": ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    "DeepSeek": ("openai", "https://api.deepseek.com/v1", "deepseek-chat"),
    "通义千问 (DashScope)": ("openai", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "智谱 GLM": ("openai", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    "Moonshot (Kimi)": ("openai", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    "SiliconFlow": ("openai", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-7B-Instruct"),
    "Ollama (本地)": ("openai", "http://localhost:11434/v1", "qwen2.5:7b"),
    "Claude (Anthropic)": ("anthropic", "", "claude-opus-5-5"),
    "自定义 OpenAI 兼容接口": ("openai", "", ""),
}

OUTPUT_MODES = ("mono", "dual", "alt")


@dataclass
class Settings:
    preset: str = "DeepSeek"
    provider: str = "openai"
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    target_lang: str = "zh-CN"
    # mono: translation only, dual: original | translation side by side,
    # alt: original and translated pages alternating
    modes: list[str] = field(default_factory=lambda: ["mono", "dual"])
    pages: str = ""  # e.g. "1-3,7"; empty means all pages
    concurrency: int = 4
    batch_chars: int = 2500
    temperature: float = 0.3
    effort: str = "low"  # Claude only
    extra_prompt: str = ""  # glossary / style instructions
    font_file: str = ""  # optional TTF/OTF used for the translated text
    use_cache: bool = True
    output_dir: str = ""
    saved_keys: dict[str, str] = field(default_factory=dict)  # API key per preset

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", "")
        return os.environ.get("PDFTRANS_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def load_settings(path: Path = CONFIG_FILE) -> Settings:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in data.items() if k in known})


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
