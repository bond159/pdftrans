"""Command-line front end, sharing settings with the GUI."""

from __future__ import annotations

import argparse
import sys

from .config import LANGUAGES, OUTPUT_MODES, PRESETS, load_settings
from .llm import TranslationCache, Translator, make_backend
from .pipeline import translate_pdf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdftrans",
        description="用大模型翻译 PDF 并保留原排版。未指定的选项沿用 GUI 中保存的设置。",
    )
    p.add_argument("pdf", nargs="+", help="要翻译的 PDF 文件")
    p.add_argument("--preset", choices=list(PRESETS), help="服务商预设")
    p.add_argument("--base-url", help="OpenAI 兼容接口地址，如 https://api.deepseek.com/v1")
    p.add_argument("--api-key", help="API Key（也可用环境变量 PDFTRANS_API_KEY / ANTHROPIC_API_KEY）")
    p.add_argument("--model", help="模型名称")
    p.add_argument("--lang", choices=list(LANGUAGES), help="目标语言")
    p.add_argument("--modes", help=f"输出方式，逗号分隔：{','.join(OUTPUT_MODES)}")
    p.add_argument("--pages", help="页码范围，如 1-5,8")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--output-dir", "-o")
    p.add_argument("--font", help="译文字体文件")
    p.add_argument("--no-cache", action="store_true", help="不使用译文缓存")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    st = load_settings()
    if args.preset:
        st.preset = args.preset
        st.provider, st.base_url, st.model = PRESETS[args.preset]
        st.api_key = st.saved_keys.get(args.preset, "")
    for attr, value in [
        ("base_url", args.base_url),
        ("api_key", args.api_key),
        ("model", args.model),
        ("target_lang", args.lang),
        ("pages", args.pages),
        ("concurrency", args.concurrency),
        ("output_dir", args.output_dir),
        ("font_file", args.font),
    ]:
        if value is not None:
            setattr(st, attr, value)
    if args.modes:
        st.modes = [m.strip() for m in args.modes.split(",") if m.strip() in OUTPUT_MODES]
    if args.no_cache:
        st.use_cache = False

    translator = Translator(make_backend(st), st, TranslationCache() if st.use_cache else None)

    def progress(done: int, total: int, msg: str) -> None:
        print(f"\r[{done}/{total}] {msg}".ljust(70), end="", file=sys.stderr, flush=True)

    status = 0
    for path in args.pdf:
        print(f"{path}:", file=sys.stderr)
        try:
            result = translate_pdf(path, st, translator, progress)
        except Exception as e:
            print(f"\n  失败：{e}", file=sys.stderr)
            status = 1
            continue
        print(file=sys.stderr)
        for out in result.outputs.values():
            print(out)
    return status


if __name__ == "__main__":
    sys.exit(main())
