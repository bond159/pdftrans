"""Command-line front end, sharing settings with the GUI."""

from __future__ import annotations

import argparse
import sys

from .config import ENGINES, PROOFREAD_MODES, load_settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdftrans",
        description="把英文 PDF 翻译成中文并保留原版面（BabelDOC 引擎 + 自动校对）。未给出的选项沿用图形界面保存的设置。",
    )
    p.add_argument("pdf", nargs="+", help="要翻译的 PDF 文件")
    p.add_argument("--engine", choices=list(ENGINES), help="翻译引擎：llm（大模型）/ google / microsoft")
    p.add_argument("--proxy", help="谷歌/微软翻译使用的网络代理，如 http://127.0.0.1:7890")
    p.add_argument("--base-url", help="OpenAI 兼容接口地址，默认阿里云百炼")
    p.add_argument("--api-key", help="API Key（也可用环境变量 DASHSCOPE_API_KEY）")
    p.add_argument("--model", help="翻译模型，如 qwen-plus")
    p.add_argument("--review-model", help="审校模型，如 qwen-max")
    p.add_argument("--proofread", choices=list(PROOFREAD_MODES), help="自动校对：full / rules / off")
    p.add_argument("--dual", action="store_true", default=None, help="同时输出左右双语对照 PDF")
    p.add_argument("--pages", help="页码范围，如 1-5,8")
    p.add_argument("--glossary", help="术语表 CSV（列：source,target）")
    p.add_argument("--output-dir", "-o")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    st = load_settings()
    for attr, value in [
        ("engine", args.engine),
        ("proxy", args.proxy),
        ("base_url", args.base_url),
        ("api_key", args.api_key),
        ("model", args.model),
        ("review_model", args.review_model),
        ("proofread", args.proofread),
        ("dual", args.dual),
        ("pages", args.pages),
        ("glossary_file", args.glossary),
        ("output_dir", args.output_dir),
    ]:
        if value is not None:
            setattr(st, attr, value)
    if st.engine == "llm" and not st.resolved_api_key():
        print("缺少 API Key：用 --api-key 指定，或设置环境变量 DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    from .engine import Job

    def progress(pct: float, msg: str) -> None:
        print(f"\r{pct:5.1f}%  {msg}".ljust(60), end="", file=sys.stderr, flush=True)

    status = 0
    for path in args.pdf:
        print(f"{path}:", file=sys.stderr)
        try:
            result = Job(path, st, progress).run()
        except Exception as e:
            print(f"\n  失败：{e}", file=sys.stderr)
            status = 1
            continue
        print(f"\n  {result.report.summary()}", file=sys.stderr)
        for out in (result.mono, result.dual):
            if out:
                print(out)
    return status


if __name__ == "__main__":
    sys.exit(main())
