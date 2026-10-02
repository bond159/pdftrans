"""`python -m pdftrans` opens the GUI; `python -m pdftrans --cli file.pdf` translates headlessly."""

import multiprocessing
import sys


def _child_imports() -> str:
    # What BabelDOC's helper processes import first (font subsetting, PDF cleanup).
    import babeldoc.format.pdf.document_il.backend.pdf_creater  # noqa: F401

    return "ok"


def self_test() -> int:
    """Import everything a translation needs; used to check frozen builds in CI."""
    import babeldoc.docvision.doclayout  # noqa: F401  (onnxruntime, opencv)
    import babeldoc.format.pdf.high_level  # noqa: F401
    import pymupdf
    import tiktoken_ext.openai_public  # noqa: F401
    from PySide6.QtWidgets import QApplication  # noqa: F401

    from . import __version__
    from .engine import Job, bundled_assets  # noqa: F401
    from .translator import ProofreadingTranslator  # noqa: F401

    # BabelDOC runs some steps in child processes and silently falls back when they
    # crash, so check that a spawned child can import what it needs.
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        pool.apply(_child_imports)
    print(f"pdftrans {__version__} imports OK; PyMuPDF {pymupdf.VersionBind}; bundled assets: {bundled_assets()}")
    pdfs = [a for a in sys.argv[1:] if a.lower().endswith(".pdf")]
    if pdfs:
        # Parse the PDF and rebuild it without translating: exercises BabelDOC's parser,
        # fonts and PDF writer inside the frozen app without a model or an API key.
        import tempfile

        import babeldoc.format.pdf.high_level as high_level
        from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
        from babeldoc.translator.translator import OpenAITranslator

        high_level.init()
        from .engine import prepare_assets

        prepare_assets(lambda pct, msg: print(msg))
        with tempfile.TemporaryDirectory() as out:
            config = TranslationConfig(
                translator=OpenAITranslator("en", "zh-CN", "none", api_key="none"),
                input_file=pdfs[0],
                lang_in="en",
                lang_out="zh-CN",
                doc_layout_model=None,
                output_dir=out,
                no_dual=True,
                only_parse_generate_pdf=True,
                use_rich_pbar=False,
                watermark_output_mode=WatermarkOutputMode.NoWatermark,
            )
            result = high_level.translate(config)
            print(f"rebuilt {pdfs[0]} -> {result.mono_pdf_path.name}: {pymupdf.open(result.mono_pdf_path).page_count} pages")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from .cli import main as run
    else:
        from .gui import main as run
    return run()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for frozen (PyInstaller) builds on Windows
    sys.exit(main())
