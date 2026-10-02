# PyInstaller spec for the pdftrans desktop app (Windows and macOS).
#
# Build from the repository root:
#   babeldoc --generate-offline-assets build/assets   # layout model + fonts, bundled into the app
#   pyinstaller packaging/pdftrans.spec
#
# Produces two executables sharing one bundle: "pdftrans" (the GUI) and
# "pdftrans-cli" (console: command-line translation and --self-test).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).parent
VERSION = "0.3.0"

datas, binaries, hiddenimports = [], [], []
for package in (
    "babeldoc", "bitstring", "tiktoken", "tiktoken_ext", "rtree", "freetype",
    "uharfbuzz", "hyperscan", "onnxruntime", "xsdata",
):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
for dist in ("babeldoc", "openai", "tiktoken", "xsdata", "pymupdf"):
    datas += copy_metadata(dist)
for package in sorted((ROOT / "build" / "assets").glob("offline_assets_*.zip")):
    datas.append((str(package), "assets"))
datas.append((str(ROOT / "LICENSE"), "."))

common = dict(
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["pdftrans.gui", "pdftrans.cli", "tiktoken_ext.openai_public"],
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6"],
)
gui = Analysis([str(ROOT / "packaging" / "launcher.py")], **common)
cli = Analysis([str(ROOT / "packaging" / "cli_launcher.py")], **common)

gui_exe = EXE(PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="pdftrans", console=False)
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="pdftrans-cli", console=True)
coll = COLLECT(
    gui_exe, cli_exe,
    gui.binaries + cli.binaries,
    gui.datas + cli.datas,
    name="pdftrans",
)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="pdftrans.app",
        bundle_identifier="io.github.bond159.pdftrans",
        info_plist={
            "CFBundleDisplayName": "PDF英译中",
            "CFBundleShortVersionString": VERSION,
            "NSHighResolutionCapable": True,
        },
    )
