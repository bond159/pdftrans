"""Start-up fixes for the PyInstaller build."""

import os
import sys


def add_dll_directories() -> None:
    """Make bundled DLLs findable on Windows, in the main process and in its children.

    Wheels repaired with delvewheel (hyperscan, numpy, scipy, ...) keep their DLLs in
    "<package>.libs" folders and register them when imported. Processes that
    BabelDOC spawns (font subsetting, PDF cleanup) can import such a package first,
    before anything registered the folder, and fail with "DLL load failed".
    """
    base = getattr(sys, "_MEIPASS", None)
    if os.name != "nt" or not base or not hasattr(os, "add_dll_directory"):
        return
    folders = [base] + [os.path.join(base, d) for d in os.listdir(base) if d.endswith(".libs")]
    for folder in folders:
        if os.path.isdir(folder):
            os.add_dll_directory(folder)
    os.environ["PATH"] = os.pathsep.join(folders + [os.environ.get("PATH", "")])
