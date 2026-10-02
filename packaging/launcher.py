"""Entry script for the frozen app (PyInstaller)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from pdftrans.__main__ import main

    sys.exit(main())
