"""Entry script for the frozen app (PyInstaller)."""

import multiprocessing
import sys

from pdftrans._frozen import add_dll_directories

add_dll_directories()  # before freeze_support(): child processes start here too

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from pdftrans.__main__ import main

    sys.exit(main())
