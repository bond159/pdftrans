"""Entry script for the console executable (pdftrans-cli) in the frozen app."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--self-test" in sys.argv:
        from pdftrans.__main__ import self_test

        sys.exit(self_test())
    from pdftrans.cli import main

    sys.exit(main())
