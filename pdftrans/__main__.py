"""`python -m pdftrans` opens the GUI; `python -m pdftrans file.pdf --cli` translates headlessly."""

import sys

if __name__ == "__main__":
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from .cli import main
    else:
        from .gui import main
    sys.exit(main())
