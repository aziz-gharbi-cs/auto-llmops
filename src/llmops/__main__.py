"""Entry point for executing llmops as a module (`python -m llmops`)."""

import sys

from llmops.interfaces.cli import main

if __name__ == "__main__":
    sys.exit(main())
