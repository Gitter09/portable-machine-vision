"""Small Windows entrypoint for the Admin PC runtime-only app."""

import sys

from vision_sdk.cli import main


if __name__ == "__main__":
    sys.argv.extend(["--mode", "runtime"])
    main()
