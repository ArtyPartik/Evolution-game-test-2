"""Module entry point for running the CLI with `python -m evo_game.main`."""
from __future__ import annotations

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

