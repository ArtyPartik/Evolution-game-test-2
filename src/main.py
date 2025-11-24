"""Entry point for running the Typer CLI."""
from __future__ import annotations

from evo_game.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

