"""Typer-powered CLI entrypoints."""
from __future__ import annotations

from pathlib import Path

import typer

from . import neat_runner
from .config import write_default_config

app = typer.Typer(help="Command line interface for the evolution game.")


@app.command()
def train(
    generations: int = typer.Option(10, help="Number of generations to train."),
    render: bool = typer.Option(False, help="Render the simulation while training."),
    config: Path | None = typer.Option(None, help="Path to a TOML config file."),
    show_sensors: bool | None = typer.Option(
        None, help="Override whether sensor overlays are drawn when rendering."
    ),
) -> None:
    """Run evolutionary training."""

    neat_runner.run_training(generations, render=render, config_path=config, show_sensors=show_sensors)


@app.command(name="visualize-best")
def visualize_best(
    config: Path | None = typer.Option(None, help="Path to a TOML config file."),
    show_sensors: bool | None = typer.Option(
        None, help="Override whether sensor overlays are drawn when rendering."
    ),
) -> None:
    """Visualize the best saved genome."""

    neat_runner.run_best(render=True, config_path=config, show_sensors=show_sensors)


@app.command()
def resume(
    render: bool = typer.Option(False, help="Render while resuming training."),
    config: Path | None = typer.Option(None, help="Path to a TOML config file."),
    show_sensors: bool | None = typer.Option(
        None, help="Override whether sensor overlays are drawn when rendering."
    ),
) -> None:
    """Resume training from the last checkpoint."""

    neat_runner.resume_training(render=render, config_path=config, show_sensors=show_sensors)


@app.command(name="export-config")
def export_config(
    path: Path = typer.Option(Path("config.toml"), help="Where to write the default TOML config."),
    overwrite: bool = typer.Option(False, help="Overwrite an existing file."),
) -> None:
    """Write a default configuration file to disk."""

    try:
        destination = write_default_config(path, overwrite=overwrite)
    except FileExistsError as exc:  # pragma: no cover - exercised via CLI usage
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    typer.echo(f"Wrote default config to {destination}")


if __name__ == "__main__":
    app()

