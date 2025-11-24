"""NEAT integration helpers."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import neat

from .config import AppConfig, load_config
from .simulation import Simulation


def _load_neat_config(path: Path) -> neat.Config:
    if not path.exists():
        raise FileNotFoundError(f"NEAT config not found at {path}")
    return neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        path,
    )


def _evaluate_genomes(genomes, neat_config: neat.Config, app_config: AppConfig, render: bool, generation: int) -> None:
    simulation = Simulation(genomes, neat_config, app_config, render=render, generation=generation)
    simulation.run()


def run_training(num_generations: int, render: bool = False, config_path: Optional[Path] = None) -> None:
    """Run training for a set number of generations."""

    app_config = load_config(config_path)
    neat_config = _load_neat_config(app_config.neat_config_path)

    population = neat.Population(neat_config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    checkpoint_dir = app_config.population.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    population.add_reporter(neat.Checkpointer(app_config.population.checkpoint_interval, filename_prefix=str(checkpoint_dir / "neat-checkpoint-")))

    winner = population.run(lambda g, c: _evaluate_genomes(g, c, app_config, render, population.generation), num_generations)

    best_path = checkpoint_dir / "best-genome.pkl"
    with best_path.open("wb") as f:
        pickle.dump((neat_config, winner), f)
    print(f"Training finished. Best genome saved to {best_path}")


def run_best(render: bool = True, config_path: Optional[Path] = None) -> None:
    """Load the best genome from checkpoint and run a demo."""

    app_config = load_config(config_path)
    checkpoint_dir = app_config.population.checkpoint_dir
    best_path = checkpoint_dir / "best-genome.pkl"
    if not best_path.exists():
        print("No best genome found. Run training first.")
        return

    with best_path.open("rb") as f:
        neat_config, genome = pickle.load(f)

    genomes = [(0, genome)]
    simulation = Simulation(genomes, neat_config, app_config, render=render, generation=0)
    simulation.run()


def resume_training(render: bool = False, config_path: Optional[Path] = None) -> None:
    """Resume training from the latest checkpoint."""

    app_config = load_config(config_path)
    checkpoint_dir = app_config.population.checkpoint_dir
    latest = _find_latest_checkpoint(checkpoint_dir)
    if not latest:
        print("No checkpoint found; starting new training run.")
        run_training(app_config.population.max_generations, render=render, config_path=config_path)
        return

    neat_config = _load_neat_config(app_config.neat_config_path)
    population = neat.Checkpointer.restore_checkpoint(str(latest))
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(app_config.population.checkpoint_interval, filename_prefix=str(checkpoint_dir / "neat-checkpoint-")))
    population.run(lambda g, c: _evaluate_genomes(g, c, app_config, render, population.generation), app_config.population.max_generations)


def _find_latest_checkpoint(directory: Path) -> Optional[Path]:
    checkpoints = sorted(directory.glob("neat-checkpoint-*.pkl"))
    return checkpoints[-1] if checkpoints else None

