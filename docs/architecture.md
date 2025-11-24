# Architecture Overview

This project is split into small, well-named modules so each responsibility is clear and easy to extend.

## Configuration (`config.py`)
- Defines Pydantic models for simulation, world, and population/NEAT settings.
- `load_config()` reads an optional `config.toml` and falls back to sane defaults.
- Values such as gravity, ticks per second, and checkpoint intervals live here to avoid magic numbers in the code.

## World (`world.py`)
- Wraps a `pymunk.Space` with gravity, boundaries, obstacles, and a target object agents can chase.
- `World.step(dt)` advances physics without any gameplay logic.

## Agent (`agent.py`)
- Represents one creature with a circular body and a NEAT-controlled brain.
- Provides `get_sensor_values()` for network inputs (distances, velocity, ground offset).
- `update(dt, network)` applies forces/impulses from network outputs, updates fitness, and marks agents as dead when they fall.

## Simulation (`simulation.py`)
- Runs one generation: builds a `World`, constructs agents and NEAT networks, steps physics, and records fitness back to genomes.
- Rendering is optional and injected through the `Renderer` class; the simulation itself is headless.

## Renderer (`render.py`)
- Handles the `pygame` window, drawing boundaries, obstacles, target, and agents.
- Includes a small HUD with generation, step, and best fitness values.

## NEAT Runner (`neat_runner.py`)
- Loads the NEAT configuration file, wires up reporters/checkpointing, and connects NEAT to the `Simulation`.
- Exposes functions to train, resume from checkpoints, or visualize the best saved genome.

## CLI (`cli.py` and `main.py`)
- Typer-based CLI with commands: `train`, `visualize-best`, and `resume`.
- `src/main.py` is a thin entry point so the project can run via `python -m evo_game.main`.
