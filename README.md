# Evolution-game-test-2

Evo Game is a small but well-architected 2D evolution sandbox built in Python.

Simple physics-based creatures live in a 2D world and evolve over generations to perform a task such as:

- moving toward a target
- surviving as long as possible
- traveling as far as possible

The project focuses on clean structure and easy extension rather than flashy graphics. It is meant to be a base you can grow, refactor, and experiment with.

Core stack:

- `pygame` for rendering and input  
- `pymunk` for 2D physics  
- `neat-python` for neuroevolution  
- `typer` for the command line interface  
- `pydantic` for configuration  
- `pytest` for tests

---

## Features

- 2D physics world using `pymunk` (gravity, boundaries, simple obstacles)
- Creatures (agents) controlled by NEAT neural networks
- Fitness-based evolution over generations
- Headless training or training with a `pygame` window
- Checkpointing and resume support
- Simple CLI for training and visualization
- Configurable through code and optional config file
- Basic tests to keep the core stable

---

## Requirements

- Python 3.x (3.10+ recommended)
- A virtual environment is recommended

Main dependencies are listed in `requirements.txt` and include:

- `pygame`
- `pymunk`
- `neat-python`
- `typer`
- `pydantic`
- `pytest`
- plus small helpers if used (e.g. `pyyaml` or `tomli` for config files)

---

## Installation

Clone the repository and install the dependencies.

```bash
git clone <YOUR_REPO_URL> evo-game
cd evo-game

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
