"""Application configuration models and loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - fallback for Python <3.11
    import tomli as tomllib


class SimulationSettings(BaseModel):
    """Settings that control simulation timing and agent actions."""

    ticks_per_second: int = Field(60, description="Physics steps per second.")
    max_steps: int = Field(600, description="Maximum physics steps per generation.")
    sensor_range: float = Field(400.0, description="Maximum distance sensors can read.")
    move_force: float = Field(500.0, description="Force applied for horizontal movement.")
    jump_impulse: float = Field(1500.0, description="Impulse applied when jumping.")
    agent_radius: float = Field(12.0, description="Radius of the circular agent.")
    energy_per_force: float = Field(0.002, description="Energy cost per unit of applied horizontal force.")
    energy_per_jump: float = Field(0.5, description="Energy cost per jump.")
    max_energy: float = Field(15.0, description="Total energy budget before the agent exhausts.")


class WorldSettings(BaseModel):
    """Settings used when constructing the physics world."""

    width: float = Field(800.0, description="World width in pixels.")
    height: float = Field(600.0, description="World height in pixels.")
    gravity_x: float = Field(0.0, description="Horizontal gravity component.")
    gravity_y: float = Field(-900.0, description="Vertical gravity component.")
    ground_height: float = Field(40.0, description="Height of the ground segment from the bottom.")
    obstacles: Tuple[Tuple[float, float, float, float], ...] = Field(
        (
            (200.0, 120.0, 120.0, 20.0),
            (400.0, 200.0, 140.0, 20.0),
        ),
        description="Rectangular obstacles represented as (x, y, width, height).",
    )
    hazards: Tuple[Tuple[float, float, float, float], ...] = Field(
        ((520.0, 70.0, 120.0, 16.0),),
        description="Hazard rectangles (x, y, width, height) that eliminate agents on contact.",
    )
    target_position: Tuple[float, float] = Field((700.0, 100.0), description="X/Y target position.")
    target_motion_amplitude: float = Field(
        80.0, description="Horizontal oscillation amplitude for the target (0 to disable)."
    )
    target_motion_speed: float = Field(1.5, description="Speed multiplier for the moving target.")


class PopulationSettings(BaseModel):
    """Settings for NEAT population and fitness goals."""

    population_size: int = Field(20, description="Number of genomes per generation.")
    fitness_goal: float = Field(200.0, description="Fitness threshold to stop training early.")
    max_generations: int = Field(10, description="Maximum generations to run.")
    checkpoint_interval: int = Field(5, description="Generations between checkpoints.")
    checkpoint_dir: Path = Field(Path("checkpoints"), description="Directory for checkpoint files.")


class RenderSettings(BaseModel):
    """Optional rendering controls."""

    show_sensors: bool = Field(False, description="Draw basic sensor overlays when rendering.")
    show_trails: bool = Field(True, description="Render short motion trails for the best agent.")


class AppConfig(BaseModel):
    """Top-level configuration container."""

    simulation: SimulationSettings = SimulationSettings()
    world: WorldSettings = WorldSettings()
    population: PopulationSettings = PopulationSettings()
    neat_config_path: Path = Field(Path("neat-config.cfg"), description="Path to NEAT configuration file.")
    render: RenderSettings = Field(default_factory=RenderSettings)


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load configuration from an optional TOML file or return defaults.

    Args:
        config_path: Optional path to a TOML configuration file.

    Returns:
        AppConfig: Fully populated configuration object.
    """

    path = Path(config_path) if config_path else Path("config.toml")
    if not path.exists():
        return AppConfig()

    content = path.read_bytes()
    data = tomllib.loads(content.decode("utf-8")) if content else {}

    return AppConfig.model_validate(cast_dict(data))


def cast_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert nested dict keys to match AppConfig structure."""

    return raw


def write_default_config(path: Path, overwrite: bool = False) -> Path:
    """Write the default configuration to a TOML file.

    Args:
        path: Destination path for the generated TOML file.
        overwrite: Whether to replace an existing file.

    Returns:
        Path: The path that was written.
    """

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists. Use --overwrite to replace it.")

    # Import locally so consumers who only read configs do not require the
    # optional writer dependency until they actually generate a file.
    from tomli_w import dumps

    config = AppConfig()
    payload = config.model_dump(mode="json")
    destination.write_text(dumps(payload))
    return destination

