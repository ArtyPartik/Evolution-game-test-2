from pathlib import Path

from pathlib import Path

from evo_game.config import AppConfig, load_config, write_default_config


def test_load_default_config() -> None:
    config = load_config(None)
    assert isinstance(config, AppConfig)
    assert config.simulation.ticks_per_second == 60
    assert config.world.width == 800.0


def test_load_from_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[simulation]
ticks_per_second = 30
max_steps = 10

[world]
width = 640
height = 480

[population]
population_size = 5
fitness_goal = 5.0
checkpoint_dir = "checkpoints"

neat_config_path = "neat-config.cfg"
"""
    )

    config = load_config(config_path)
    assert config.simulation.ticks_per_second == 30
    assert config.world.width == 640
    assert config.population.population_size == 5
    assert config.neat_config_path == Path("neat-config.cfg")
    assert config.render.show_sensors is False


def test_write_default_config(tmp_path: Path) -> None:
    destination = tmp_path / "config.toml"
    written = write_default_config(destination)

    assert written == destination
    generated = load_config(destination)
    assert isinstance(generated, AppConfig)
    assert generated.render.show_sensors is False

