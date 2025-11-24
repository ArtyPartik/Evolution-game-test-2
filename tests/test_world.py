from evo_game.config import load_config
from evo_game.world import World


def test_world_constructs() -> None:
    config = load_config()
    world = World(config.world)
    assert len(world.boundaries) == 3
    assert len(world.obstacles) == len(config.world.obstacles)
    assert world.target_body is not None

