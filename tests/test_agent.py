from evo_game.agent import Agent
from evo_game.config import load_config
from evo_game.world import World


class DummyNetwork:
    def activate(self, inputs):
        return [0.0, 0.0]


def test_agent_creation_and_sensors() -> None:
    config = load_config()
    world = World(config.world)
    agent = Agent(world, config.simulation)

    sensors = agent.get_sensor_values()
    assert len(sensors) == 5
    assert all(isinstance(v, float) for v in sensors)


def test_agent_update_does_not_crash() -> None:
    config = load_config()
    world = World(config.world)
    agent = Agent(world, config.simulation)
    network = DummyNetwork()

    agent.update(1.0 / config.simulation.ticks_per_second, network)  # type: ignore[arg-type]
    assert agent.fitness >= 0.0

