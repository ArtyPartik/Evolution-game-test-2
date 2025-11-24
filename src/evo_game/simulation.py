"""Simulation loop for a single generation."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import neat

from .agent import Agent
from .config import AppConfig
from .render import Renderer
from .world import World


class Simulation:
    """Runs a population of agents through a physics simulation."""

    def __init__(self, genomes: Iterable[Tuple[int, neat.DefaultGenome]], neat_config: neat.Config, app_config: AppConfig, render: bool = False, generation: int = 0) -> None:
        self.genomes = list(genomes)
        self.neat_config = neat_config
        self.app_config = app_config
        self.world = World(app_config.world)
        self.render_enabled = render
        self.renderer: Renderer | None = Renderer(self.world, [], app_config) if render else None
        self.generation = generation

        self.networks: List[neat.nn.FeedForwardNetwork] = []
        self.agents: List[Agent] = []
        self._create_agents()
        if self.renderer:
            self.renderer.agents = self.agents

    def _create_agents(self) -> None:
        for _, genome in self.genomes:
            genome.fitness = 0.0
            network = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
            agent = Agent(self.world, self.app_config.simulation)
            self.networks.append(network)
            self.agents.append(agent)

    def run(self) -> None:
        dt = 1.0 / self.app_config.simulation.ticks_per_second
        max_steps = self.app_config.simulation.max_steps

        for step in range(max_steps):
            if self.renderer and not self.renderer.handle_events():
                break

            all_dead = True
            for agent, network in zip(self.agents, self.networks):
                if not agent.alive:
                    continue
                all_dead = False
                agent.update(dt, network)

            self.world.step(dt)

            if self.renderer:
                best = max((a.fitness for a in self.agents), default=0.0)
                self.renderer.draw(self.generation, step, best)

            if all_dead:
                break

        for (_, genome), agent in zip(self.genomes, self.agents):
            genome.fitness = max(agent.fitness, 0.0)

