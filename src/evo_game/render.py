"""Rendering helpers built on pygame."""
from __future__ import annotations

from typing import List

import pygame
import pymunk

from .config import AppConfig
from .world import World


class Renderer:
    """Simple pygame-based renderer for the simulation."""

    def __init__(self, world: World, agents: List, app_config: AppConfig) -> None:
        self.world = world
        self.agents = agents
        self.app_config = app_config
        pygame.init()
        self.screen = pygame.display.set_mode((int(world.settings.width), int(world.settings.height)))
        pygame.display.set_caption("Evolution Game")
        self.clock = pygame.time.Clock()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def _to_screen(self, position: pymunk.Vec2d | tuple[float, float]) -> tuple[int, int]:
        x, y = position
        return int(x), int(self.world.settings.height - y)

    def draw(self, generation: int, step: int, best_fitness: float) -> None:
        self.screen.fill((30, 30, 40))
        self._draw_world()
        self._draw_agents()
        self._draw_hud(generation, step, best_fitness)
        pygame.display.flip()
        self.clock.tick(self.app_config.simulation.ticks_per_second)

    def _draw_world(self) -> None:
        for boundary in self.world.boundaries:
            a = self._to_screen(boundary.a)
            b = self._to_screen(boundary.b)
            pygame.draw.line(self.screen, (200, 200, 200), a, b, 3)

        for obstacle in self.world.obstacles:
            points = [self._to_screen(p) for p in obstacle.get_vertices()]  # type: ignore[arg-type]
            pygame.draw.polygon(self.screen, (100, 120, 200), points)

        target_pos = self._to_screen(self.world.target_body.position)
        pygame.draw.circle(self.screen, (200, 80, 80), target_pos, 10)

    def _draw_agents(self) -> None:
        for agent in self.agents:
            pos = self._to_screen(agent.body.position)
            color = (80, 200, 120) if agent.alive else (120, 120, 120)
            pygame.draw.circle(self.screen, color, pos, int(agent.sim_settings.agent_radius))

    def _draw_hud(self, generation: int, step: int, best_fitness: float) -> None:
        font = pygame.font.SysFont("arial", 18)
        lines = [
            f"Generation: {generation}",
            f"Step: {step}",
            f"Best fitness: {best_fitness:.2f}",
        ]
        for i, text in enumerate(lines):
            surface = font.render(text, True, (230, 230, 230))
            self.screen.blit(surface, (10, 10 + i * 20))

