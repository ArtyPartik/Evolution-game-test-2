"""Rendering helpers built on pygame."""
from __future__ import annotations

from typing import List

import pygame
import pymunk

from .agent import Agent
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
        self.trails: dict[int, List[tuple[int, int]]] = {}
        self.paused: bool = False

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.app_config.render.show_sensors = not self.app_config.render.show_sensors
                elif event.key == pygame.K_t:
                    self.app_config.render.show_trails = not self.app_config.render.show_trails
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True

    def _to_screen(self, position: pymunk.Vec2d | tuple[float, float]) -> tuple[int, int]:
        x, y = position
        return int(x), int(self.world.settings.height - y)

    def draw(self, generation: int, step: int, best_fitness: float) -> None:
        self.screen.fill((30, 30, 40))
        best_agent = self._best_agent()
        if best_agent and self.app_config.render.show_trails:
            self._update_trails(best_agent)
        self._draw_world()
        if best_agent and self.app_config.render.show_trails:
            self._draw_trails(best_agent)
        self._draw_agents(best_agent)
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

        for hazard in self.world.hazards:
            points = [self._to_screen(p) for p in hazard.get_vertices()]  # type: ignore[arg-type]
            pygame.draw.polygon(self.screen, (200, 90, 60), points)

        if self.world.settings.target_motion_amplitude > 0:
            base_x, base_y = self.world.settings.target_position
            amplitude = self.world.settings.target_motion_amplitude
            preview = [
                self._to_screen((base_x + amplitude, base_y)),
                self._to_screen((base_x - amplitude, base_y)),
            ]
            pygame.draw.lines(self.screen, (120, 90, 160), False, preview, 1)

        target_pos = self._to_screen(self.world.target_body.position)
        pygame.draw.circle(self.screen, (200, 80, 80), target_pos, 10)

    def _draw_agents(self, best_agent: Agent | None = None) -> None:
        for agent in self.agents:
            pos = self._to_screen(agent.body.position)
            color = (80, 200, 120) if agent.alive else (120, 120, 120)
            if agent is best_agent:
                color = (90, 220, 180)
            pygame.draw.circle(self.screen, color, pos, int(agent.sim_settings.agent_radius))
            if agent is best_agent:
                pygame.draw.circle(self.screen, (240, 230, 120), pos, int(agent.sim_settings.agent_radius) + 3, 1)
            if self.app_config.render.show_sensors:
                self._draw_sensors(agent)

    def _draw_hud(self, generation: int, step: int, best_fitness: float) -> None:
        font = pygame.font.SysFont("arial", 18)
        alive_count = sum(1 for a in self.agents if a.alive)
        best_energy = max((a.energy for a in self.agents if a.alive), default=0.0)
        lines = [
            f"Generation: {generation}",
            f"Step: {step}",
            f"Best fitness: {best_fitness:.2f}",
            f"Alive: {alive_count}/{len(self.agents)} | Best energy: {best_energy:.1f}",
            f"Sensors: {'on' if self.app_config.render.show_sensors else 'off'} | "
            f"Trails: {'on' if self.app_config.render.show_trails else 'off'}",
            "Paused: press SPACE to resume" if self.paused else "Press SPACE to pause",
        ]
        for i, text in enumerate(lines):
            surface = font.render(text, True, (230, 230, 230))
            self.screen.blit(surface, (10, 10 + i * 20))

    def _draw_sensors(self, agent: Agent) -> None:
        origin = self._to_screen(agent.body.position)

        target = self._to_screen(self.world.target_body.position)
        pygame.draw.line(self.screen, (210, 180, 90), origin, target, 1)

        ground_point = self._to_screen((agent.body.position.x, self.world.settings.ground_height))
        pygame.draw.line(self.screen, (120, 160, 240), origin, ground_point, 1)

        velocity = agent.body.velocity
        velocity_tip = self._to_screen(
            (agent.body.position.x + velocity.x * 0.15, agent.body.position.y + velocity.y * 0.15)
        )
        pygame.draw.line(self.screen, (140, 220, 220), origin, velocity_tip, 1)

    def _update_trails(self, best_agent: Agent) -> None:
        agent_id = id(best_agent)
        trail = self.trails.setdefault(agent_id, [])
        trail.append(self._to_screen(best_agent.body.position))
        max_length = 60
        if len(trail) > max_length:
            del trail[: len(trail) - max_length]
        self.trails = {agent_id: trail}

    def _draw_trails(self, best_agent: Agent) -> None:
        agent_id = id(best_agent)
        trail = self.trails.get(agent_id, [])
        if len(trail) > 1:
            pygame.draw.lines(self.screen, (160, 200, 255), False, trail, 2)

    def _best_agent(self) -> Agent | None:
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return None
        return max(alive, key=lambda a: a.fitness)

