"""Agent implementation that wraps a pymunk body and sensors."""
from __future__ import annotations

from typing import List

import pymunk
from pymunk.vec2d import Vec2d

import neat

from .config import SimulationSettings, WorldSettings
from .world import World


class Agent:
    """Simple circular agent controlled by a NEAT network."""

    def __init__(self, world: World, sim_settings: SimulationSettings, start_position: tuple[float, float] | None = None) -> None:
        self.world = world
        self.sim_settings = sim_settings
        radius = sim_settings.agent_radius

        mass = 1.0
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, inertia)
        initial_pos = start_position or (radius + 10.0, world.settings.ground_height + radius + 5.0)
        self.body.position = initial_pos
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = 1.0
        world.space.add(self.body, self.shape)

        self.alive: bool = True
        self.fitness: float = 0.0
        self.best_distance: float | None = None
        self.initial_distance: float = self._distance_to_target()
        self.energy: float = sim_settings.max_energy

    def _distance_to_target(self) -> float:
        target = Vec2d(*self.world.target_body.position)
        return target.get_distance(self.body.position)

    def get_sensor_values(self) -> List[float]:
        """Collect basic sensor readings for the agent."""

        target = Vec2d(*self.world.target_body.position)
        position = Vec2d(*self.body.position)
        velocity = Vec2d(*self.body.velocity)

        dx = (target.x - position.x) / self.world.settings.width
        dy = (target.y - position.y) / self.world.settings.height
        vx = velocity.x / self.sim_settings.sensor_range
        vy = velocity.y / self.sim_settings.sensor_range
        ground_dist = (position.y - self.world.settings.ground_height) / self.world.settings.height
        hazard_dist = self.world.hazard_distance(position)
        target_velocity = self.world.target_body.velocity.x / max(1.0, self.sim_settings.sensor_range)

        return [dx, dy, vx, vy, ground_dist, hazard_dist, target_velocity]

    def update(self, dt: float, network: neat.nn.FeedForwardNetwork) -> None:
        """Update the agent using the provided NEAT network."""

        if not self.alive:
            return

        sensors = self.get_sensor_values()
        output = network.activate(sensors)
        force_x = max(-1.0, min(1.0, output[0])) * self.sim_settings.move_force
        jump_signal = output[1]

        self.body.apply_force_at_local_point((force_x, 0.0))

        self._consume_energy(abs(force_x) * self.sim_settings.energy_per_force)

        if self._can_jump() and jump_signal > 0.5:
            self.body.apply_impulse_at_local_point((0.0, self.sim_settings.jump_impulse))
            self._consume_energy(self.sim_settings.energy_per_jump)

        current_distance = self._distance_to_target()
        if self.best_distance is None or current_distance < self.best_distance:
            self.best_distance = current_distance
            improvement = self.initial_distance - current_distance
            self.fitness = max(self.fitness, improvement)

        self.fitness += dt  # small reward for staying alive

        self._check_hazards()

        if position_below_floor(self.body, self.world.settings):
            self.alive = False

    def _can_jump(self) -> bool:
        return self.body.position.y <= self.world.settings.ground_height + self.sim_settings.agent_radius + 2.0

    def _check_hazards(self) -> None:
        if not self.world.hazards:
            return

        pos = Vec2d(*self.body.position)
        for hazard in self.world.hazards:
            vertices = [v + hazard.body.position for v in hazard.get_vertices()]  # type: ignore[arg-type]
            xs = [p.x for p in vertices]
            ys = [p.y for p in vertices]
            if min(xs) <= pos.x <= max(xs) and min(ys) <= pos.y <= max(ys):
                self.alive = False
                return

    def _consume_energy(self, amount: float) -> None:
        self.energy -= amount
        if self.energy <= 0:
            self.alive = False


def position_below_floor(body: pymunk.Body, settings: WorldSettings) -> bool:
    """Check if the body has fallen below the world floor."""

    return body.position.y < 0 or body.position.y < settings.ground_height - 5.0

