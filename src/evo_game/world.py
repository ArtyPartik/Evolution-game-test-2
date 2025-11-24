"""Physics world setup using pymunk."""
from __future__ import annotations

from typing import List, Tuple

import pymunk

from .config import WorldSettings


class World:
    """Container for the pymunk space and static geometry."""

    def __init__(self, settings: WorldSettings) -> None:
        self.settings = settings
        self.space = pymunk.Space()
        self.space.gravity = (settings.gravity_x, settings.gravity_y)
        self.time = 0.0

        self.static_body = self.space.static_body
        self.boundaries: List[pymunk.Shape] = []
        self.obstacles: List[pymunk.Shape] = []
        self.hazards: List[pymunk.Shape] = []

        self._create_boundaries()
        self._create_obstacles()
        self._create_hazards()
        self.target_body, self.target_shape = self._create_target(settings.target_position)

    def _create_boundaries(self) -> None:
        width, height = self.settings.width, self.settings.height
        ground_y = self.settings.ground_height
        segments = [
            pymunk.Segment(self.static_body, (0, ground_y), (width, ground_y), 1),
            pymunk.Segment(self.static_body, (0, ground_y), (0, height), 1),
            pymunk.Segment(self.static_body, (width, ground_y), (width, height), 1),
        ]
        for segment in segments:
            segment.friction = 1.0
            self.space.add(segment)
        self.boundaries.extend(segments)

    def _create_obstacles(self) -> None:
        for x, y, w, h in self.settings.obstacles:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (x, y)
            shape = pymunk.Poly.create_box(body, size=(w, h))
            shape.friction = 0.8
            self.space.add(body, shape)
            self.obstacles.append(shape)

    def _create_hazards(self) -> None:
        for x, y, w, h in self.settings.hazards:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (x, y)
            shape = pymunk.Poly.create_box(body, size=(w, h))
            shape.sensor = True
            self.space.add(body, shape)
            self.hazards.append(shape)

    def _create_target(self, position: Tuple[float, float]) -> Tuple[pymunk.Body, pymunk.Shape]:
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        shape = pymunk.Circle(body, radius=12.0)
        shape.sensor = True
        self.space.add(body, shape)
        return body, shape

    def step(self, dt: float) -> None:
        """Advance the physics simulation by dt seconds."""
        self.time += dt
        self._update_target(dt)
        self.space.step(dt)

    def _update_target(self, dt: float) -> None:
        amplitude = self.settings.target_motion_amplitude
        if amplitude <= 0:
            return

        speed = self.settings.target_motion_speed
        base_x, base_y = self.settings.target_position
        offset = amplitude * pymunk.Vec2d(1, 0)
        offset = offset.rotated(speed * self.time)
        new_x = max(20.0, min(self.settings.width - 20.0, base_x + offset.x))
        self.target_body.position = (new_x, base_y)
        self.target_body.velocity = (offset.x * speed, 0.0)

    def hazard_distance(self, position: pymunk.Vec2d) -> float:
        """Return the minimum normalized distance to any hazard box (1.0 if none)."""

        if not self.hazards:
            return 1.0

        min_dist = float("inf")
        for hazard in self.hazards:
            vertices = [v + hazard.body.position for v in hazard.get_vertices()]  # type: ignore[arg-type]
            xs = [p.x for p in vertices]
            ys = [p.y for p in vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            closest_x = min(max(position.x, min_x), max_x)
            closest_y = min(max(position.y, min_y), max_y)
            dist = (position - pymunk.Vec2d(closest_x, closest_y)).length
            min_dist = min(min_dist, dist)

        norm = max(self.settings.width, self.settings.height)
        return min(1.0, min_dist / norm)

