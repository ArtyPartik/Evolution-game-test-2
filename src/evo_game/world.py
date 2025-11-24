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

        self.static_body = self.space.static_body
        self.boundaries: List[pymunk.Shape] = []
        self.obstacles: List[pymunk.Shape] = []

        self._create_boundaries()
        self._create_obstacles()
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

    def _create_target(self, position: Tuple[float, float]) -> Tuple[pymunk.Body, pymunk.Shape]:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        shape = pymunk.Circle(body, radius=12.0)
        shape.sensor = True
        self.space.add(body, shape)
        return body, shape

    def step(self, dt: float) -> None:
        """Advance the physics simulation by dt seconds."""

        self.space.step(dt)

