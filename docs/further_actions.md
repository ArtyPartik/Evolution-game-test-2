# Recommended Follow-Up Actions

This list captures practical next steps to strengthen the evolution game beyond the initial scaffold.

## Simulation and Gameplay
- Tune physics and fitness: experiment with gravity, friction, and reward shaping (e.g., distance to target, energy penalties) to stabilize training.
- Add richer targets: support moving targets, multiple goals, or hazards that influence fitness.
- Expand agent actions: allow jumping, limited air control, or energy-based thrust to encourage efficient movement.
- Improve sensors: add downward raycasts for ground detection, obstacle density sampling, or simple line-of-sight checks to the target.

## Rendering and UX
- Visual cues: draw sensor rays, fitness text per agent, and trail paths for the best genome to make debugging easier.
- Controls: add CLI flags to throttle FPS during rendering and to toggle overlays for performance debugging.
- Recording: optionally export GIF/MP4 snippets of the best agent using pygame surfaces.

## Configuration and Tooling
- Config presets: maintain multiple config files (e.g., fast-debug vs. training) and document the differences.
- Checkpoints: implement auto-pruning or rotation to avoid disk bloat; surface resume status in CLI output.
- Logging: add structured logs for generation stats and persist best genome metadata under `checkpoints/`.

## Testing and CI
- Property tests: fuzz sensor outputs for stability across random seeds and obstacle layouts.
- Long-run smoke: add a short headless NEAT run in CI to guard against regressions in the training loop.
- Linting: integrate ruff/black/mypy for style and type safety, and include them in contributor docs.

## Extensibility
- Modular agents: separate body construction from control to enable alternative morphologies.
- Tasks: factor out task definitions (reward + termination) so new objectives can be added without touching the core loop.
- Curriculum: gradually increase difficulty (e.g., target distance or obstacle count) as generations improve.
