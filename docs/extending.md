# Extending the Evolution Game

This project is intentionally small and readable so you can experiment quickly. Here are common extension points.

## Change the fitness function
- Open `src/evo_game/agent.py` and adjust how `Agent.update()` rewards behavior.
- The default rewards progress toward the target and time alive. You can add bonuses (e.g., reaching a region, collecting items) or penalties.

## Add sensors
- Add new readings to `Agent.get_sensor_values()` and update the NEAT config's `num_inputs` to match.
- Examples: distance to obstacles, raycasts toward walls, or a timer.

## Add actions
- Map extra network outputs inside `Agent.update()` (e.g., rotate, shoot, toggle lights).
- Increase `num_outputs` in `neat-config.cfg` accordingly and update how outputs are interpreted.

## Swap the task
- Move the target or add multiple targets in `world.py`, and adapt fitness in `agent.py` to reward touching/collecting them.
- For survival tasks, reward time alive and penalize collisions or falls.

## New environments or agent types
- Create additional world builders alongside `World` (different obstacles, gravity, moving platforms).
- Add new agent classes if you need different bodies or sensors; the `Simulation` only requires that agents expose `update()` and `fitness`.
