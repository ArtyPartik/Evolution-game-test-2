from pathlib import Path

from evo_game import neat_runner


NEAT_CONFIG_TEMPLATE = """
[NEAT]
pop_size              = 5
fitness_criterion     = max
fitness_threshold     = 5.0
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
single_structural_mutation = False
structural_mutation_surer = default
enabled_default         = True
enabled_mutate_rate     = 0.01
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0
feed_forward            = True
initial_connection      = full
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = 5
num_outputs             = 2
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 3
species_elitism      = 1

[DefaultReproduction]
elitism            = 1
survival_threshold = 0.2
min_species_size   = 1
"""


def test_short_training_run(tmp_path: Path) -> None:
    neat_config_path = tmp_path / "neat-test.cfg"
    neat_config_path.write_text(NEAT_CONFIG_TEMPLATE)
    checkpoint_dir = tmp_path / "checkpoints"

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[simulation]
max_steps = 10
ticks_per_second = 30

[population]
checkpoint_dir = "{checkpoint_dir}"

neat_config_path = "{neat_config_path}"
"""
    )

    neat_runner.run_training(1, render=False, config_path=config_path)
    assert checkpoint_dir.exists()

