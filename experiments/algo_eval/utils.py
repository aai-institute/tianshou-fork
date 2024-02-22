from tianshou.highlevel.experiment import Experiment


def shortener(input_string: str | None = None,
              length: int = 1,
              show_config_stack: bool = False):
    if input_string is None or len(input_string) == 0:
        return "default"
    output_parts = []

    for part in input_string.split(","):
        key, value = part.split("=")
        modified_key = ""

        key_parts = key.split(".")
        if not show_config_stack:
            key_parts = key_parts[-1:]
        for key_part in key_parts:
            for word in key_part.split("_"):
                modified_key += word[:length] + "_"
            modified_key = modified_key[:-1] + "."
        modified_key = modified_key[:-1]

        output_parts.append(f"{modified_key}={value}")

    return ",".join(output_parts)


def watch_experiment_policy(log_dir):
    experiment = Experiment.from_directory(log_dir)
    experiment.config.train = False
    experiment.config.watch = True
    experiment.config.watch_render = 0.0
    experiment.config.watch_num_episodes = 1
    experiment.config.device = "cpu"
    experiment.config.persistence_base_dir = log_dir

    experiment_result = experiment.run('watch')
    return experiment_result
