import hydra

from experiments.algo_eval.hydra_based.tianshou_runner import run_exp


@hydra.main(version_base=None, config_path="configs", config_name="hpo_ppo_pendulum_dehb")
def main(cfg):
    return run_exp(cfg)


if __name__ == "__main__":
    main()
