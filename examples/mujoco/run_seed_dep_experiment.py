import itertools

from examples.mujoco.mujoco_sac_seed_exp import get_sac_args, run_sac

if __name__ == "__main__":
    policy_seed_list = [0]#, 1, 2, 3, 4]
    base_train_env_seed_list = [0, 1, 2, 3, 4]
    base_test_env_seed_list = [0]
    task_list = ["Pendulum-v1", "Ant-v4"]

    for task, policy_seed, base_train_env_seed, base_test_env_seed in itertools.product(task_list, policy_seed_list, base_train_env_seed_list, base_test_env_seed_list):
        print(task, policy_seed, base_train_env_seed, base_test_env_seed)
        args = get_sac_args()
        args.task = task
        args.seed = policy_seed
        args.base_train_env_seed = base_train_env_seed
        args.base_test_env_seed = base_test_env_seed
        args.runs_per_test_env = 10
        args.test_num = 10
        args.training_num = 5
        run_sac(args)