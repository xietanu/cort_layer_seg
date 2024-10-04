import experiments

CONFIGS_PATH = "experiments/configs"


def run_experiment_on_all_folds(
    config_name: str,
    n_epochs: int = 100,
    save_name: str | None = None,
    use_random: bool = False,
    use_gaussian: bool = False,
):
    """Run an experiment on a fold_data."""
    for i in range(5):
        print(f"{config_name}, FOLD {i}:")
        experiments.run_experiment_on_fold(
            config_name, i, n_epochs, save_name, use_random, use_gaussian
        )
