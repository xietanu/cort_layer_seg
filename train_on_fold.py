import argparse

import experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("fold", type=int)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--continue_training", action="store_true")
    args = parser.parse_args()

    if args.fold == -1:
        experiments.run_experiment_on_all_folds(
            config_name=args.config_name,
            n_epochs=args.n_epochs,
            save_name=args.save_name,
        )
    else:
        experiments.run_experiment_on_fold(
            config_name=args.config_name,
            fold=args.fold,
            n_epochs=args.n_epochs,
            save_name=args.save_name,
            continue_training=args.continue_training,
        )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
