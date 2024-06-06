import argparse

import experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("fold_data", type=int)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    if args.fold == -1:
        experiments.run_experiment_on_all_folds(
            config_name=args.config_name,
            n_epochs=args.n_epochs,
            save_name=args.save_name,
        )
    else:
        experiments.run_experiment_on_fold_priority(
            config_name=args.config_name,
            fold=args.fold,
            n_epochs=args.n_epochs,
            save_name=args.save_name,
        )


if __name__ == "__main__":
    main()
