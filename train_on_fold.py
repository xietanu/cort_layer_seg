import argparse
import torch

import experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("fold", type=int)
    parser.add_argument("--n_epochs", type=int, default=-1)
    parser.add_argument("--end_after", type=int, default=-1)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--use_random", action="store_true")
    parser.add_argument("--use_gaussian", action="store_true")
    parser.add_argument("--use_synth", action="store_true")
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(False)

    if args.n_epochs <= 0 and args.end_after <= 0:
        raise ValueError("Must specify either n_epochs or end_after")

    if args.fold == -1:
        experiments.run_experiment_on_all_folds(
            config_name=args.config_name,
            n_epochs=args.n_epochs,
            save_name=args.save_name,
            use_random=args.use_random,
            use_gaussian=args.use_gaussian,
            use_synth=args.use_synth,
            end_after=args.end_after,
        )
    else:
        experiments.run_experiment_on_fold(
            config_name=args.config_name,
            fold=args.fold,
            n_epochs=args.n_epochs,
            save_name=args.save_name,
            continue_training=args.continue_training,
            use_random=args.use_random,
            use_gaussian=args.use_gaussian,
            use_synth=args.use_synth,
            end_after=args.end_after,
        )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
