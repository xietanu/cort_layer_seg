import argparse

import experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, default="acc")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--save_name", type=str, default="acc_model")
    parser.add_argument("--continue_training", action="store_true")
    args = parser.parse_args()

    experiments.train_acc_model(
        config_name=args.config_name,
        n_epochs=args.n_epochs,
        save_name=args.save_name,
        continue_training=args.continue_training,
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
