import argparse
import chain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    chain.find_all_corners(args.skip)
