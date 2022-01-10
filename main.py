import argparse
import subprocess

import toml

VALID_OPTIONS = [
    "model",
    "dataset",
    "epochs",
    "learning-rate",
    "num-workers",
    "lambda",
    "batch-size",
    "test-batch-size",
    "aux-learning-rate",
    # "patch-size",
    "seed",
    "clip_max_norm",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script adapter.",
    )
    parser.add_argument(
        "--continue", dest="continue_", action="store_true", help="Continue."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = toml.load("../assets/config.toml")
    print(">>> Configuration:")
    print(config)
    print("<<<")

    prog_args = ["python", "examples/train.py", "--cuda", "--save"]
    prog_args += [
        f"--{k}={v}" for k, v in config.items() if k in VALID_OPTIONS
    ]
    if "patch-size" in config:
        h, w = config["patch-size"]
        prog_args += ["--patch-size", f"{h}", f"{w}"]
    if args.continue_:
        prog_args += ["--checkpoint=checkpoint.pth.tar"]
    print(" ".join(prog_args))
    print("", flush=True)

    subprocess.run(prog_args, check=True)


if __name__ == "__main__":
    main()
