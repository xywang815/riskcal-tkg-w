from argparse import ArgumentParser
from pathlib import Path

from riskcal_tkg.experiment import run_experiment


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    print(run_experiment(args.config, args.output_parent, args.resume_run))


if __name__ == "__main__":
    main()
