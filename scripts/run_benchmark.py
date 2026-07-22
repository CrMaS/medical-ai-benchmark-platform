import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.benchmark_service import run_random_baseline
from backend.app.services.run_store import save_run


def sample_count(value: str) -> int:
    count = int(value)
    if not 10 <= count <= 10_000:
        raise argparse.ArgumentTypeError("must be between 10 and 10000")
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Run a demo skin-lesion benchmark.")
    parser.add_argument(
        "--num-samples",
        type=sample_count,
        default=200,
        help="Number of synthetic samples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the run to the runs directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run = run_random_baseline(
        num_samples=args.num_samples,
        seed=args.seed,
    )

    if args.save:
        output_path = save_run(run)
        print(f"Saved benchmark run to {output_path}")

    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
