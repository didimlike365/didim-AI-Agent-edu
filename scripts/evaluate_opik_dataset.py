import argparse

from app.evaluation.opik_runner import run_dataset_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Opik experiment against the configured dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name. Defaults to OPIK__DATASET from .env.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional explicit experiment name.",
    )
    parser.add_argument(
        "--nb-samples",
        type=int,
        default=None,
        help="Optional sample limit for a smaller evaluation run.",
    )
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="Run only custom rule-based metrics without LLM judge metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_dataset_experiment(
        dataset_name=args.dataset,
        experiment_name=args.experiment_name,
        include_judge_metrics=not args.rule_only,
        nb_samples=args.nb_samples,
    )
    print("experiment_id:", result.experiment_id)
    print("experiment_name:", result.experiment_name)
    print("experiment_url:", result.experiment_url)
    print("dataset_id:", result.dataset_id)
    print("test_results:", len(result.test_results))


if __name__ == "__main__":
    main()
