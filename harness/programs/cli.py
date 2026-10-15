from __future__ import annotations

import argparse
import asyncio

from harness.programs.classifier import JevWorkflowClassifier, RuleWorkflowClassifier
from harness.programs.evaluation import evaluate_classifier
from harness.programs.runner import ProgramRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Systems Research Lab program runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="classify and run a natural-language program")
    run.add_argument("program")
    run.add_argument("--classifier", choices=("jev", "rules"), default="jev")
    route_eval = subparsers.add_parser("eval-routing", help="evaluate workflow classification")
    route_eval.add_argument("--dataset", default="evals/workflow_routing_v1.json")
    route_eval.add_argument("--classifier", choices=("jev", "rules"), default="jev")
    return parser


def _classifier(name: str):
    return JevWorkflowClassifier() if name == "jev" else RuleWorkflowClassifier()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    classifier = _classifier(args.classifier)
    if args.command == "run":
        result = asyncio.run(ProgramRunner(classifier).run(args.program))
        print(result.model_dump_json(indent=2))
        return 0 if result.status.value == "completed" else 2
    if args.command == "eval-routing":
        result = evaluate_classifier(classifier, args.dataset)
        print(result.model_dump_json(indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
