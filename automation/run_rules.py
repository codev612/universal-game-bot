from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from loguru import logger

from core import RuleEngine, RuleSnapshot, ScenarioRegistry, SnippetObservation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the rule-based automation engine on a saved snapshot.",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=RuleEngine.RULES_FILE,
        help="Path to rules.json (default: rules/rules.json).",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("rules") / "last_snapshot.json",
        help="Path to the snapshot JSON exported by the GUI (default: rules/last_snapshot.json).",
    )
    parser.add_argument(
        "--respect-cooldowns",
        action="store_true",
        help="Respect rule cooldowns when evaluating (default: disabled).",
    )
    parser.add_argument(
        "--show-scenario-steps",
        action="store_true",
        help="If the matched action is a scenario, list each step in the scenario sequence.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    return parser.parse_args()


def load_snapshot(path: Path) -> RuleSnapshot:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Snapshot file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid snapshot JSON at {path}: {exc}") from exc

    snippets_payload: Dict[str, Dict[str, object]] = data.get("snippets", {})
    snippets = {
        name: SnippetObservation(
            detected=bool(entry.get("detected")),
            score=entry.get("score"),
        )
        for name, entry in snippets_payload.items()
        if isinstance(entry, dict)
    }

    state_values = {
        name: str(value)
        for name, value in (data.get("state_values") or {}).items()
    }

    return RuleSnapshot(
        snippets=snippets,
        state_values=state_values,
        scenario_current=data.get("scenario_current"),
        scenario_next=data.get("scenario_next"),
        player_state=data.get("player_state"),
        metadata=data.get("metadata") or {},
    )


def main() -> int:
    args = parse_arguments()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    scenario_registry = ScenarioRegistry()
    engine = RuleEngine(scenario_registry, rules_path=args.rules)
    snapshot = load_snapshot(args.snapshot)

    logger.debug(
        "Evaluating rules using snapshot %s (snippets=%d, states=%d)",
        args.snapshot,
        len(snapshot.snippets),
        len(snapshot.state_values),
    )

    match = engine.evaluate(snapshot, respect_cooldowns=args.respect_cooldowns)
    if match is None:
        print("No rule matched the supplied snapshot.")
        return 1

    print(f"Matched rule: {match.rule.name} (priority {match.rule.priority})")
    if match.rule.description:
        print(f"Description: {match.rule.description}")

    action_description = engine.describe_action(match.action)
    print(f"Suggested action: {action_description}")

    if match.action.type == "scenario" and args.show_scenario_steps:
        scenario_name = match.action.params.get("name")
        if scenario_name:
            steps = engine.scenario_actions(scenario_name)
            if steps:
                print(f"Scenario '{scenario_name}' steps:")
                for index, step in enumerate(steps, start=1):
                    print(f"  {index}. {engine.describe_action(step)}")
            else:
                print(f"(Scenario '{scenario_name}' has no recorded steps.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

