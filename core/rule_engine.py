from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from .scenario_registry import ScenarioRegistry


_FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(slots=True)
class SnippetObservation:
    """Represents the latest detection state for a single snippet."""

    detected: bool
    score: Optional[float] = None


@dataclass(slots=True)
class RuleSnapshot:
    """Snapshot of the world when evaluating rules."""

    snippets: Dict[str, SnippetObservation]
    state_values: Dict[str, str]
    scenario_current: Optional[str] = None
    scenario_next: Optional[str] = None
    player_state: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snippet_detected(self, name: str) -> bool:
        entry = self.snippets.get(name)
        return bool(entry and entry.detected)

    def snippet_score(self, name: str) -> Optional[float]:
        entry = self.snippets.get(name)
        return entry.score if entry else None

    def state_text(self, name: str) -> Optional[str]:
        return self.state_values.get(name)

    def state_float(self, name: str) -> Optional[float]:
        value = self.state_values.get(name)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            match = _FLOAT_PATTERN.search(value)
            if not match:
                return None
            try:
                return float(match.group(0))
            except ValueError:
                return None


@dataclass(slots=True)
class RuleAction:
    """Action produced when a rule fires."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuleTest:
    """Single test within an AND-group."""

    kind: str
    params: Dict[str, Any]
    compiled_expression: Optional[ast.Expression] = None


@dataclass(slots=True)
class ConditionGroup:
    """OR-group of AND-tests. A rule fires if any group evaluates true."""

    tests: List[RuleTest]


@dataclass(slots=True)
class RuleDefinition:
    """Parsed representation of a rule in the ruleset."""

    identifier: str
    name: str
    priority: int
    description: Optional[str]
    cooldown_sec: float
    groups: List[ConditionGroup]
    action: RuleAction


@dataclass(slots=True)
class RuleMatch:
    """Result from evaluating rules."""

    rule: RuleDefinition
    action: RuleAction
    matched_group_index: int


class RuleEngine:
    """Evaluate rule-based automation decisions built from snippets and OCR results."""

    RULES_FILE = Path("rules") / "rules.json"

    _ALLOWED_EXPR_NODES: Tuple[type, ...] = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.IfExp,
        ast.Compare,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Subscript,
        ast.Slice,
        ast.Index,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.Set,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    _ALLOWED_EXPR_FUNCTIONS: Dict[str, str] = {
        "score": "snippet score lookup",
        "detected": "snippet detection lookup",
        "state_text": "state board text lookup",
        "state_float": "state board float lookup",
        "state_exists": "state board existence check",
        "player_state": "current player state",
        "scenario_current": "current scenario",
        "scenario_next": "next scenario",
        "len": "length",
        "max": "maximum",
        "min": "minimum",
        "abs": "absolute value",
    }

    def __init__(
        self,
        scenario_registry: ScenarioRegistry,
        rules_path: Optional[Path] = None,
    ) -> None:
        self._scenario_registry = scenario_registry
        self._rules_path = Path(rules_path) if rules_path else self.RULES_FILE
        self._rules: List[RuleDefinition] = []
        self._scenario_actions: Dict[str, List[RuleAction]] = {}
        self._last_triggered: Dict[str, float] = {}
        self.reload()

    # --------------------------------------------------------------------- public
    def reload(self) -> None:
        """Reload the rule definitions from disk."""
        try:
            data = json.loads(self._rules_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.info("Rule file {} not found; creating default stub.", self._rules_path)
            default = {"version": 1, "rules": [], "scenario_actions": {}}
            self._rules_path.parent.mkdir(parents=True, exist_ok=True)
            self._rules_path.write_text(json.dumps(default, indent=2), encoding="utf-8")
            data = default
        except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
            logger.error("Unable to parse rules file {}: {}", self._rules_path, exc)
            data = {"rules": [], "scenario_actions": {}}
        except OSError as exc:  # pragma: no cover - configuration error
            logger.error("Unable to read rules file {}: {}", self._rules_path, exc)
            data = {"rules": [], "scenario_actions": {}}

        self._rules = self._parse_rules(data.get("rules", []))
        self._scenario_actions = self._parse_scenario_actions(data.get("scenario_actions", {}))
        self._last_triggered.clear()

    def evaluate(
        self,
        snapshot: RuleSnapshot,
        *,
        now: Optional[float] = None,
        respect_cooldowns: bool = True,
    ) -> Optional[RuleMatch]:
        """Evaluate the rules against the provided snapshot."""
        now = now if now is not None else time.time()
        for rule in self._sorted_rules():
            if respect_cooldowns and not self._cooldown_elapsed(rule, now):
                continue
            match_idx = self._evaluate_rule(rule, snapshot)
            if match_idx is None:
                continue
            logger.debug("Rule '{}' matched (group #{})", rule.name, match_idx + 1)
            if respect_cooldowns and rule.cooldown_sec > 0:
                self._last_triggered[rule.identifier] = now
            return RuleMatch(rule=rule, action=rule.action, matched_group_index=match_idx)
        return None

    def scenario_actions(self, name: str) -> List[RuleAction]:
        """Return the sequence of actions associated with the named scenario."""
        return list(self._scenario_actions.get(name, []))

    def describe_action(self, action: RuleAction) -> str:
        """Return a human readable action summary."""
        if action.type == "tap":
            target = action.params.get("target_snippet")
            coords = action.params.get("coordinates")
            if target:
                return f"Tap snippet '{target}'"
            if coords:
                return f"Tap at {tuple(coords)}"
            return "Tap (unspecified target)"
        if action.type == "swipe":
            start = (
                action.params.get("start")
                or action.params.get("coordinates")
                or action.params.get("position")
            )
            end = (
                action.params.get("end")
                or action.params.get("end_coordinates")
                or action.params.get("destination")
            )
            duration = action.params.get("duration_ms", 300)
            start_tuple = tuple(start) if isinstance(start, (list, tuple)) else ("?", "?")
            end_tuple = tuple(end) if isinstance(end, (list, tuple)) else ("?", "?")
            return f"Swipe {start_tuple} → {end_tuple} ({duration}ms)"
        if action.type == "scenario":
            scenario = action.params.get("name", "<unnamed>")
            return f"Run scenario '{scenario}'"
        return action.type

    # ------------------------------------------------------------------ internals
    def _sorted_rules(self) -> Sequence[RuleDefinition]:
        return sorted(self._rules, key=lambda rule: (-rule.priority, rule.name.lower()))

    def _cooldown_elapsed(self, rule: RuleDefinition, now: float) -> bool:
        cooldown = max(0.0, rule.cooldown_sec)
        if cooldown <= 0:
            return True
        last = self._last_triggered.get(rule.identifier)
        if last is None:
            return True
        return (now - last) >= cooldown

    def _evaluate_rule(self, rule: RuleDefinition, snapshot: RuleSnapshot) -> Optional[int]:
        for index, group in enumerate(rule.groups):
            if self._evaluate_group(group, snapshot):
                return index
        return None

    def _evaluate_group(self, group: ConditionGroup, snapshot: RuleSnapshot) -> bool:
        for test in group.tests:
            if not self._evaluate_test(test, snapshot):
                return False
        return True

    # ------------------------------------------------------------------ test eval
    def _evaluate_test(self, test: RuleTest, snapshot: RuleSnapshot) -> bool:
        kind = test.kind
        params = test.params

        if kind == "snippet_detected":
            return snapshot.snippet_detected(params.get("name", ""))

        if kind == "snippet_not_detected":
            return not snapshot.snippet_detected(params.get("name", ""))

        if kind == "snippet_score_at_least":
            name = params.get("name", "")
            threshold = float(params.get("threshold", 0))
            score = snapshot.snippet_score(name)
            return score is not None and score >= threshold

        if kind == "snippet_score_at_most":
            name = params.get("name", "")
            threshold = float(params.get("threshold", 1.0))
            score = snapshot.snippet_score(name)
            return score is not None and score <= threshold

        if kind == "snippet_score_between":
            name = params.get("name", "")
            score = snapshot.snippet_score(name)
            minimum = float(params.get("min", 0.0))
            maximum = float(params.get("max", 1.0))
            if minimum > maximum:
                minimum, maximum = maximum, minimum
            return score is not None and minimum <= score <= maximum

        if kind == "state_exists":
            name = params.get("name", "")
            return name in snapshot.state_values and snapshot.state_values[name] != ""

        if kind == "state_text_equals":
            name = params.get("name", "")
            expected = params.get("value", "")
            case_sensitive = bool(params.get("case_sensitive", False))
            actual = snapshot.state_text(name)
            if actual is None:
                return False
            if not case_sensitive:
                return actual.lower() == str(expected).lower()
            return actual == str(expected)

        if kind == "state_text_contains":
            name = params.get("name", "")
            needle = params.get("value", "")
            case_sensitive = bool(params.get("case_sensitive", False))
            actual = snapshot.state_text(name)
            if actual is None:
                return False
            if not case_sensitive:
                return str(needle).lower() in actual.lower()
            return str(needle) in actual

        if kind == "state_numeric_at_least":
            name = params.get("name", "")
            threshold = self._coerce_float(params.get("threshold"))
            value = snapshot.state_float(name)
            return value is not None and threshold is not None and value >= threshold

        if kind == "state_numeric_at_most":
            name = params.get("name", "")
            threshold = self._coerce_float(params.get("threshold"))
            value = snapshot.state_float(name)
            return value is not None and threshold is not None and value <= threshold

        if kind == "state_numeric_between":
            name = params.get("name", "")
            minimum = self._coerce_float(params.get("min"))
            maximum = self._coerce_float(params.get("max"))
            value = snapshot.state_float(name)
            if value is None or minimum is None or maximum is None:
                return False
            if minimum > maximum:
                minimum, maximum = maximum, minimum
            return minimum <= value <= maximum

        if kind == "player_state_is":
            expected = params.get("value")
            if expected is None:
                return snapshot.player_state is None
            return (snapshot.player_state or "").lower() == str(expected).lower()

        if kind == "scenario_is":
            expected = params.get("value")
            return (snapshot.scenario_current or "").lower() == str(expected).lower()

        if kind == "scenario_next_is":
            expected = params.get("value")
            return (snapshot.scenario_next or "").lower() == str(expected).lower()

        if kind == "custom_expression":
            return self._evaluate_custom_expression(test, snapshot)

        logger.warning("Unknown rule test kind '{}'", kind)
        return False

    def _evaluate_custom_expression(self, test: RuleTest, snapshot: RuleSnapshot) -> bool:
        expression = test.compiled_expression
        if expression is None:
            raw = test.params.get("expression", "")
            try:
                expression = self._compile_expression(raw)
            except ValueError as exc:
                logger.error("Invalid rule expression '{}': {}", raw, exc)
                test.compiled_expression = None
                return False
            test.compiled_expression = expression

        env = self._expression_environment(snapshot)
        try:
            return bool(eval(compile(expression, "<rule-expression>", "eval"), {"__builtins__": {}}, env))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Rule expression evaluation failed: {}", exc)
            return False

    def _expression_environment(self, snapshot: RuleSnapshot) -> Dict[str, Any]:
        return {
            "score": lambda name: snapshot.snippet_score(str(name)),
            "detected": lambda name: snapshot.snippet_detected(str(name)),
            "state_text": lambda name: snapshot.state_text(str(name)),
            "state_float": lambda name: snapshot.state_float(str(name)),
            "state_exists": lambda name: snapshot.state_text(str(name)) is not None,
            "player_state": lambda: snapshot.player_state,
            "scenario_current": lambda: snapshot.scenario_current,
            "scenario_next": lambda: snapshot.scenario_next,
            "len": len,
            "max": max,
            "min": min,
            "abs": abs,
        }

    def _compile_expression(self, raw: str) -> ast.Expression:
        if not raw or not raw.strip():
            raise ValueError("expression is empty")
        node = ast.parse(raw, mode="eval")
        self._validate_expression(node)
        return node

    def _validate_expression(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if not isinstance(child, self._ALLOWED_EXPR_NODES):
                raise ValueError(f"Unsupported syntax element: {type(child).__name__}")
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id not in self._ALLOWED_EXPR_FUNCTIONS:
                        raise ValueError(f"Call to '{child.func.id}' is not permitted")
                else:
                    raise ValueError("Only simple function calls are permitted")

    def _parse_rules(self, raw_rules: Iterable[Dict[str, Any]]) -> List[RuleDefinition]:
        parsed: List[RuleDefinition] = []
        seen_ids: set[str] = set()

        for index, raw in enumerate(raw_rules):
            identifier = str(raw.get("id") or raw.get("name") or f"rule_{index}")
            if identifier in seen_ids:
                logger.warning("Duplicate rule identifier '{}'; skipping duplicate entry.", identifier)
                continue
            seen_ids.add(identifier)

            name = str(raw.get("name") or identifier)
            priority = int(raw.get("priority", 0))
            description = raw.get("description")
            cooldown = float(raw.get("cooldown_sec", 0))
            action_data = raw.get("action") or {}
            action = RuleAction(
                type=str(action_data.get("type") or "tap"),
                params={k: v for k, v in action_data.items() if k != "type"},
            )

            groups = self._parse_condition_groups(raw.get("conditions") or [])
            parsed.append(
                RuleDefinition(
                    identifier=identifier,
                    name=name,
                    priority=priority,
                    description=str(description) if description else None,
                    cooldown_sec=cooldown,
                    groups=groups,
                    action=action,
                )
            )

        return parsed

    def _parse_condition_groups(self, raw_groups: Iterable[Dict[str, Any]]) -> List[ConditionGroup]:
        groups: List[ConditionGroup] = []
        for raw_group in raw_groups:
            tests_data = raw_group.get("tests") or []
            tests: List[RuleTest] = []
            for raw_test in tests_data:
                kind = str(raw_test.get("type") or "snippet_detected")
                params = {k: v for k, v in raw_test.items() if k not in {"type", "expression"}}
                compiled: Optional[ast.Expression] = None
                if kind == "custom_expression":
                    expression = str(raw_test.get("expression") or "").strip()
                    params["expression"] = expression
                    if expression:
                        try:
                            compiled = self._compile_expression(expression)
                        except ValueError as exc:
                            logger.error("Invalid expression '{}': {}", expression, exc)
                            compiled = None
                tests.append(
                    RuleTest(
                        kind=kind,
                        params=params,
                        compiled_expression=compiled,
                    )
                )
            groups.append(ConditionGroup(tests=tests))
        if not groups:
            groups.append(ConditionGroup(tests=[]))  # Always true
        return groups

    def _parse_scenario_actions(
        self,
        raw_actions: Dict[str, Any],
    ) -> Dict[str, List[RuleAction]]:
        parsed: Dict[str, List[RuleAction]] = {}
        known_scenarios = {name.lower() for name in self._scenario_registry.scenarios()}

        for name, steps in raw_actions.items():
            if not isinstance(steps, list):
                logger.warning("Scenario '{}' steps must be a list; skipping.", name)
                continue
            if known_scenarios and name.lower() not in known_scenarios:
                logger.warning(
                    "Scenario '{}' referenced in rules but not registered. Add it via Scenario Manager.",
                    name,
                )
            parsed[name] = [
                RuleAction(
                    type=str(step.get("type") or "tap"),
                    params={k: v for k, v in step.items() if k != "type"},
                )
                for step in steps
                if isinstance(step, dict)
            ]
        return parsed

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        value_str = str(value).strip()
        if not value_str:
            return None
        try:
            return float(value_str)
        except ValueError:
            match = _FLOAT_PATTERN.search(value_str)
            if not match:
                return None
            try:
                return float(match.group(0))
            except ValueError:
                return None


