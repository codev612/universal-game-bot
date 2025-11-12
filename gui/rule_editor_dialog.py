from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.rule_engine import RuleEngine, RuleSnapshot, SnippetObservation
from core.scenario_registry import ScenarioRegistry


@dataclass
class _RuleRecord:
    identifier: str
    name: str
    description: str
    priority: int
    cooldown_sec: float
    conditions: List[Dict[str, Any]]
    action: Dict[str, Any]


SWIPE_STYLE_AUTO = "Auto (from history)"
SWIPE_STYLE_CUSTOM = "Custom (manual)"
SWIPE_STYLE_OPTIONS = [
    SWIPE_STYLE_AUTO,
    "Up → Down",
    "Down → Up",
    "Left → Right",
    "Right → Left",
    SWIPE_STYLE_CUSTOM,
]


def swipe_points_from_rect(rect: Tuple[int, int, int, int], style: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    quarter_w = max(4, w // 4)
    quarter_h = max(4, h // 4)

    if style == "Up → Down":
        start = (cx, y + quarter_h)
        end = (cx, y + h - quarter_h)
    elif style == "Down → Up":
        start = (cx, y + h - quarter_h)
        end = (cx, y + quarter_h)
    elif style == "Left → Right":
        start = (x + quarter_w, cy)
        end = (x + w - quarter_w, cy)
    elif style == "Right → Left":
        start = (x + w - quarter_w, cy)
        end = (x + quarter_w, cy)
    else:
        return None

    return (start, end)


def clamp_point_to_rect(point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y = point
    rx, ry, rw, rh = rect
    clamped_x = min(max(int(x), rx), rx + max(int(rw) - 1, 0))
    clamped_y = min(max(int(y), ry), ry + max(int(rh) - 1, 0))
    return (clamped_x, clamped_y)


class RuleEditorDialog(QDialog):
    """Dialog for editing rule-based automation logic."""

    def __init__(
        self,
        scenario_registry: ScenarioRegistry,
        rules_path: Optional[Path] = None,
        snippet_names: Optional[List[str]] = None,
        snippet_positions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        snippet_swipes: Optional[Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
        snapshot_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rule Designer")
        self.resize(840, 640)

        self._scenario_registry = scenario_registry
        self._rules_path = Path(rules_path) if rules_path else RuleEngine.RULES_FILE
        self._snapshot_path = (
            Path(snapshot_path) if snapshot_path else Path("rules") / "last_snapshot.json"
        )

        self._rules: List[_RuleRecord] = []
        self._scenario_actions: Dict[str, List[Dict[str, Any]]] = {}
        self._current_rule_index: Optional[int] = None
        self._is_loading_form = False
        self._snippet_names = sorted(set(snippet_names or []), key=str.lower)
        self._snippet_positions = dict(snippet_positions or {})
        self._snippet_swipes = {
            name: (
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
            )
            for name, (start, end) in (snippet_swipes or {}).items()
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == 2 and len(end) == 2
        }
        self._latest_swipe_template: Optional[Tuple[str, Tuple[int, int], Tuple[int, int]]] = None

        self._load_rules_file()
        self._build_ui()
        self._refresh_rule_list()
        self._refresh_scenario_selector()

    # ------------------------------------------------------------------ loading
    def _load_rules_file(self) -> None:
        try:
            data = json.loads(self._rules_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {"rules": [], "scenario_actions": {}}
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "Rules", f"Unable to parse rules file:\n{exc}")
            data = {"rules": [], "scenario_actions": {}}

        self._rules = [
            _RuleRecord(
                identifier=str(entry.get("id") or entry.get("name") or self._generate_rule_id()),
                name=str(entry.get("name") or entry.get("id") or "Unnamed Rule"),
                description=str(entry.get("description") or ""),
                priority=int(entry.get("priority", 0)),
                cooldown_sec=float(entry.get("cooldown_sec", 0.0)),
                conditions=list(entry.get("conditions") or []),
                action=dict(entry.get("action") or {"type": "tap"}),
            )
            for entry in data.get("rules", [])
        ]
        self._scenario_actions = {
            name: [dict(step) for step in steps if isinstance(step, dict)]
            for name, steps in (data.get("scenario_actions") or {}).items()
            if isinstance(steps, list)
        }

    def _write_rules_file(self) -> bool:
        payload = {
            "version": 1,
            "rules": [
                {
                    "id": record.identifier,
                    "name": record.name,
                    "description": record.description,
                    "priority": record.priority,
                    "cooldown_sec": record.cooldown_sec,
                    "conditions": record.conditions,
                    "action": record.action,
                }
                for record in self._rules
            ],
            "scenario_actions": self._scenario_actions,
        }

        try:
            self._rules_path.parent.mkdir(parents=True, exist_ok=True)
            self._rules_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return True
        except OSError as exc:
            QMessageBox.critical(self, "Save Rules", f"Unable to write rules file:\n{exc}")
            return False

    # --------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        tabs = QTabWidget(self)
        layout.addWidget(tabs, stretch=1)

        rules_tab = QWidget()
        rules_layout = QHBoxLayout(rules_tab)
        rules_layout.setContentsMargins(8, 8, 8, 8)
        rules_layout.setSpacing(12)

        # Rule list --------------------------------------------------------
        list_panel = QVBoxLayout()
        self._rule_list = QListWidget()
        self._rule_list.currentRowChanged.connect(self._on_rule_selected)
        list_panel.addWidget(self._rule_list, stretch=1)

        list_buttons = QHBoxLayout()
        add_rule_button = QPushButton("Add Rule")
        add_rule_button.clicked.connect(self._on_add_rule)
        remove_rule_button = QPushButton("Remove Rule")
        remove_rule_button.clicked.connect(self._on_remove_rule)
        list_buttons.addWidget(add_rule_button)
        list_buttons.addWidget(remove_rule_button)
        list_panel.addLayout(list_buttons)

        rules_layout.addLayout(list_panel, stretch=1)

        # Rule form --------------------------------------------------------
        form_panel = QVBoxLayout()
        form_panel.setSpacing(10)

        general_group = QGroupBox("General")
        general_layout = QFormLayout(general_group)

        self._rule_name_edit = QLineEdit()
        self._rule_name_edit.editingFinished.connect(self._on_form_changed)

        self._rule_id_edit = QLineEdit()
        self._rule_id_edit.setReadOnly(True)

        self._priority_spin = QSpinBox()
        self._priority_spin.setMinimum(-1000)
        self._priority_spin.setMaximum(1000)
        self._priority_spin.valueChanged.connect(self._on_form_changed)

        self._cooldown_spin = QDoubleSpinBox()
        self._cooldown_spin.setRange(0.0, 3600.0)
        self._cooldown_spin.setSuffix(" sec")
        self._cooldown_spin.setSingleStep(0.5)
        self._cooldown_spin.valueChanged.connect(self._on_form_changed)

        self._description_edit = QTextEdit()
        self._description_edit.textChanged.connect(self._on_form_changed)
        self._description_edit.setPlaceholderText("Optional note about this rule…")

        general_layout.addRow("Name:", self._rule_name_edit)
        general_layout.addRow("Identifier:", self._rule_id_edit)
        general_layout.addRow("Priority:", self._priority_spin)
        general_layout.addRow("Cooldown:", self._cooldown_spin)
        general_layout.addRow("Description:", self._description_edit)

        form_panel.addWidget(general_group)

        # Conditions -------------------------------------------------------
        condition_group = QGroupBox("Conditions")
        condition_layout = QVBoxLayout(condition_group)
        condition_layout.setSpacing(6)

        hint = QLabel("Rules trigger when any group evaluates true. Each group contains AND-ed tests.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        condition_layout.addWidget(hint)

        self._condition_list = QListWidget()
        self._condition_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        condition_layout.addWidget(self._condition_list, stretch=1)

        condition_buttons = QHBoxLayout()
        add_group_button = QPushButton("Add Group")
        add_group_button.clicked.connect(self._on_add_group)
        edit_group_button = QPushButton("Edit Group")
        edit_group_button.clicked.connect(self._on_edit_group)
        remove_group_button = QPushButton("Remove Group")
        remove_group_button.clicked.connect(self._on_remove_group)
        condition_buttons.addWidget(add_group_button)
        condition_buttons.addWidget(edit_group_button)
        condition_buttons.addWidget(remove_group_button)
        condition_layout.addLayout(condition_buttons)

        form_panel.addWidget(condition_group, stretch=1)

        # Action -----------------------------------------------------------
        action_group = QGroupBox("Action")
        action_layout = QFormLayout(action_group)

        self._action_type_combo = QComboBox()
        self._action_type_combo.addItems(["tap", "swipe", "scenario"])
        self._action_type_combo.currentTextChanged.connect(self._on_action_type_changed)

        self._action_target_combo = QComboBox()
        self._action_target_combo.setEditable(True)
        self._populate_snippet_combo(self._action_target_combo)
        target_line_edit = self._action_target_combo.lineEdit()
        if target_line_edit is not None:
            target_line_edit.setPlaceholderText("Snippet name (optional)")
            target_line_edit.editingFinished.connect(self._on_form_changed)
        self._action_target_combo.currentTextChanged.connect(self._on_action_target_changed)

        self._action_coordinates_edit = QLineEdit()
        self._action_coordinates_edit.setPlaceholderText("x,y (pixels)")
        self._action_coordinates_edit.editingFinished.connect(self._on_form_changed)

        self._action_swipe_end_edit = QLineEdit()
        self._action_swipe_end_edit.setPlaceholderText("x2,y2 (pixels)")
        self._action_swipe_end_edit.editingFinished.connect(self._on_form_changed)

        self._swipe_style_combo = QComboBox()
        self._swipe_style_combo.addItems(SWIPE_STYLE_OPTIONS)
        self._swipe_style_combo.currentTextChanged.connect(self._on_swipe_style_changed)

        self._action_duration_spin = QSpinBox()
        self._action_duration_spin.setRange(50, 5000)
        self._action_duration_spin.setSingleStep(50)
        self._action_duration_spin.valueChanged.connect(self._on_form_changed)
        self._action_duration_spin.setSuffix(" ms")

        self._action_scenario_combo = QComboBox()
        self._action_scenario_combo.currentTextChanged.connect(self._on_form_changed)

        action_layout.addRow("Type:", self._action_type_combo)
        action_layout.addRow("Target snippet:", self._action_target_combo)
        action_layout.addRow("Coordinates:", self._action_coordinates_edit)
        action_layout.addRow("Swipe end:", self._action_swipe_end_edit)
        action_layout.addRow("Swipe style:", self._swipe_style_combo)
        action_layout.addRow("Duration:", self._action_duration_spin)
        action_layout.addRow("Scenario:", self._action_scenario_combo)

        form_panel.addWidget(action_group)
        rules_layout.addLayout(form_panel, stretch=2)

        # Scenario actions tab --------------------------------------------
        scenarios_tab = QWidget()
        scenarios_layout = QVBoxLayout(scenarios_tab)
        scenarios_layout.setContentsMargins(8, 8, 8, 8)
        scenarios_layout.setSpacing(8)

        self._scenario_selector = QComboBox()
        self._scenario_selector.currentTextChanged.connect(self._on_scenario_selected)
        scenarios_layout.addWidget(self._scenario_selector)

        self._scenario_steps_list = QListWidget()
        self._scenario_steps_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        scenarios_layout.addWidget(self._scenario_steps_list, stretch=1)

        scenario_buttons = QHBoxLayout()
        add_step_button = QPushButton("Add Step")
        add_step_button.clicked.connect(self._on_add_scenario_step)
        edit_step_button = QPushButton("Edit Step")
        edit_step_button.clicked.connect(self._on_edit_scenario_step)
        remove_step_button = QPushButton("Remove Step")
        remove_step_button.clicked.connect(self._on_remove_scenario_step)
        scenario_buttons.addWidget(add_step_button)
        scenario_buttons.addWidget(edit_step_button)
        scenario_buttons.addWidget(remove_step_button)

        scenarios_layout.addLayout(scenario_buttons)

        tabs.addTab(rules_tab, "Rules")
        tabs.addTab(scenarios_tab, "Scenario Actions")

        # Footer -----------------------------------------------------------
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        self._test_button = buttons.addButton("Test Rule…", QDialogButtonBox.ButtonRole.ActionRole)
        self._test_button.clicked.connect(self._on_test_clicked)
        buttons.accepted.connect(self._on_save_clicked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_action_fields()

    # ----------------------------------------------------------------- rule list
    def _refresh_rule_list(self) -> None:
        current_id = None
        current_item = self._rule_list.currentItem()
        if current_item:
            current_id = current_item.data(Qt.ItemDataRole.UserRole)
        self._rule_list.blockSignals(True)
        try:
            self._rule_list.clear()
            for record in self._rules:
                item = QListWidgetItem(record.name)
                item.setData(Qt.ItemDataRole.UserRole, record.identifier)
                self._rule_list.addItem(item)
            if self._rules:
                if current_id:
                    row_to_select = next(
                        (
                            row
                            for row in range(self._rule_list.count())
                            if self._rule_list.item(row).data(Qt.ItemDataRole.UserRole) == current_id
                        ),
                        -1,
                    )
                    if row_to_select != -1:
                        self._rule_list.setCurrentRow(row_to_select)
                    else:
                        self._rule_list.setCurrentRow(0)
                else:
                    self._rule_list.setCurrentRow(0)
            else:
                self._current_rule_index = None
                self._clear_rule_form()
        finally:
            self._rule_list.blockSignals(False)
        if self._rules:
            selected_row = self._rule_list.currentRow()
            if selected_row < 0 or selected_row >= len(self._rules):
                selected_row = 0
            self._current_rule_index = selected_row
            self._load_rule_into_form(self._current_rule())

    def _current_rule(self) -> Optional[_RuleRecord]:
        if self._current_rule_index is None:
            return None
        if self._current_rule_index < 0 or self._current_rule_index >= len(self._rules):
            return None
        return self._rules[self._current_rule_index]

    def _on_rule_selected(self, row: int) -> None:
        if self._is_loading_form:
            return
        previous_index = self._current_rule_index
        previous_item: Optional[QListWidgetItem] = None
        if previous_index is not None and 0 <= previous_index < self._rule_list.count():
            previous_item = self._rule_list.item(previous_index)
        self._save_form_to_rule(previous_item)
        self._current_rule_index = row if 0 <= row < len(self._rules) else None
        self._load_rule_into_form(self._current_rule())

    def _generate_rule_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _on_add_rule(self) -> None:
        self._save_form_to_rule()
        new_rule = _RuleRecord(
            identifier=self._generate_rule_id(),
            name="New Rule",
            description="",
            priority=0,
            cooldown_sec=0.0,
            conditions=[{"tests": []}],
            action={"type": "tap"},
        )
        self._rules.append(new_rule)
        self._refresh_rule_list()
        self._rule_list.setCurrentRow(len(self._rules) - 1)

    def _on_remove_rule(self) -> None:
        row = self._rule_list.currentRow()
        if row < 0 or row >= len(self._rules):
            return
        record = self._rules[row]
        if QMessageBox.question(
            self,
            "Remove Rule",
            f"Remove rule '{record.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        del self._rules[row]
        self._refresh_rule_list()

    # ------------------------------------------------------------------ form io
    def _load_rule_into_form(self, record: Optional[_RuleRecord]) -> None:
        self._is_loading_form = True
        try:
            if record is None:
                self._clear_rule_form()
                return
            self._rule_name_edit.setText(record.name)
            self._rule_id_edit.setText(record.identifier)
            self._priority_spin.setValue(record.priority)
            self._cooldown_spin.setValue(record.cooldown_sec)
            self._description_edit.setPlainText(record.description)
            self._populate_condition_list(record.conditions)
            self._load_action_into_form(record.action)
        finally:
            self._is_loading_form = False
        self._apply_default_coordinates()

    def _populate_condition_list(self, groups: List[Dict[str, Any]]) -> None:
        self._condition_list.clear()
        if not groups:
            self._condition_list.addItem("(Always true)")
            return

        for index, group in enumerate(groups, start=1):
            tests = group.get("tests") or []
            summary = f"Group {index}: "
            if not tests:
                summary += "Always true"
            else:
                pieces = []
                for test in tests:
                    kind = test.get("type", "snippet_detected")
                    pieces.append(self._summarize_test(kind, test))
                summary += " AND ".join(pieces)
            self._condition_list.addItem(summary)

    def _load_action_into_form(self, action: Dict[str, Any]) -> None:
        action_type = str(action.get("type") or "tap")
        self._action_type_combo.setCurrentText(action_type)
        target_text = str(action.get("target_snippet") or "")
        if target_text and self._action_target_combo.findText(target_text, Qt.MatchFlag.MatchExactly) == -1:
            self._action_target_combo.addItem(target_text)
        self._action_target_combo.setCurrentText(target_text)
        coords = action.get("coordinates")
        self._action_coordinates_edit.setText(self._format_coords(coords))
        swipe_end = action.get("end") or action.get("end_coordinates") or action.get("swipe_end")
        self._action_swipe_end_edit.setText(self._format_coords(swipe_end))
        self._action_duration_spin.setValue(int(action.get("duration_ms", 300)))
        style = str(action.get("swipe_style") or SWIPE_STYLE_AUTO)
        if style not in SWIPE_STYLE_OPTIONS:
            style = SWIPE_STYLE_AUTO
        self._swipe_style_combo.setCurrentText(style)
        scenario_name = str(action.get("name") or "")
        index = self._action_scenario_combo.findText(scenario_name)
        if index == -1:
            self._action_scenario_combo.addItem(scenario_name)
            index = self._action_scenario_combo.count() - 1
        self._action_scenario_combo.setCurrentIndex(index if index >= 0 else 0)
        self._update_action_fields()
        self._apply_default_coordinates()

    def _save_form_to_rule(self, target_item: Optional[QListWidgetItem] = None) -> None:
        record = self._current_rule()
        if record is None or self._is_loading_form:
            return

        record.name = self._rule_name_edit.text().strip() or "Unnamed Rule"
        record.description = self._description_edit.toPlainText().strip()
        record.priority = int(self._priority_spin.value())
        record.cooldown_sec = float(self._cooldown_spin.value())
        record.conditions = self._extract_conditions()
        record.action = self._extract_action()

        if target_item is None and self._current_rule_index is not None:
            if 0 <= self._current_rule_index < self._rule_list.count():
                target_item = self._rule_list.item(self._current_rule_index)
        if target_item:
            target_item.setText(record.name)

    def _extract_conditions(self) -> List[Dict[str, Any]]:
        record = self._current_rule()
        if record is None:
            return []
        return record.conditions

    def _extract_action(self) -> Dict[str, Any]:
        action_type = self._action_type_combo.currentText()
        target = self._action_target_combo.currentText().strip()
        coords = self._parse_coords(self._action_coordinates_edit.text())
        duration = int(self._action_duration_spin.value())
        end = self._parse_coords(self._action_swipe_end_edit.text())
        scenario = self._action_scenario_combo.currentText().strip()
        style = self._current_swipe_style()

        data: Dict[str, Any] = {"type": action_type}
        if target:
            data["target_snippet"] = target
            if target not in self._snippet_names:
                self._snippet_names.append(target)
                self._snippet_names.sort(key=str.lower)
                self._populate_snippet_combo(self._action_target_combo)
                self._action_target_combo.setCurrentText(target)
            rect = self._snippet_positions.get(target)
            start_point = end_point = None
            if style != SWIPE_STYLE_AUTO and rect:
                points = swipe_points_from_rect(rect, style)
                if points:
                    start_point, end_point = points
            if start_point is None:
                template = self._snippet_swipes.get(target)
                if template:
                    start_point, end_point = template
            if start_point is None and rect:
                fallback_points = swipe_points_from_rect(rect, "Left → Right")
                if fallback_points:
                    start_point, end_point = fallback_points
                else:
                    start_point = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
                    offset_dx = max(4, rect[2] // 6) if rect[2] > 0 else 10
                    offset_dy = max(4, rect[3] // 6) if rect[3] > 0 else 10
                    end_point = (
                        start_point[0] + offset_dx,
                        start_point[1] - offset_dy if rect[3] >= rect[2] else start_point[1] + offset_dy,
                    )
            if start_point and rect:
                start_point = clamp_point_to_rect(start_point, rect)
            if end_point and rect:
                end_point = clamp_point_to_rect(end_point, rect)
            if start_point and not coords:
                coords = [int(start_point[0]), int(start_point[1])]
            if action_type == "swipe" and end_point and not end:
                end = [int(end_point[0]), int(end_point[1])]
        if coords:
            data["coordinates"] = coords
        if action_type == "swipe":
            if end:
                data["end"] = end
            data["duration_ms"] = duration
            data["swipe_style"] = style
            if target and coords and end:
                self._snippet_swipes[target] = (tuple(coords), tuple(end))
        if action_type == "scenario" and scenario:
            data["name"] = scenario
        return data

    def _clear_rule_form(self) -> None:
        self._is_loading_form = True
        try:
            self._rule_name_edit.clear()
            self._rule_id_edit.clear()
            self._priority_spin.setValue(0)
            self._cooldown_spin.setValue(0.0)
            self._description_edit.clear()
            self._condition_list.clear()
            self._condition_list.addItem("(No rule selected)")
            self._load_action_into_form({"type": "tap"})
        finally:
            self._is_loading_form = False

    # -------------------------------------------------------------- conditions
    def _on_add_group(self) -> None:
        record = self._current_rule()
        if record is None:
            return
        dialog = _ConditionGroupDialog(
            self,
            snippet_names=self._snippet_names,
            snippet_positions=self._snippet_positions,
            snippet_swipes=self._snippet_swipes,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        record.conditions.append({"tests": dialog.tests})
        self._merge_snippet_names(dialog.collect_snippet_names())
        self._merge_snippet_positions(dialog.collect_snippet_positions())
        self._merge_snippet_swipes(dialog.collect_snippet_swipes())
        self._populate_condition_list(record.conditions)

    def _on_edit_group(self) -> None:
        record = self._current_rule()
        if record is None:
            return
        row = self._condition_list.currentRow()
        if row < 0 or row >= len(record.conditions):
            return
        dialog = _ConditionGroupDialog(
            self,
            tests=list(record.conditions[row].get("tests") or []),
            snippet_names=self._snippet_names,
            snippet_positions=self._snippet_positions,
            snippet_swipes=self._snippet_swipes,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        record.conditions[row] = {"tests": dialog.tests}
        self._merge_snippet_names(dialog.collect_snippet_names())
        self._merge_snippet_positions(dialog.collect_snippet_positions())
        self._merge_snippet_swipes(dialog.collect_snippet_swipes())
        self._populate_condition_list(record.conditions)

    def _on_remove_group(self) -> None:
        record = self._current_rule()
        if record is None:
            return
        row = self._condition_list.currentRow()
        if row < 0 or row >= len(record.conditions):
            return
        record.conditions.pop(row)
        self._populate_condition_list(record.conditions)

    # ------------------------------------------------------------------ action
    def _on_action_type_changed(self, _text: str) -> None:
        self._update_action_fields()
        self._apply_default_coordinates()
        self._on_form_changed()

    def _update_action_fields(self) -> None:
        action_type = self._action_type_combo.currentText()
        is_tap = action_type == "tap"
        is_swipe = action_type == "swipe"
        is_scenario = action_type == "scenario"

        self._action_target_combo.setEnabled(is_tap or is_swipe)
        self._action_coordinates_edit.setEnabled(is_tap or is_swipe)
        self._action_swipe_end_edit.setEnabled(is_swipe)
        self._action_duration_spin.setEnabled(is_swipe)
        self._swipe_style_combo.setEnabled(is_swipe)
        self._swipe_style_combo.setVisible(is_swipe)
        self._action_scenario_combo.setEnabled(is_scenario)

    # --------------------------------------------------------------- scenarios
    def _refresh_scenario_selector(self) -> None:
        self._action_scenario_combo.clear()
        self._scenario_selector.clear()

        scenarios = self._scenario_registry.scenarios()
        if not scenarios:
            self._action_scenario_combo.addItem("")
            return

        for scenario in scenarios:
            self._action_scenario_combo.addItem(scenario)
            self._scenario_selector.addItem(scenario)

        current = self._scenario_selector.currentText()
        if not current and scenarios:
            self._scenario_selector.setCurrentIndex(0)
        self._refresh_scenario_steps()

    def _on_scenario_selected(self, _name: str) -> None:
        self._refresh_scenario_steps()

    def _refresh_scenario_steps(self) -> None:
        name = self._scenario_selector.currentText()
        steps = self._scenario_actions.get(name, [])
        self._scenario_steps_list.clear()
        for step in steps:
            self._scenario_steps_list.addItem(self._summarize_action(step))

    def _on_add_scenario_step(self) -> None:
        name = self._scenario_selector.currentText()
        if not name:
            QMessageBox.information(self, "Scenario Actions", "Select a scenario first.")
            return
        dialog = _ActionEditorDialog(
            self,
            scenario_mode=False,
            snippet_names=self._snippet_names,
            snippet_positions=self._snippet_positions,
            snippet_swipes=self._snippet_swipes,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._scenario_actions.setdefault(name, []).append(dialog.action_data)
        current_text = self._action_target_combo.currentText()
        self._populate_snippet_combo(self._action_target_combo)
        self._action_target_combo.setCurrentText(current_text)
        if dialog._latest_swipe_template:
            target, start_point, end_point = dialog._latest_swipe_template
            self._merge_snippet_swipes({target: (start_point, end_point)})
        self._refresh_scenario_steps()

    def _on_edit_scenario_step(self) -> None:
        name = self._scenario_selector.currentText()
        if not name:
            return
        steps = self._scenario_actions.setdefault(name, [])
        row = self._scenario_steps_list.currentRow()
        if row < 0 or row >= len(steps):
            return
        dialog = _ActionEditorDialog(
            self,
            initial=steps[row],
            scenario_mode=False,
            snippet_names=self._snippet_names,
            snippet_positions=self._snippet_positions,
            snippet_swipes=self._snippet_swipes,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        steps[row] = dialog.action_data
        current_text = self._action_target_combo.currentText()
        self._populate_snippet_combo(self._action_target_combo)
        self._action_target_combo.setCurrentText(current_text)
        if dialog._latest_swipe_template:
            target, start_point, end_point = dialog._latest_swipe_template
            self._merge_snippet_swipes({target: (start_point, end_point)})
        self._refresh_scenario_steps()

    def _on_remove_scenario_step(self) -> None:
        name = self._scenario_selector.currentText()
        if not name:
            return
        steps = self._scenario_actions.setdefault(name, [])
        row = self._scenario_steps_list.currentRow()
        if row < 0 or row >= len(steps):
            return
        steps.pop(row)
        self._refresh_scenario_steps()

    # ----------------------------------------------------------------- helpers
    def _summarize_test(self, kind: str, params: Dict[str, Any]) -> str:
        if kind == "snippet_detected":
            return f"{params.get('name', '<snippet>')} detected"
        if kind == "snippet_not_detected":
            return f"{params.get('name', '<snippet>')} not detected"
        if kind == "snippet_score_at_least":
            return f"{params.get('name', '<snippet>')} score ≥ {params.get('threshold', 0)}"
        if kind == "state_numeric_at_least":
            return f"{params.get('name', '<state>')} ≥ {params.get('threshold', 0)}"
        if kind == "state_numeric_at_most":
            return f"{params.get('name', '<state>')} ≤ {params.get('threshold', 0)}"
        if kind == "state_numeric_between":
            return (
                f"{params.get('name', '<state>')} between "
                f"{params.get('min', 0)} and {params.get('max', 0)}"
            )
        if kind == "state_text_equals":
            return f"{params.get('name', '<state>')} == '{params.get('value', '')}'"
        if kind == "state_text_contains":
            return f"{params.get('name', '<state>')} contains '{params.get('value', '')}'"
        if kind == "player_state_is":
            return f"player state == '{params.get('value', '')}'"
        if kind == "scenario_is":
            return f"scenario == '{params.get('value', '')}'"
        if kind == "scenario_next_is":
            return f"next scenario == '{params.get('value', '')}'"
        if kind == "custom_expression":
            return f"expr({params.get('expression', '')})"
        return kind

    def _summarize_action(self, action: Dict[str, Any]) -> str:
        action_type = str(action.get("type") or "tap")
        if action_type == "tap":
            target = action.get("target_snippet")
            coords = action.get("coordinates")
            if target:
                return f"Tap '{target}'"
            if coords:
                return f"Tap at {tuple(coords)}"
            return "Tap (unspecified)"
        if action_type == "swipe":
            start = action.get("coordinates") or action.get("start")
            end = action.get("end")
            duration = action.get("duration_ms", 300)
            return f"Swipe {tuple(start or ('?', '?'))} → {tuple(end or ('?', '?'))} ({duration}ms)"
        if action_type == "scenario":
            return f"Scenario → {action.get('name', '<unnamed>')}"
        return action_type

    def _format_coords(self, value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"{value[0]},{value[1]}"
        return str(value)

    def _parse_coords(self, text: str) -> Optional[List[int]]:
        text = text.strip()
        if not text:
            return None
        if "," not in text:
            QMessageBox.warning(self, "Coordinates", "Use comma-separated coordinates, e.g. 540,960.")
            return None
        try:
            parts = [int(float(part.strip())) for part in text.split(",", 1)]
        except ValueError:
            QMessageBox.warning(self, "Coordinates", "Invalid coordinate value.")
            return None
        return parts

    def _populate_snippet_combo(self, combo: QComboBox) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("")
            for name in self._snippet_names:
                combo.addItem(name)
        finally:
            combo.blockSignals(False)

    def _merge_snippet_names(self, names: List[str]) -> None:
        if not names:
            return
        updated = set(self._snippet_names)
        changed = False
        for name in names:
            if name and name not in updated:
                updated.add(name)
                changed = True
        if changed:
            self._snippet_names = sorted(updated, key=str.lower)

    def _merge_snippet_positions(self, positions: Dict[str, Tuple[int, int, int, int]]) -> None:
        if not positions:
            return
        updated = dict(self._snippet_positions)
        changed = False
        for name, rect in positions.items():
            if name and rect and name not in updated:
                updated[name] = rect
                changed = True
        if changed:
            self._snippet_positions = dict(sorted(updated.items(), key=lambda item: item[0].lower()))

    def _merge_snippet_swipes(self, templates: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]) -> None:
        if not templates:
            return
        updated = dict(self._snippet_swipes)
        changed = False
        for name, (start, end) in templates.items():
            if name and name not in updated:
                updated[name] = (
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                )
                changed = True
        if changed:
            self._snippet_swipes = dict(sorted(updated.items(), key=lambda item: item[0].lower()))
            self._populate_snippet_combo(self._action_target_combo)

    def _merge_snippet_positions(self, positions: Dict[str, Tuple[int, int, int, int]]) -> None:
        changed = False
        for name, rect in positions.items():
            if name and rect and name not in self._snippet_positions:
                self._snippet_positions[name] = rect
                changed = True
        if changed:
            self._snippet_positions = dict(sorted(self._snippet_positions.items(), key=lambda item: item[0].lower()))
            self._apply_default_coordinates()

    def _merge_snippet_swipes(self, templates: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]) -> None:
        if not templates:
            return
        changed = False
        for name, (start, end) in templates.items():
            if not name:
                continue
            if name not in self._snippet_swipes:
                self._snippet_swipes[name] = (
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                )
                changed = True
        if changed:
            self._apply_default_coordinates()

    def _apply_default_coordinates(self) -> None:
        if self._is_loading_form:
            return
        target = self._action_target_combo.currentText().strip()
        if not target:
            return
        rect = self._snippet_positions.get(target)
        style = self._current_swipe_style()

        if style == SWIPE_STYLE_CUSTOM:
            return
        start_point = end_point = None
        if style != SWIPE_STYLE_AUTO and rect:
            points = swipe_points_from_rect(rect, style)
            if points:
                start_point, end_point = points
        if start_point is None:
            swipe_template = self._snippet_swipes.get(target)
            if swipe_template and swipe_template[0] != swipe_template[1]:
                start_point, end_point = swipe_template
        if start_point is None and rect:
            points = swipe_points_from_rect(rect, "Left → Right")
            if points:
                start_point, end_point = points
            else:
                centre = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
                offset_dx = max(4, rect[2] // 6) if rect[2] > 0 else 10
                offset_dy = max(4, rect[3] // 6) if rect[3] > 0 else 10
                start_point = centre
                end_point = (
                    centre[0] + offset_dx,
                    centre[1] - offset_dy if rect[3] >= rect[2] else centre[1] + offset_dy,
                )
        if start_point is None:
            return
        self._action_coordinates_edit.setText(f"{start_point[0]},{start_point[1]}")
        if self._action_type_combo.currentText() == "swipe" and end_point is not None:
            self._action_swipe_end_edit.setText(f"{end_point[0]},{end_point[1]}")

    def _on_action_target_changed(self, _text: str) -> None:
        if self._is_loading_form:
            return
        self._apply_default_coordinates()
        self._on_form_changed()

    def _on_swipe_style_changed(self, _text: str) -> None:
        if self._is_loading_form:
            return
        self._apply_default_coordinates()
        self._on_form_changed()

    def _current_swipe_style(self) -> str:
        return self._swipe_style_combo.currentText()

    def _on_form_changed(self) -> None:
        if self._is_loading_form:
            return
        self._save_form_to_rule()

    def _on_save_clicked(self) -> None:
        self._save_form_to_rule()
        if not self._write_rules_file():
            return
        QMessageBox.information(self, "Rules", "Rules saved successfully.")
        self.accept()

    def _on_test_clicked(self) -> None:
        self._save_form_to_rule()
        snapshot_path = self._snapshot_path
        if not snapshot_path.exists():
            QMessageBox.information(
                self,
                "Rule Test",
                f"No snapshot found at {snapshot_path}. Capture a screenshot first.",
            )
            return
        try:
            snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Rule Test", f"Unable to load snapshot:\n{exc}")
            return

        snippets_payload = snapshot_payload.get("snippets") or {}
        snippets: Dict[str, SnippetObservation] = {}
        for name, data in snippets_payload.items():
            if not isinstance(data, dict):
                continue
            snippets[name] = SnippetObservation(
                detected=bool(data.get("detected")),
                score=data.get("score"),
            )

        snapshot = RuleSnapshot(
            snippets=snippets,
            state_values=snapshot_payload.get("state_values") or {},
            scenario_current=snapshot_payload.get("scenario_current"),
            scenario_next=snapshot_payload.get("scenario_next"),
            player_state=snapshot_payload.get("player_state"),
            metadata=snapshot_payload.get("metadata") or {},
        )

        engine = RuleEngine(self._scenario_registry, rules_path=self._rules_path)
        rules_payload = [
            {
                "id": record.identifier,
                "name": record.name,
                "description": record.description,
                "priority": record.priority,
                "cooldown_sec": record.cooldown_sec,
                "conditions": record.conditions,
                "action": record.action,
            }
            for record in self._rules
        ]
        engine._rules = engine._parse_rules(rules_payload)
        engine._scenario_actions = engine._parse_scenario_actions(self._scenario_actions)

        result = engine.evaluate(snapshot, respect_cooldowns=False)
        if result is None:
            QMessageBox.information(
                self,
                "Rule Test",
                "No rule matched the last snapshot.",
            )
            return

        rule = result.rule
        action_text = engine.describe_action(result.action)
        current = self._current_rule()
        extra = ""
        if current and rule.identifier == current.identifier:
            extra = "\n(This is the currently selected rule.)"
        QMessageBox.information(
            self,
            "Rule Test",
            f"Matched rule '{rule.name}'\nAction: {action_text}{extra}",
        )


# --------------------------------------------------------------------------- util
class _ConditionGroupDialog(QDialog):
    """Dialog for editing an AND group of tests."""

    def __init__(
        self,
        parent: QWidget,
        tests: Optional[List[Dict[str, Any]]] = None,
        snippet_names: Optional[List[str]] = None,
        snippet_positions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        snippet_swipes: Optional[Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Condition Group")
        self.resize(520, 400)

        self.tests: List[Dict[str, Any]] = tests or []
        self._snippet_names = sorted(set(snippet_names or []), key=str.lower)
        self._snippet_positions = dict(snippet_positions or {})
        self._snippet_swipes = {
            name: (
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
            )
            for name, (start, end) in (snippet_swipes or {}).items()
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == 2 and len(end) == 2
        }

        layout = QVBoxLayout(self)
        self._test_list = QListWidget()
        self._test_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self._test_list, stretch=1)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add Test")
        add_button.clicked.connect(self._on_add_test)
        edit_button = QPushButton("Edit Test")
        edit_button.clicked.connect(self._on_edit_test)
        remove_button = QPushButton("Remove Test")
        remove_button.clicked.connect(self._on_remove_test)
        buttons.addWidget(add_button)
        buttons.addWidget(edit_button)
        buttons.addWidget(remove_button)
        layout.addLayout(buttons)

        close_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        close_buttons.accepted.connect(self.accept)
        close_buttons.rejected.connect(self.reject)
        layout.addWidget(close_buttons)

        self._refresh_test_list()

    def _refresh_test_list(self) -> None:
        self._test_list.clear()
        for test in self.tests:
            kind = test.get("type", "snippet_detected")
            summary = f"{kind}: " + json.dumps({k: v for k, v in test.items() if k != "type"})
            self._test_list.addItem(summary)

    def _on_add_test(self) -> None:
        dialog = _TestEditorDialog(
            self,
            snippet_names=self._snippet_names,
            snippet_positions=self._snippet_positions,
            snippet_swipes=self._snippet_swipes,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.tests.append(dialog.test_data)
        self._merge_snippet_names(dialog.collect_snippet_names())
        self._merge_snippet_positions(dialog.collect_snippet_positions())
        self._merge_snippet_swipes(dialog.collect_snippet_swipes())
        self._refresh_test_list()

    def _on_edit_test(self) -> None:
        row = self._test_list.currentRow()
        if row < 0 or row >= len(self.tests):
            return
        dialog = _TestEditorDialog(
            self,
            initial=self.tests[row],
            snippet_names=self._snippet_names,
            snippet_positions=self._snippet_positions,
            snippet_swipes=self._snippet_swipes,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.tests[row] = dialog.test_data
        self._merge_snippet_names(dialog.collect_snippet_names())
        self._merge_snippet_positions(dialog.collect_snippet_positions())
        self._merge_snippet_swipes(dialog.collect_snippet_swipes())
        self._refresh_test_list()

    def _on_remove_test(self) -> None:
        row = self._test_list.currentRow()
        if row < 0 or row >= len(self.tests):
            return
        self.tests.pop(row)
        self._refresh_test_list()

    def collect_snippet_names(self) -> List[str]:
        names = set(self._snippet_names)
        for test in self.tests:
            name = test.get("name")
            if isinstance(name, str) and name:
                names.add(name)
        return sorted(names, key=str.lower)

    def collect_snippet_positions(self) -> Dict[str, Tuple[int, int, int, int]]:
        positions: Dict[str, Tuple[int, int, int, int]] = dict(self._snippet_positions)
        for test in self.tests:
            if not test.get("_use_rect"):
                continue
            name = test.get("name")
            rect = test.get("_rect")
            if (
                isinstance(name, str)
                and name
                and isinstance(rect, (list, tuple))
                and len(rect) == 4
            ):
                try:
                    x, y, w, h = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                except (TypeError, ValueError):
                    continue
                positions[name] = (x, y, w, h)
        return positions

    def collect_snippet_swipes(self) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
        return dict(self._snippet_swipes)

    def _merge_snippet_positions(self, positions: Dict[str, Tuple[int, int, int, int]]) -> None:
        if not positions:
            return
        updated = dict(self._snippet_positions)
        changed = False
        for name, rect in positions.items():
            if name and rect and name not in updated:
                updated[name] = rect
                changed = True
        if changed:
            self._snippet_positions = dict(sorted(updated.items(), key=lambda item: item[0].lower()))

    def _merge_snippet_names(self, names: List[str]) -> None:
        if not names:
            return
        updated = set(self._snippet_names)
        changed = False
        for name in names:
            if name and name not in updated:
                updated.add(name)
                changed = True
        if changed:
            self._snippet_names = sorted(updated, key=str.lower)

    def _merge_snippet_swipes(self, swipes: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]) -> None:
        if not swipes:
            return
        updated = dict(self._snippet_swipes)
        changed = False
        for name, swipe in swipes.items():
            if name and swipe and name not in updated:
                updated[name] = swipe
                changed = True
        if changed:
            self._snippet_swipes = dict(sorted(updated.items(), key=lambda item: item[0].lower()))


class _TestEditorDialog(QDialog):
    """Dialog for adding/editing a single test."""

    _TEST_KIND_LABELS = {
        "snippet_detected": "Snippet detected",
        "snippet_not_detected": "Snippet not detected",
        "snippet_score_at_least": "Snippet score ≥ threshold",
        "snippet_score_at_most": "Snippet score ≤ threshold",
        "state_numeric_at_least": "State value ≥ threshold",
        "state_numeric_at_most": "State value ≤ threshold",
        "state_numeric_between": "State value between min/max",
        "state_text_equals": "State text equals",
        "state_text_contains": "State text contains",
        "player_state_is": "Player state equals",
        "scenario_is": "Current scenario equals",
        "scenario_next_is": "Next scenario equals",
        "custom_expression": "Custom expression",
    }

    def __init__(
        self,
        parent: QWidget,
        initial: Optional[Dict[str, Any]] = None,
        snippet_names: Optional[List[str]] = None,
        snippet_positions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        snippet_swipes: Optional[Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Test")
        self.setModal(True)
        self.resize(420, 320)

        initial = initial or {"type": "snippet_detected"}
        kind = initial.get("type", "snippet_detected")
        self._snippet_names = sorted(set(snippet_names or []), key=str.lower)
        self._snippet_positions = dict(snippet_positions or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._kind_combo = QComboBox()
        self._kind_combo.addItems(list(self._TEST_KIND_LABELS.keys()))
        self._kind_combo.setCurrentText(kind)
        self._kind_combo.currentTextChanged.connect(self._update_field_visibility)
        form.addRow("Test type:", self._kind_combo)

        self._snippet_name_combo = QComboBox()
        self._snippet_name_combo.setEditable(True)
        self._populate_snippet_combo()
        snippet_initial = str(initial.get("name", ""))
        if snippet_initial and self._snippet_name_combo.findText(snippet_initial, Qt.MatchFlag.MatchExactly) == -1:
            self._snippet_name_combo.addItem(snippet_initial)
        self._snippet_name_combo.setCurrentText(snippet_initial)
        self._snippet_name_combo.currentTextChanged.connect(self._on_snippet_changed)
        form.addRow("Snippet name:", self._snippet_name_combo)

        self._use_rect_checkbox = QCheckBox("Use snippet centroid for coordinates")
        self._use_rect_checkbox.setChecked(bool(initial.get("_use_rect")))
        form.addRow("", self._use_rect_checkbox)

        self._state_name_edit = QLineEdit(initial.get("name", ""))
        form.addRow("State name:", self._state_name_edit)

        self._value_edit = QLineEdit(str(initial.get("value", "")))
        form.addRow("Value:", self._value_edit)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._threshold_spin.setDecimals(4)
        self._threshold_spin.setValue(float(initial.get("threshold", 0.0)))
        form.addRow("Threshold:", self._threshold_spin)

        self._min_spin = QDoubleSpinBox()
        self._min_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._min_spin.setDecimals(4)
        self._min_spin.setValue(float(initial.get("min", 0.0)))
        form.addRow("Min value:", self._min_spin)

        self._max_spin = QDoubleSpinBox()
        self._max_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._max_spin.setDecimals(4)
        self._max_spin.setValue(float(initial.get("max", 0.0)))
        form.addRow("Max value:", self._max_spin)

        self._expression_edit = QTextEdit(initial.get("expression", ""))
        self._expression_edit.setPlaceholderText(
            "Expression using helpers like detected('Snippet'), score('Snippet'),\n"
            "state_float('Health') < 0.5, player_state() == 'low blood', etc."
        )
        form.addRow("Expression:", self._expression_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.test_data: Dict[str, Any] = dict(initial)
        self._update_field_visibility(kind)

    def _update_field_visibility(self, kind: str) -> None:
        snippet_fields = kind.startswith("snippet")
        state_fields = kind.startswith("state")
        value_fields = kind in {
            "state_text_equals",
            "state_text_contains",
            "player_state_is",
            "scenario_is",
            "scenario_next_is",
        }
        threshold_fields = kind in {
            "snippet_score_at_least",
            "snippet_score_at_most",
            "state_numeric_at_least",
            "state_numeric_at_most",
        }
        between_fields = kind in {"state_numeric_between"}

        self._set_row_visible(self._snippet_name_combo, snippet_fields)
        self._set_row_visible(self._state_name_edit, state_fields)
        self._set_row_visible(self._value_edit, value_fields)
        self._set_row_visible(self._threshold_spin, threshold_fields)
        self._set_row_visible(self._min_spin, between_fields)
        self._set_row_visible(self._max_spin, between_fields)
        self._expression_edit.setVisible(kind == "custom_expression")
        self._use_rect_checkbox.setVisible(snippet_fields)

    @staticmethod
    def _set_row_visible(widget: QWidget, visible: bool) -> None:
        parent = widget.parentWidget()
        if parent is not None:
            parent.setVisible(visible)
        widget.setVisible(visible)

    def _populate_snippet_combo(self) -> None:
        self._snippet_name_combo.blockSignals(True)
        try:
            self._snippet_name_combo.clear()
            self._snippet_name_combo.addItem("")
            for name in self._snippet_names:
                self._snippet_name_combo.addItem(name)
        finally:
            self._snippet_name_combo.blockSignals(False)

    def _on_snippet_changed(self, text: str) -> None:
        rect = self._snippet_positions.get(text.strip())
        if rect:
            self._use_rect_checkbox.setEnabled(True)
            if not self._use_rect_checkbox.isChecked():
                self._use_rect_checkbox.setChecked(True)
        else:
            self._use_rect_checkbox.setChecked(False)
            self._use_rect_checkbox.setEnabled(False)

    def _on_accept(self) -> None:
        kind = self._kind_combo.currentText()
        data: Dict[str, Any] = {"type": kind}

        if kind.startswith("snippet"):
            name = self._snippet_name_combo.currentText().strip()
            if not name:
                QMessageBox.warning(self, "Test", "Provide a snippet name.")
                return
            data["name"] = name
            if self._use_rect_checkbox.isChecked():
                rect = self._snippet_positions.get(name)
                if rect:
                    data["_use_rect"] = True
                    data["_rect"] = list(rect)

        if kind.startswith("state"):
            name = self._state_name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Test", "Provide a state board name.")
                return
            data["name"] = name

        if kind in {"state_text_equals", "state_text_contains", "player_state_is", "scenario_is", "scenario_next_is"}:
            data["value"] = self._value_edit.text().strip()

        if kind in {"snippet_score_at_least", "snippet_score_at_most", "state_numeric_at_least", "state_numeric_at_most"}:
            data["threshold"] = float(self._threshold_spin.value())

        if kind == "state_numeric_between":
            data["min"] = float(self._min_spin.value())
            data["max"] = float(self._max_spin.value())

        if kind == "custom_expression":
            expression = self._expression_edit.toPlainText().strip()
            if not expression:
                QMessageBox.warning(self, "Expression", "Provide a Python expression.")
                return
            data["expression"] = expression

        self.test_data = data
        self.accept()

    def collect_snippet_names(self) -> List[str]:
        """Collect snippet names used in this test."""
        names = set()
        if hasattr(self, 'test_data') and self.test_data:
            name = self.test_data.get("name")
            if isinstance(name, str) and name:
                names.add(name)
        return sorted(names, key=str.lower)

    def collect_snippet_positions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Collect snippet positions from this test."""
        positions: Dict[str, Tuple[int, int, int, int]] = {}
        if hasattr(self, 'test_data') and self.test_data:
            if not self.test_data.get("_use_rect"):
                return positions
            name = self.test_data.get("name")
            rect = self.test_data.get("_rect")
            if (
                isinstance(name, str)
                and name
                and isinstance(rect, (list, tuple))
                and len(rect) == 4
            ):
                try:
                    x, y, w, h = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                    positions[name] = (x, y, w, h)
                except (TypeError, ValueError):
                    pass
        return positions

    def collect_snippet_swipes(self) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
        return {}


class _ActionEditorDialog(QDialog):
    """Dialog for creating or editing an action."""

    def __init__(
        self,
        parent: QWidget,
        initial: Optional[Dict[str, Any]] = None,
        scenario_mode: bool = False,
        snippet_names: Optional[List[str]] = None,
        snippet_positions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Action")
        self.resize(420, 320)

        self.action_data: Dict[str, Any] = dict(initial or {"type": "tap"})
        if snippet_names is None:
            self._snippet_names: List[str] = []
        else:
            self._snippet_names = snippet_names
        if self._snippet_names:
            unique_sorted = sorted(set(self._snippet_names), key=str.lower)
            self._snippet_names[:] = unique_sorted
        self._snippet_positions = dict(snippet_positions or {})
        self._snippet_swipes = {
            name: (
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
            )
            for name, (start, end) in (snippet_swipes or {}).items()
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == 2 and len(end) == 2
        }

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._type_combo = QComboBox()
        types = ["tap", "swipe"]
        if not scenario_mode:
            types.append("scenario")
        self._type_combo.addItems(types)
        self._type_combo.setCurrentText(self.action_data.get("type", "tap"))
        self._type_combo.currentTextChanged.connect(self._update_visibility)
        form.addRow("Action type:", self._type_combo)

        self._target_combo = QComboBox()
        self._target_combo.setEditable(True)
        self._populate_snippet_combo()
        target_line_edit = self._target_combo.lineEdit()
        if target_line_edit is not None:
            target_line_edit.setPlaceholderText("Snippet name (optional)")
        target_text = str(self.action_data.get("target_snippet", ""))
        if target_text and self._target_combo.findText(target_text, Qt.MatchFlag.MatchExactly) == -1:
            self._target_combo.addItem(target_text)
        self._target_combo.setCurrentText(target_text)
        self._target_combo.currentTextChanged.connect(self._on_target_changed)
        form.addRow("Target snippet:", self._target_combo)

        self._coords_edit = QLineEdit(self._format_coords(self.action_data.get("coordinates")))
        form.addRow("Coordinates:", self._coords_edit)

        self._end_edit = QLineEdit(self._format_coords(self.action_data.get("end")))
        form.addRow("End coordinates:", self._end_edit)

        self._swipe_style_combo = QComboBox()
        self._swipe_style_combo.addItems(SWIPE_STYLE_OPTIONS)
        style_value = str(self.action_data.get("swipe_style") or SWIPE_STYLE_AUTO)
        if style_value not in SWIPE_STYLE_OPTIONS:
            style_value = SWIPE_STYLE_AUTO
        self._swipe_style_combo.setCurrentText(style_value)
        self._swipe_style_combo.currentTextChanged.connect(self._on_swipe_style_changed)
        form.addRow("Swipe style:", self._swipe_style_combo)

        self._duration_spin = QSpinBox()
        self._duration_spin.setRange(50, 5000)
        self._duration_spin.setSingleStep(50)
        self._duration_spin.setSuffix(" ms")
        self._duration_spin.setValue(int(self.action_data.get("duration_ms", 300)))
        form.addRow("Duration:", self._duration_spin)

        self._scenario_name_edit = QLineEdit(str(self.action_data.get("name", "")))
        form.addRow("Scenario name:", self._scenario_name_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_visibility(self._type_combo.currentText())
        self._apply_default_coordinates()

    def _update_visibility(self, action_type: str) -> None:
        is_tap = action_type == "tap"
        is_swipe = action_type == "swipe"
        is_scenario = action_type == "scenario"

        self._target_combo.parentWidget().setVisible(is_tap or is_swipe)
        self._coords_edit.parentWidget().setVisible(is_tap or is_swipe)
        self._end_edit.parentWidget().setVisible(is_swipe)
        self._swipe_style_combo.parentWidget().setVisible(is_swipe)
        self._duration_spin.parentWidget().setVisible(is_swipe)
        self._scenario_name_edit.parentWidget().setVisible(is_scenario)
        self._apply_default_coordinates()

    def _on_accept(self) -> None:
        action_type = self._type_combo.currentText()
        data: Dict[str, Any] = {"type": action_type}

        target = self._target_combo.currentText().strip()
        coords = self._parse_coords(self._coords_edit.text())
        end = self._parse_coords(self._end_edit.text())
        scenario = self._scenario_name_edit.text().strip()

        if action_type in {"tap", "swipe"}:
            if target:
                data["target_snippet"] = target
                if target not in self._snippet_names:
                    self._snippet_names.append(target)
                    self._snippet_names.sort(key=str.lower)
                    self._populate_snippet_combo()
                    self._target_combo.setCurrentText(target)
                rect = self._snippet_positions.get(target)
                style = self._current_swipe_style()
                start_point = end_point = None
                if style != SWIPE_STYLE_AUTO and rect:
                    points = swipe_points_from_rect(rect, style)
                    if points:
                        start_point, end_point = points
                if start_point is None:
                    template = self._snippet_swipes.get(target)
                    if template:
                        start_point, end_point = template
                if start_point is None and rect:
                    points = swipe_points_from_rect(rect, "Left → Right")
                    if points:
                        start_point, end_point = points
                    else:
                        start_point = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
                        offset_dx = max(4, rect[2] // 6) if rect[2] > 0 else 10
                        offset_dy = max(4, rect[3] // 6) if rect[3] > 0 else 10
                        end_point = (
                            start_point[0] + offset_dx,
                            start_point[1] - offset_dy if rect[3] >= rect[2] else start_point[1] + offset_dy,
                        )
                if start_point and not coords:
                    coords = [int(start_point[0]), int(start_point[1])]
                if action_type == "swipe" and end_point and not end:
                    end = [int(end_point[0]), int(end_point[1])]
            if coords:
                data["coordinates"] = coords
            if action_type == "swipe":
                if end:
                    data["end"] = end
                data["duration_ms"] = int(self._duration_spin.value())
                data["swipe_style"] = self._current_swipe_style()
                if target and coords and end:
                    self._snippet_swipes[target] = (tuple(coords), tuple(end))
                    self._latest_swipe_template = (
                        target,
                        (coords[0], coords[1]),
                        (end[0], end[1]),
                    )
                else:
                    self._latest_swipe_template = None
        elif action_type == "scenario":
            if not scenario:
                QMessageBox.warning(self, "Action", "Provide a scenario name.")
                return
            data["name"] = scenario
            self._latest_swipe_template = None

        self.action_data = data
        self.accept()

    def _on_target_changed(self, _text: str) -> None:
        self._apply_default_coordinates()

    def _current_swipe_style(self) -> str:
        return self._swipe_style_combo.currentText()

    def _on_swipe_style_changed(self, _text: str) -> None:
        self._apply_default_coordinates()

    def _format_coords(self, value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"{value[0]},{value[1]}"
        return str(value)

    def _parse_coords(self, text: str) -> Optional[List[int]]:
        text = text.strip()
        if not text:
            return None
        if "," not in text:
            QMessageBox.warning(self, "Coordinates", "Use comma-separated coordinates, e.g. 540,960.")
            return None
        try:
            parts = [int(float(part.strip())) for part in text.split(",", 1)]
        except ValueError:
            QMessageBox.warning(self, "Coordinates", "Invalid coordinate value.")
            return None
        return parts

    def _populate_snippet_combo(self) -> None:
        self._target_combo.blockSignals(True)
        try:
            self._target_combo.clear()
            self._target_combo.addItem("")
            for name in self._snippet_names:
                self._target_combo.addItem(name)
        finally:
            self._target_combo.blockSignals(False)
        self._apply_default_coordinates()

    def _apply_default_coordinates(self) -> None:
        target = self._target_combo.currentText().strip()
        if not target:
            return
        style = self._current_swipe_style()
        if style == SWIPE_STYLE_CUSTOM:
            return
        rect = self._snippet_positions.get(target)
        start_point = end_point = None
        if style != SWIPE_STYLE_AUTO and rect:
            points = swipe_points_from_rect(rect, style)
            if points:
                start_point, end_point = points
        if start_point is None:
            swipe_template = self._snippet_swipes.get(target)
            if swipe_template and swipe_template[0] != swipe_template[1]:
                start_point, end_point = swipe_template
        if start_point is None and rect:
            points = swipe_points_from_rect(rect, "Left → Right")
            if points:
                start_point, end_point = points
            else:
                centre = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
                offset_dx = max(4, rect[2] // 6) if rect[2] > 0 else 10
                offset_dy = max(4, rect[3] // 6) if rect[3] > 0 else 10
                start_point = centre
                end_point = (
                    centre[0] + offset_dx,
                    centre[1] - offset_dy if rect[3] >= rect[2] else centre[1] + offset_dy,
                )
        if start_point is None:
            return
        if rect:
            start_point = clamp_point_to_rect(start_point, rect)
            if end_point is not None:
                end_point = clamp_point_to_rect(end_point, rect)
        self._coords_edit.setText(f"{start_point[0]},{start_point[1]}")
        if self._type_combo.currentText() == "swipe" and end_point is not None:
            self._end_edit.setText(f"{end_point[0]},{end_point[1]}")


