from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.scenario_registry import ScenarioRegistry


class ScenarioTab(QWidget):
    """Tab for managing available scenarios."""

    def __init__(
        self,
        scenario_registry: ScenarioRegistry,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._scenario_registry = scenario_registry

        self._build_ui()
        self._wire_signals()
        self._refresh_list()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("Manage reusable scenario labels.")
        header.setWordWrap(True)
        layout.addWidget(header)

        self._list_widget = QListWidget()
        self._list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self._list_widget, stretch=1)

        add_row = QHBoxLayout()
        self._add_input = QLineEdit()
        self._add_input.setPlaceholderText("New scenario name")
        self._add_button = QPushButton("Add")
        add_row.addWidget(self._add_input, stretch=1)
        add_row.addWidget(self._add_button)
        layout.addLayout(add_row)

        action_row = QHBoxLayout()
        self._remove_button = QPushButton("Remove Selected")
        self._remove_button.setEnabled(False)
        self._reset_button = QPushButton("Reset to Defaults")
        action_row.addWidget(self._remove_button)
        action_row.addWidget(self._reset_button)
        layout.addLayout(action_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #666666;")
        layout.addWidget(self._status_label)

    def _wire_signals(self) -> None:
        self._add_button.clicked.connect(self._on_add_clicked)
        self._remove_button.clicked.connect(self._on_remove_clicked)
        self._reset_button.clicked.connect(self._on_reset_clicked)
        self._list_widget.currentItemChanged.connect(lambda *_: self._update_button_states())
        self._scenario_registry.scenarios_changed.connect(self._on_registry_changed)

    def _refresh_list(self) -> None:
        self._list_widget.clear()
        for scenario in self._scenario_registry.scenarios():
            self._list_widget.addItem(QListWidgetItem(scenario))
        self._add_input.clear()
        self._update_button_states()
        self._status_label.setText(f"{self._list_widget.count()} scenario(s) available.")

    def _on_add_clicked(self) -> None:
        name = self._add_input.text().strip()
        if not name:
            QMessageBox.information(self, "Add Scenario", "Enter a scenario name to add.")
            return

        if not self._scenario_registry.add_scenario(name):
            QMessageBox.information(
                self,
                "Add Scenario",
                f"Scenario '{name}' already exists.",
            )
            return

        self._add_input.clear()
        self._status_label.setText(f"Scenario '{name}' added.")

    def _on_remove_clicked(self) -> None:
        item = self._list_widget.currentItem()
        if not item:
            return
        name = item.text()
        confirm = QMessageBox.question(
            self,
            "Remove Scenario",
            f"Remove scenario '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        if not self._scenario_registry.remove_scenario(name):
            QMessageBox.information(
                self,
                "Remove Scenario",
                f"Scenario '{name}' could not be removed.",
            )
            return

        self._status_label.setText(f"Scenario '{name}' removed.")

    def _on_reset_clicked(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Reset Scenarios",
            "Reset the scenario list to the default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._scenario_registry.reset_to_defaults()
        self._status_label.setText("Scenario list reset to defaults.")

    def _on_registry_changed(self, _: list) -> None:
        self._refresh_list()

    def _update_button_states(self) -> None:
        self._remove_button.setEnabled(self._list_widget.currentItem() is not None)

