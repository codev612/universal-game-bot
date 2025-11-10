from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from core.player_state_registry import PlayerStateRegistry, DEFAULT_PLAYER_STATES


class PlayerStateDialog(QDialog):
    def __init__(self, registry: PlayerStateRegistry, parent: Optional[QDialog] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Player States")
        self._registry = registry

        self._list = QListWidget()
        self._refresh_list()

        self._input = QLineEdit()
        self._input.setPlaceholderText("New player state")
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._on_add_clicked)

        add_row = QHBoxLayout()
        add_row.addWidget(self._input, stretch=1)
        add_row.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._on_remove_clicked)

        reset_button = QPushButton("Reset Defaults")
        reset_button.clicked.connect(self._on_reset_clicked)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Player states:"))
        layout.addWidget(self._list)
        layout.addLayout(add_row)
        layout.addWidget(remove_button)
        layout.addWidget(reset_button)
        layout.addWidget(button_box)

        self._registry.states_changed.connect(self._on_states_changed)

    def _refresh_list(self) -> None:
        self._list.clear()
        for state in self._registry.player_states():
            self._list.addItem(state)

    def _on_states_changed(self, states: list[str]) -> None:
        self._refresh_list()

    def _on_add_clicked(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._registry.add_state(text)
        self._input.clear()

    def _on_remove_clicked(self) -> None:
        selected = self._list.currentItem()
        if not selected:
            return
        state = selected.text()
        if state in DEFAULT_PLAYER_STATES:
            return
        self._registry.remove_state(state)

    def _on_reset_clicked(self) -> None:
        self._registry.reset_states()

