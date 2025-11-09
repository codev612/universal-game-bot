from __future__ import annotations

from typing import Optional, Tuple

from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
)


class AddGameDialog(QDialog):
    """Collects game name and package identifier."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Game")

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Example: Clash of Clans")

        self._package_input = QLineEdit()
        self._package_input.setPlaceholderText("Example: com.supercell.clashofclans")
        package_regex = QRegularExpression(r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)+$")
        self._package_input.setValidator(QRegularExpressionValidator(package_regex))

        form_layout = QFormLayout()
        form_layout.addRow("Game Name", self._name_input)
        form_layout.addRow("Package Name", self._package_input)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form_layout)
        layout.addWidget(self._buttons)

    def _on_accept(self) -> None:
        if not self._name_input.text().strip():
            self._name_input.setFocus()
            return
        if not self._package_input.text().strip():
            self._package_input.setFocus()
            return
        self.accept()

    def get_data(self) -> Tuple[str, str]:
        name = self._name_input.text().strip()
        package = self._package_input.text().strip()
        return name, package

    @staticmethod
    def prompt(parent=None) -> Optional[Tuple[str, str]]:
        dialog = AddGameDialog(parent)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return dialog.get_data()
        return None

