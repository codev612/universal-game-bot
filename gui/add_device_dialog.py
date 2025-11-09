from __future__ import annotations

from typing import Optional, Tuple

from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QIntValidator, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
)


class AddDeviceDialog(QDialog):
    """Collects alias, host, and port for a manual ADB connection."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Device")

        self._alias_input = QLineEdit()
        self._alias_input.setPlaceholderText("Example: BlueStacks Instance 3")

        self._host_input = QLineEdit()
        self._host_input.setPlaceholderText("127.0.0.1")
        ip_regex = QRegularExpression(
            r"^([0-9]{1,3}\.){3}[0-9]{1,3}$|^([A-Za-z0-9\-\._]+)$"
        )
        self._host_input.setValidator(QRegularExpressionValidator(ip_regex))

        self._port_input = QLineEdit()
        self._port_input.setPlaceholderText("5555")
        self._port_input.setValidator(QIntValidator(1, 65535))

        form_layout = QFormLayout()
        form_layout.addRow("Name", self._alias_input)
        form_layout.addRow("IP / Host", self._host_input)
        form_layout.addRow("Port", self._port_input)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form_layout)
        layout.addWidget(self._buttons)

    def _on_accept(self) -> None:
        if not self._alias_input.text().strip():
            self._alias_input.setFocus()
            return
        if not self._host_input.text().strip():
            self._host_input.setFocus()
            return
        if not self._port_input.text().strip():
            self._port_input.setFocus()
            return
        self.accept()

    def get_data(self) -> Tuple[str, str, int]:
        alias = self._alias_input.text().strip()
        host = self._host_input.text().strip()
        port = int(self._port_input.text())
        return alias, host, port

    @staticmethod
    def prompt(parent=None) -> Optional[Tuple[str, str, int]]:
        dialog = AddDeviceDialog(parent)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return dialog.get_data()
        return None

