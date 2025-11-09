from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant

from core.adb_client import DeviceInfo


class DeviceTableModel(QAbstractTableModel):
    headers = ("Name", "Serial", "State", "Type", "Game Status")

    def __init__(self, devices: Optional[List[DeviceInfo]] = None, parent=None) -> None:
        super().__init__(parent)
        self._devices: List[DeviceInfo] = devices or []
        self._game_status: Dict[str, Tuple[Optional[bool], Optional[bool]]] = {}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid():
            return 0
        return len(self._devices)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid():
            return 0
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return QVariant()

        device = self._devices[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            column_getters = [
                lambda d: d.alias or d.display_name,
                lambda d: d.serial,
                lambda d: d.state.capitalize() if d.state else "Unknown",
                lambda d: "Emulator" if d.is_emulator else "Physical",
                lambda d: self._format_status(self._game_status.get(d.serial)),
            ]
            return column_getters[index.column()](device)

        if role == Qt.ItemDataRole.ToolTipRole:
            alias_line = f"Alias: {device.alias}\n" if device.alias else ""
            status_line = ""
            if index.column() == 4:
                status_line = (
                    f"Installed: {self._status_value(self._game_status.get(device.serial), 0)}\n"
                    f"Running: {self._status_value(self._game_status.get(device.serial), 1)}\n"
                )

            return (
                f"{alias_line}"
                f"{status_line}"
                f"Name: {device.display_name}\n"
                f"Serial: {device.serial}\n"
                f"State: {device.state}"
            )

        return QVariant()

    def headerData(  # type: ignore[override]
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return section + 1

    def device_at(self, row: int) -> Optional[DeviceInfo]:
        if 0 <= row < len(self._devices):
            return self._devices[row]
        return None

    def update_devices(self, devices: List[DeviceInfo]) -> None:
        self.beginResetModel()
        self._devices = list(devices)
        self.endResetModel()

    def update_game_status(self, serial: str, installed: Optional[bool], running: Optional[bool]) -> None:
        self._game_status[serial] = (installed, running)
        for row, device in enumerate(self._devices):
            if device.serial == serial:
                top_left = self.index(row, 4)
                self.dataChanged.emit(top_left, top_left, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole])
                break

    @staticmethod
    def _format_status(status: Optional[Tuple[Optional[bool], Optional[bool]]]) -> str:
        if not status:
            return "—"
        installed, running = status
        if installed is None:
            return "Unknown"
        if not installed:
            return "Not Installed"
        if running:
            return "Installed & Running"
        return "Installed"

    @staticmethod
    def _status_value(
        status: Optional[Tuple[Optional[bool], Optional[bool]]],
        index: int,
    ) -> str:
        if not status:
            return "Unknown"
        value = status[index]
        if value is None:
            return "Unknown"
        return "Yes" if value else "No"

