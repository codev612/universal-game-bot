from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from loguru import logger

from core.device_manager import DeviceManager
from core.game_registry import GameConfig, GameRegistry
from core.layout_registry import LayoutRegistry
from gui.add_device_dialog import AddDeviceDialog
from gui.add_game_dialog import AddGameDialog
from gui.device_table_model import DeviceTableModel
from gui.layout_designer_tab import LayoutDesignerTab
from gui.train_tab import TrainTab


class MainWindow(QMainWindow):
    """Top-level window for the Universal Game Bot device manager."""

    def __init__(
        self,
        device_manager: DeviceManager,
        game_registry: GameRegistry,
        layout_registry: LayoutRegistry,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Universal Game Bot • Device Manager")
        self.resize(960, 600)

        self._device_manager = device_manager
        self._table_model = DeviceTableModel()
        self._game_registry = game_registry
        self._layout_registry = layout_registry
        self._pending_checks: Dict[Tuple[str, str], str] = {}
        self._auto_check_on_startup = True
        self._suppress_selection_checks = False
        self._game_device_status: Dict[str, Dict[str, Tuple[bool, bool]]] = {}

        self._central_widget = QWidget()
        self.setCentralWidget(self._central_widget)

        root_layout = QVBoxLayout(self._central_widget)

        self._tabs = QTabWidget()
        root_layout.addWidget(self._tabs)

        self._devices_tab = QWidget()
        self._devices_layout = QVBoxLayout(self._devices_tab)
        self._devices_layout.addLayout(self._build_header())
        self._devices_layout.addLayout(self._build_game_controls())
        self._devices_layout.addWidget(self._build_table())
        self._devices_layout.addWidget(self._build_log_panel())

        self._tabs.addTab(self._devices_tab, "Devices")

        self._train_tab = TrainTab(
            device_manager=self._device_manager,
            layout_registry=self._layout_registry,
        )
        self._train_tab.bind_signals()
        self._tabs.addTab(self._train_tab, "Training")

        self._layout_designer_tab = LayoutDesignerTab(
            device_manager=self._device_manager,
            layout_registry=self._layout_registry,
            train_tab=self._train_tab,
        )
        self._layout_designer_tab.bind_signals()
        self._tabs.insertTab(1, self._layout_designer_tab, "Control Layout")

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._wire_signals()

        self._update_game_combo(self._game_registry.games)
        self._update_layout_designer_device()

        self._device_manager.refresh_devices()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._train_tab.shutdown()
        self._device_manager.shutdown()
        super().closeEvent(event)

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        title_label = QLabel("Connected Devices")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(title_label)
        layout.addStretch(1)

        self._add_device_button = QPushButton("Add Device")
        self._add_device_button.clicked.connect(self._on_add_device_clicked)
        layout.addWidget(self._add_device_button)

        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._on_refresh_clicked)
        layout.addWidget(self._refresh_button)

        self._open_shell_button = QPushButton("Open Shell")
        self._open_shell_button.clicked.connect(self._on_open_shell_clicked)
        self._open_shell_button.setEnabled(False)
        layout.addWidget(self._open_shell_button)

        return layout

    def _build_game_controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        game_label = QLabel("Game")
        game_label.setStyleSheet("font-weight: 500;")

        self._game_combo = QComboBox()
        self._game_combo.setEditable(False)
        self._game_combo.currentIndexChanged.connect(self._on_game_selection_changed)

        self._add_game_button = QPushButton("Add Game")
        self._add_game_button.clicked.connect(self._on_add_game_clicked)

        self._remove_game_button = QPushButton("Delete Game")
        self._remove_game_button.clicked.connect(self._on_remove_game_clicked)
        self._remove_game_button.setEnabled(False)

        self._check_game_button = QPushButton("Check Game")
        self._check_game_button.clicked.connect(self._on_check_game_clicked)
        self._check_game_button.setEnabled(False)

        layout.addWidget(game_label)
        layout.addWidget(self._game_combo, stretch=1)
        layout.addWidget(self._add_game_button)
        layout.addWidget(self._remove_game_button)
        layout.addWidget(self._check_game_button)

        return layout

    def _build_table(self) -> QTableView:
        self._table_view = QTableView()
        self._table_view.setModel(self._table_model)
        self._table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table_view.horizontalHeader().setStretchLastSection(True)
        self._table_view.verticalHeader().setVisible(False)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setSortingEnabled(True)

        self._table_view.selectionModel().selectionChanged.connect(self._on_selection_changed)

        return self._table_view

    def _build_log_panel(self) -> QPlainTextEdit:
        self._log_panel = QPlainTextEdit()
        self._log_panel.setReadOnly(True)
        self._log_panel.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._log_panel.setPlaceholderText("Status messages will appear here.")
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setVerticalStretch(1)
        self._log_panel.setSizePolicy(size_policy)
        return self._log_panel

    def _wire_signals(self) -> None:
        self._device_manager.devices_updated.connect(self._on_devices_updated)
        self._device_manager.error_occurred.connect(self._on_error)
        self._device_manager.device_connected.connect(self._on_device_connected)
        self._device_manager.app_status_ready.connect(self._on_app_status_ready)
        self._game_registry.games_changed.connect(self._update_game_combo)

    def _on_add_device_clicked(self) -> None:
        result = AddDeviceDialog.prompt(self)
        if not result:
            return
        alias, host, port = result
        self._append_log(f"Connecting to {alias} ({host}:{port})...")
        self._status_bar.showMessage(f"Connecting to {alias}...", 3000)
        self._device_manager.connect_device(alias=alias, host=host, port=port)

    def _on_refresh_clicked(self) -> None:
        self._append_log("Refreshing devices...")
        self._status_bar.showMessage("Refreshing devices...", 2000)
        self._device_manager.refresh_devices()

    def _on_open_shell_clicked(self) -> None:
        device = self._selected_device()
        if not device:
            return

        self._append_log(f"Opening shell for {device.serial}")
        try:
            launch_adb_shell(serial=device.serial)
        except RuntimeError as exc:
            logger.exception("Failed to open shell")
            QMessageBox.critical(self, "Shell Error", str(exc))
        else:
            self._status_bar.showMessage(f"Shell launched for {device.serial}", 3000)

    def _on_selection_changed(self, *_args) -> None:
        has_selection = self._table_view.selectionModel().hasSelection()
        self._open_shell_button.setEnabled(has_selection)
        self._update_check_button_state()
        self._update_layout_designer_device()
        self._update_train_tab_device()

    def _on_devices_updated(self, devices) -> None:
        self._table_model.update_devices(devices)
        self._append_log(f"Found {len(devices)} device(s).")
        self._status_bar.showMessage("Devices refreshed", 2000)
        self._update_check_button_state()
        self._update_layout_designer_device()
        if self._auto_check_on_startup and self._selected_game() and devices:
            self._check_selected_game()

    def _on_error(self, message: str) -> None:
        self._append_log(f"Error: {message}")
        QMessageBox.warning(self, "ADB Error", message)
        status_text = "Failed to refresh devices" if "refresh" in message.lower() else "Operation failed"
        self._status_bar.showMessage(status_text, 3000)

    def _on_device_connected(self, serial: str, alias: str) -> None:
        label = alias or serial
        self._append_log(f"Connected to {label} ({serial})")
        self._status_bar.showMessage(f"Connected to {label}", 2000)

    def _on_game_selection_changed(self, _index: int) -> None:
        self._update_check_button_state()
        self._remove_game_button.setEnabled(self._selected_game() is not None)
        game = self._selected_game()
        self._layout_designer_tab.set_active_game(game.name if game else None)
        self._train_tab.set_active_game(game.name if game else None)
        self._update_layout_designer_device()
        if (
            not self._suppress_selection_checks
            and self._selected_game()
            and self._device_manager.devices
        ):
            self._check_selected_game()

    def _on_add_game_clicked(self) -> None:
        result = AddGameDialog.prompt(self)
        if not result:
            return
        name, package = result
        self._game_registry.add_game(name=name, package=package)
        self._append_log(f"Added game '{name}' ({package})")
        self._status_bar.showMessage(f"Game '{name}' added", 2000)

    def _on_remove_game_clicked(self) -> None:
        game = self._selected_game()
        if not game:
            return
        removed = self._game_registry.remove_game(game.name)
        if removed:
            self._append_log(f"Removed game '{game.name}'")
            self._status_bar.showMessage(f"Game '{game.name}' removed", 2000)
        else:
            self._append_log(f"Failed to remove game '{game.name}'")
            QMessageBox.warning(self, "Remove Game", f"Could not remove '{game.name}'.")

    def _on_check_game_clicked(self) -> None:
        self._check_selected_game()

    def _on_app_status_ready(self, serial: str, package: str, installed: bool, running: bool) -> None:
        game_name = self._pending_checks.pop((serial, package), package)

        device_info = next((d for d in self._device_manager.devices if d.serial == serial), None)
        device_label = device_info.alias or device_info.display_name if device_info else serial

        self._table_model.update_game_status(serial, installed, running)
        self._game_device_status.setdefault(game_name, {})[serial] = (installed, running)

        if not installed:
            message = f"'{game_name}' is not installed on {device_label}."
            status = "not installed"
        elif running:
            message = f"'{game_name}' is running on {device_label}."
            status = "running"
        else:
            message = f"'{game_name}' is installed on {device_label} but not running."
            status = "installed"

        self._append_log(message)
        self._status_bar.showMessage(f"{game_name}: {status}", 3000)
        current_game = self._selected_game()
        if current_game and current_game.name == game_name:
            self._update_layout_designer_device()

    def _selected_device(self) -> Optional[object]:
        selection_model = self._table_view.selectionModel()
        if not selection_model or not selection_model.hasSelection():
            return None
        index: QModelIndex = selection_model.selectedRows()[0]
        return self._table_model.device_at(index.row())

    def _selected_game(self) -> Optional[GameConfig]:
        index = self._game_combo.currentIndex()
        if index < 0:
            return None
        return self._game_combo.itemData(index)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_panel.appendPlainText(f"[{timestamp}] {message}")

    def _update_game_combo(self, games) -> None:
        current_name = self._game_combo.currentText()
        self._game_combo.blockSignals(True)
        self._game_combo.clear()
        for game in games:
            self._game_combo.addItem(game.name, game)
        if current_name:
            idx = self._game_combo.findText(current_name, Qt.MatchFlag.MatchExactly)
            if idx >= 0:
                self._game_combo.setCurrentIndex(idx)
        if self._game_combo.count() > 0 and self._game_combo.currentIndex() < 0:
            self._suppress_selection_checks = True
            self._game_combo.setCurrentIndex(0)
            self._suppress_selection_checks = False
        self._game_combo.blockSignals(False)
        self._update_check_button_state()
        self._remove_game_button.setEnabled(self._selected_game() is not None)
        game = self._selected_game()
        self._layout_designer_tab.set_active_game(game.name if game else None)
        self._train_tab.set_active_game(game.name if game else None)

    def _update_check_button_state(self) -> None:
        has_device = bool(self._device_manager.devices)
        has_game = bool(self._selected_game())
        self._check_game_button.setEnabled(has_device and has_game)

    def _target_devices(self):
        selected = self._selected_device()
        if selected:
            return [selected]
        return self._device_manager.devices

    def _update_layout_designer_device(self) -> None:
        current_game = self._selected_game()
        target_serial: Optional[str] = None
        target_label: Optional[str] = None
        fallback_device = self._selected_device()

        if current_game:
            status_map = self._game_device_status.get(current_game.name, {})
            devices_by_serial = {device.serial: device for device in self._device_manager.devices}

            for serial, (installed, running) in status_map.items():
                if running and serial in devices_by_serial:
                    device = devices_by_serial[serial]
                    target_serial = serial
                    target_label = device.alias or device.display_name
                    break

            if target_serial is None:
                for serial, (installed, running) in status_map.items():
                    if installed and serial in devices_by_serial:
                        device = devices_by_serial[serial]
                        target_serial = serial
                        target_label = device.alias or device.display_name
                        break

        if target_serial is None and fallback_device:
            target_serial = fallback_device.serial
            target_label = fallback_device.alias or fallback_device.display_name

        if target_serial is None and self._device_manager.devices:
            device = self._device_manager.devices[0]
            target_serial = device.serial
            target_label = device.alias or device.display_name

        self._layout_designer_tab.set_active_device(target_serial, target_label)
        self._train_tab.set_active_device(target_serial, target_label)

    def _check_selected_game(self) -> None:
        game = self._selected_game()
        devices = self._target_devices()

        if not game or not devices:
            return

        self._auto_check_on_startup = False

        for device in devices:
            label = device.alias or device.display_name
            self._append_log(f"Checking '{game.name}' status on {label}...")
            self._status_bar.showMessage(f"Checking {game.name}...", 2000)
            self._pending_checks[(device.serial, game.package)] = game.name
            self._device_manager.check_game_running(serial=device.serial, package=game.package)


def launch_adb_shell(serial: str) -> None:
    """
    Spawn a system terminal window running an interactive adb shell session.

    On Windows, launches a new PowerShell window; on macOS/Linux, uses the default
    terminal via `x-terminal-emulator`/`open`.
    """
    if sys.platform.startswith("win"):
        command = [
            "powershell",
            "-NoExit",
            "-Command",
            f'adb -s "{serial}" shell',
        ]
        creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
        try:
            subprocess.Popen(command, creationflags=creationflags)
        except FileNotFoundError as exc:
            raise RuntimeError("PowerShell or adb not found on PATH.") from exc
    elif sys.platform == "darwin":
        osa_script = f'''
            tell application "Terminal"
                activate
                do script "adb -s {serial} shell"
            end tell
        '''.strip()
        try:
            subprocess.Popen(["osascript", "-e", osa_script])
        except FileNotFoundError as exc:
            raise RuntimeError("osascript not available to launch Terminal.") from exc
    else:
        terminal = _pick_unix_terminal()
        try:
            subprocess.Popen([terminal, "-e", "adb", "-s", serial, "shell"])
        except FileNotFoundError as exc:
            raise RuntimeError(f"{terminal} not available to launch shell.") from exc


def _pick_unix_terminal() -> str:
    for candidate in ("x-terminal-emulator", "gnome-terminal", "konsole", "xfce4-terminal"):
        if shutil.which(candidate):
            return candidate
    return "x-terminal-emulator"

