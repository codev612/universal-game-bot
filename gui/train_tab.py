from __future__ import annotations

import datetime as dt
import io
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from loguru import logger

try:
    import easyocr
except Exception as exc:  # noqa: BLE001
    easyocr = None  # type: ignore[assignment]
    logger.warning("EasyOCR not available: {}", exc)

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except Exception as exc:  # noqa: BLE001
    Image = None  # type: ignore[assignment]
    ImageEnhance = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    logger.warning("Pillow not available: {}", exc)

from core.device_manager import DeviceManager
from core.layout_registry import LayoutRegistry
from core.scenario_registry import ScenarioRegistry


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)
    dragged = pyqtSignal(int, int, int, int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._press_position: Optional[tuple[float, float]] = None
        self._drag_threshold = 6.0
        self._is_dragging = False

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._press_position = (pos.x(), pos.y())
            self._is_dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._press_position and event.buttons() & Qt.MouseButton.LeftButton:
            pos = event.position()
            dx = pos.x() - self._press_position[0]
            dy = pos.y() - self._press_position[1]
            if (dx * dx + dy * dy) ** 0.5 >= self._drag_threshold:
                self._is_dragging = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._press_position:
            pos = event.position()
            start_x, start_y = self._press_position
            end_x, end_y = pos.x(), pos.y()
            if self._is_dragging:
                self.dragged.emit(int(start_x), int(start_y), int(end_x), int(end_y))
            else:
                self.clicked.emit(int(end_x), int(end_y))
        self._press_position = None
        self._is_dragging = False
        super().mouseReleaseEvent(event)


class TrainTab(QWidget):
    """
    Training workflow tab: captures labeled screenshots for dataset creation.

    Screenshots are saved under the configured dataset directory, organized by game name.
    """

    scenario_manage_requested = pyqtSignal()
    _GLOBAL_SCENARIO_KEY = "__GLOBAL_SCENARIO__"

    def __init__(
        self,
        device_manager: DeviceManager,
        layout_registry: LayoutRegistry,  # reserved for future use (e.g., auto-annotating regions)
        scenario_registry: ScenarioRegistry,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._device_manager = device_manager
        self._layout_registry = layout_registry
        self._scenario_registry = scenario_registry

        self._current_serial: Optional[str] = None
        self._current_device_label = "No device selected"
        self._current_game: Optional[str] = None
        self._current_scenario: Optional[str] = None
        self._next_scenario: Optional[str] = None
        self._scenario_history: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._available_scenarios: List[str] = []

        self._pending_capture: bool = False
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._trigger_live_capture)
        self._last_raw_pixmap: Optional[QPixmap] = None
        self._last_scaled_size: Optional[tuple[int, int]] = None
        self._last_screenshot_bytes: Optional[bytes] = None
        self._awaiting_tap: bool = False
        self._awaiting_swipe: bool = False
        self._ocr_reader: Optional["easyocr.Reader"] = None
        self._ocr_enabled: bool = True
        self._state_board_inputs: Dict[str, QLineEdit] = {}
        self._state_board_captured: Dict[str, List[str]] = {}

        self._build_ui()
        self._wire_signals()
        self._reload_scenarios()
        self._scenario_registry.scenarios_changed.connect(self._on_scenarios_changed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_group = QGroupBox("Context")
        info_form = QFormLayout(info_group)

        self._device_value = QLabel(self._current_device_label)
        self._game_value = QLabel("No game selected")

        info_form.addRow("Device:", self._device_value)
        info_form.addRow("Game:", self._game_value)

        scenario_row_widget = QWidget()
        scenario_row_layout = QHBoxLayout(scenario_row_widget)
        scenario_row_layout.setContentsMargins(0, 0, 0, 0)
        scenario_row_layout.setSpacing(6)

        scenario_labels = QVBoxLayout()
        scenario_labels.setContentsMargins(0, 0, 0, 0)
        scenario_labels.setSpacing(2)

        current_label = QLabel("Current:")
        next_label = QLabel("Next:")
        current_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        next_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        scenario_labels.addWidget(current_label)
        scenario_labels.addWidget(next_label)

        scenario_combo_layout = QVBoxLayout()
        scenario_combo_layout.setContentsMargins(0, 0, 0, 0)
        scenario_combo_layout.setSpacing(2)

        self._scenario_combo_current = QComboBox()
        self._scenario_combo_current.addItem("No Scenario", None)
        self._scenario_combo_current.setEnabled(False)

        self._scenario_combo_next = QComboBox()
        self._scenario_combo_next.addItem("No Scenario", None)
        self._scenario_combo_next.setEnabled(False)

        scenario_combo_layout.addWidget(self._scenario_combo_current)
        scenario_combo_layout.addWidget(self._scenario_combo_next)

        self._manage_scenarios_button = QPushButton("Manage…")
        self._manage_scenarios_button.setEnabled(True)

        scenario_row_layout.addLayout(scenario_labels)
        scenario_row_layout.addLayout(scenario_combo_layout, stretch=1)
        scenario_row_layout.addWidget(self._manage_scenarios_button)

        info_form.addRow("Scenario:", scenario_row_widget)

        dataset_group = QGroupBox("Dataset")
        dataset_layout = QHBoxLayout(dataset_group)

        self._dataset_path_edit = QLineEdit(str(Path("data") / "training").replace("\\", "/"))
        self._dataset_path_edit.setPlaceholderText("Directory to store captured screenshots")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._on_browse_dataset)

        dataset_layout.addWidget(self._dataset_path_edit, stretch=1)
        dataset_layout.addWidget(browse_button)

        action_row = QHBoxLayout()
        self._capture_button = QPushButton("Capture Screenshot")
        self._capture_button.clicked.connect(self._on_capture_clicked)
        self._capture_button.setEnabled(False)

        self._live_toggle = QPushButton("Start Live Capture")
        self._live_toggle.setCheckable(True)
        self._live_toggle.setEnabled(False)
        self._live_toggle.toggled.connect(self._on_live_toggled)

        self._interval_edit = QLineEdit("2.0")
        self._interval_edit.setFixedWidth(60)
        self._interval_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._interval_edit.setToolTip("Capture interval in seconds (>=0.2 recommended)")

        interval_box = QHBoxLayout()
        interval_box.addWidget(QLabel("Interval (s):"))
        interval_box.addWidget(self._interval_edit)

        self._status_label = QLabel("Select a game and device, then capture screenshots to build a dataset.")
        self._status_label.setWordWrap(True)

        action_row.addWidget(self._capture_button)
        action_row.addWidget(self._live_toggle)
        action_row.addLayout(interval_box)
        action_row.addStretch(1)

        layout.addWidget(info_group)
        layout.addWidget(dataset_group)
        layout.addLayout(action_row)
        layout.addWidget(self._status_label)

        self._preview_group = QGroupBox("Last Capture Preview")
        preview_layout = QVBoxLayout(self._preview_group)

        self._preview_label = ClickableLabel("No capture yet.")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.clicked.connect(self._on_preview_clicked)
        self._preview_label.dragged.connect(self._on_preview_dragged)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumSize(720, 440)
        scroll_area.setWidget(self._preview_label)

        self._click_position_label = QLabel("Click on the preview to send a tap. Coordinates will appear here.")
        self._click_position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        preview_layout.addWidget(scroll_area)
        preview_layout.addWidget(self._click_position_label)
        layout.addWidget(self._preview_group, stretch=1)

        self._ocr_results_group = QGroupBox("State Board Values")
        ocr_layout = QVBoxLayout(self._ocr_results_group)
        self._state_board_message = QLabel(
            "Configure State Boards in the Layout Designer to see OCR results here."
        )
        self._state_board_message.setWordWrap(True)
        ocr_layout.addWidget(self._state_board_message)

        self._state_board_results_widget = QWidget()
        self._state_board_results_layout = QVBoxLayout(self._state_board_results_widget)
        self._state_board_results_layout.setContentsMargins(0, 0, 0, 0)
        self._state_board_results_layout.setSpacing(8)
        ocr_layout.addWidget(self._state_board_results_widget)
        self._state_board_results_widget.hide()
        layout.addWidget(self._ocr_results_group)

    def _wire_signals(self) -> None:
        self._device_manager.error_occurred.connect(self._on_error)
        self._device_manager.screenshot_captured.connect(self._on_screenshot_captured)
        self._device_manager.tap_sent.connect(self._on_tap_sent)
        self._device_manager.swipe_sent.connect(self._on_swipe_sent)
        self._scenario_combo_current.currentIndexChanged.connect(self._on_current_scenario_changed)
        self._scenario_combo_next.currentIndexChanged.connect(self._on_next_scenario_changed)
        self._manage_scenarios_button.clicked.connect(self._on_manage_scenarios_clicked)

    def bind_signals(self) -> None:
        """
        Compatibility shim with LayoutDesignerTab.

        Signals are already wired in __init__, but MainWindow expects a bind_signals() hook.
        """
        # Nothing additional to bind; method exists for API parity.
        return

    # External context setters -------------------------------------------------

    def set_active_device(self, serial: Optional[str], display_name: Optional[str]) -> None:
        self._current_serial = serial
        self._current_device_label = display_name or "No device selected"
        self._device_value.setText(self._current_device_label)
        self._update_capture_state()

    def set_active_game(self, game_name: Optional[str]) -> None:
        previous_game = self._current_game
        if previous_game and previous_game != game_name:
            self._scenario_history[previous_game] = (self._current_scenario, self._next_scenario)

        self._current_game = game_name
        self._game_value.setText(game_name or "No game selected")
        self._reload_scenarios()
        self._update_capture_state()
        if self._last_screenshot_bytes:
            self._update_state_board_values(self._last_screenshot_bytes)
        else:
            self._set_state_board_message("Capture a screenshot to populate State Board values.")

    def shutdown(self) -> None:
        """Stop timers and pending operations before application exit."""
        if self._live_toggle.isChecked():
            self._stop_live_capture()
        self._live_timer.stop()
        self._pending_capture = False
        self._awaiting_tap = False

    # UI callbacks -------------------------------------------------------------

    def _on_browse_dataset(self) -> None:
        start_dir = Path(self._dataset_path_edit.text() or ".").resolve()
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", str(start_dir))
        if directory:
            self._dataset_path_edit.setText(directory.replace("\\", "/"))

    def _on_capture_clicked(self) -> None:
        if not self._current_serial:
            QMessageBox.information(
                self,
                "Select Device",
                "Please select a device in the Devices tab before capturing a screenshot.",
            )
            return
        if not self._current_game:
            QMessageBox.information(
                self,
                "Select Game",
                "Please select a game in the Devices tab before capturing a screenshot.",
            )
            return

        path = self._dataset_directory()
        if path is None:
            return

        self._pending_capture = True
        self._capture_button.setEnabled(False)
        game_label = self._current_game or "current game"
        self._status_label.setText(
            f"Capturing screenshot for {game_label}{self._scenario_suffix()} on {self._current_device_label}…"
        )
        self._device_manager.capture_screenshot(self._current_serial)

    # Signal handlers ----------------------------------------------------------

    def _on_screenshot_captured(self, serial: str, data: bytes) -> None:
        if not self._pending_capture or serial != self._current_serial:
            return

        self._pending_capture = False
        self._capture_button.setEnabled(True)
        if self._live_toggle.isChecked():
            self._status_label.setText(
                f"Live capture active for {self._current_game}{self._scenario_suffix()}…"
            )

        dataset_dir = self._dataset_directory(create=True)
        if dataset_dir is None:
            return

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        game_name = self._current_game or "unknown_game"
        game_dir = dataset_dir / game_name
        game_dir.mkdir(parents=True, exist_ok=True)
        target_dir = game_dir
        scenario_dir = self._scenario_directory_name()
        if scenario_dir:
            target_dir = game_dir / scenario_dir
            target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{timestamp}{self._scenario_filename_suffix()}.png"

        try:
            file_path.write_bytes(data)
        except OSError as exc:
            logger.error("Failed to write screenshot {}: {}", file_path, exc)
            QMessageBox.critical(self, "Capture Error", f"Failed to save screenshot:\n{exc}")
            return

        self._last_screenshot_bytes = bytes(data)
        pixmap = QPixmap()
        if pixmap.loadFromData(data, "PNG"):
            self._last_raw_pixmap = pixmap
            scaled = pixmap.scaled(
                960,
                540,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._last_scaled_size = (scaled.width(), scaled.height())
            self._preview_label.setPixmap(scaled)
            self._preview_label.resize(scaled.size())
        else:
            self._preview_label.setText("Unable to load preview image.")
            self._last_raw_pixmap = None
            self._last_scaled_size = None

        self._status_label.setText(f"Screenshot saved to {file_path}")
        self._click_position_label.setText("Click on the preview to send a tap. Coordinates will appear here.")
        logger.info("Training screenshot saved to {}", file_path)
        self._update_state_board_values(data)

    def _on_error(self, message: str) -> None:
        if self._pending_capture:
            self._pending_capture = False
            self._capture_button.setEnabled(True)
            if self._live_toggle.isChecked():
                self._stop_live_capture()
            self._status_label.setText(f"Capture failed: {message}")
        elif self._awaiting_tap:
            self._awaiting_tap = False
            self._status_label.setText(message)
        elif self._awaiting_swipe:
            self._awaiting_swipe = False
            self._status_label.setText(message)
        else:
            self._status_label.setText(message)

    # Helpers ------------------------------------------------------------------

    def _dataset_directory(self, create: bool = False) -> Optional[Path]:
        text = self._dataset_path_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Dataset Path", "Please specify a directory to store screenshots.")
            return None

        path = Path(text).expanduser().resolve()
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                QMessageBox.critical(self, "Dataset Path", f"Unable to create dataset directory:\n{exc}")
                return None
        return path

    def _update_capture_state(self) -> None:
        has_device = self._current_serial is not None
        has_game = self._current_game is not None
        can_capture = has_device and has_game and not self._pending_capture
        self._capture_button.setEnabled(can_capture and not self._live_toggle.isChecked())
        self._live_toggle.setEnabled(has_device and has_game)

        if self._live_toggle.isChecked() and not can_capture:
            self._stop_live_capture()

        if not has_game:
            self._status_label.setText("Select a game in the Devices tab to begin data collection.")
        elif not has_device:
            self._status_label.setText(
                f"{self._current_game}{self._scenario_suffix()} selected. "
                "Choose a device in the Devices tab to capture screenshots."
            )
        elif self._live_toggle.isChecked():
            self._status_label.setText(
                f"Live capture active for {self._current_game}{self._scenario_suffix()} "
                f"on {self._current_device_label}."
            )
        else:
            self._status_label.setText(
                f"Ready to capture screenshots for {self._current_game}{self._scenario_suffix()} "
                f"on {self._current_device_label}."
            )

    def _reload_scenarios(self) -> None:
        if not hasattr(self, "_scenario_combo_current"):
            return

        self._scenario_combo_current.blockSignals(True)
        self._scenario_combo_next.blockSignals(True)

        for combo in (self._scenario_combo_current, self._scenario_combo_next):
            combo.clear()
            combo.addItem("No Scenario", None)

        self._available_scenarios = self._scenario_registry.scenarios()
        for scenario in self._available_scenarios:
            self._scenario_combo_current.addItem(scenario, scenario)

        history_key = self._current_game or self._GLOBAL_SCENARIO_KEY
        stored_current, stored_next = self._scenario_history.get(history_key, (None, None))

        if not self._select_scenario_in_combo(self._scenario_combo_current, stored_current):
            self._scenario_combo_current.setCurrentIndex(0)
            self._current_scenario = None
        else:
            self._current_scenario = stored_current

        self._populate_next_combo(stored_next)

        has_any = bool(self._available_scenarios)
        self._scenario_combo_current.setEnabled(has_any)
        self._scenario_combo_next.setEnabled(has_any and self._scenario_combo_next.count() > 1)

        self._scenario_combo_current.blockSignals(False)
        self._scenario_combo_next.blockSignals(False)
        self._scenario_history[history_key] = (self._current_scenario, self._next_scenario)
        self._update_capture_state()

    def _select_scenario_in_combo(self, combo: QComboBox, scenario: Optional[str]) -> bool:
        if not scenario:
            return False
        index = combo.findData(scenario)
        if index == -1:
            index = combo.findText(scenario)
        if index == -1:
            return False
        combo.setCurrentIndex(index)
        return True

    def _populate_next_combo(self, desired: Optional[str]) -> None:
        previous = desired if desired else self._next_scenario
        self._scenario_combo_next.blockSignals(True)
        self._scenario_combo_next.clear()
        self._scenario_combo_next.addItem("No Scenario", None)
        current = self._current_scenario
        for scenario in self._available_scenarios:
            if scenario == current:
                continue
            self._scenario_combo_next.addItem(scenario, scenario)

        self._scenario_combo_next.setEnabled(self._scenario_combo_next.count() > 1)

        if previous and self._select_scenario_in_combo(self._scenario_combo_next, previous):
            self._next_scenario = previous
        else:
            self._scenario_combo_next.setCurrentIndex(0)
            self._next_scenario = None
        self._scenario_combo_next.blockSignals(False)

    def _on_current_scenario_changed(self, index: int) -> None:
        data = self._scenario_combo_current.itemData(index)
        scenario = data if isinstance(data, str) and data.strip() else None
        self._current_scenario = scenario
        self._populate_next_combo(self._next_scenario)
        self._store_scenario_history()
        self._update_capture_state()

    def _on_next_scenario_changed(self, index: int) -> None:
        data = self._scenario_combo_next.itemData(index)
        scenario = data if isinstance(data, str) and data.strip() else None
        self._next_scenario = scenario
        self._store_scenario_history()
        self._update_capture_state()

    def _on_manage_scenarios_clicked(self) -> None:
        self.scenario_manage_requested.emit()

    def _on_scenarios_changed(self, _: List[str]) -> None:
        self._reload_scenarios()

    def _scenario_suffix(self) -> str:
        if self._current_scenario and self._next_scenario:
            return f" (Scenario: {self._current_scenario} → {self._next_scenario})"
        if self._current_scenario:
            return f" (Scenario: {self._current_scenario})"
        if self._next_scenario:
            return f" (Next Scenario: {self._next_scenario})"
        return ""

    def _scenario_directory_name(self) -> Optional[str]:
        if not self._current_scenario:
            return None
        slug = self._scenario_slug(self._current_scenario)
        return slug or None

    def _scenario_filename_suffix(self) -> str:
        parts = []
        if self._current_scenario:
            parts.append(f"curr-{self._scenario_slug(self._current_scenario)}")
        if self._next_scenario:
            parts.append(f"next-{self._scenario_slug(self._next_scenario)}")
        return ("__" + "__".join(parts)) if parts else ""

    @staticmethod
    def _scenario_slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        return slug or "unspecified"

    def _store_scenario_history(self) -> None:
        key = self._current_game or self._GLOBAL_SCENARIO_KEY
        self._scenario_history[key] = (self._current_scenario, self._next_scenario)

    def _on_live_toggled(self, checked: bool) -> None:
        if checked:
            if not self._prepare_live_capture():
                self._live_toggle.setChecked(False)
                return
            self._start_live_capture()
        else:
            self._stop_live_capture()

    def _prepare_live_capture(self) -> bool:
        if not self._current_serial or not self._current_game:
            QMessageBox.warning(
                self,
                "Live Capture",
                "Please select both a device and a game before starting live capture.",
            )
            return False

        if self._pending_capture:
            QMessageBox.warning(
                self,
                "Live Capture",
                "A capture is already in progress. Please wait.",
            )
            return False

        try:
            interval = float(self._interval_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Live Capture", "Interval must be a number (seconds).")
            return False

        if interval < 0.2:
            QMessageBox.warning(self, "Live Capture", "Interval must be at least 0.2 seconds.")
            return False

        dataset_dir = self._dataset_directory(create=True)
        if dataset_dir is None:
            return False

        self._live_interval_ms = int(interval * 1000)
        return True

    def _start_live_capture(self) -> None:
        self._live_toggle.setText("Stop Live Capture")
        self._capture_button.setEnabled(False)
        self._pending_capture = False
        self._status_label.setText(
            f"Live capture active for {self._current_game}{self._scenario_suffix()} "
            f"on {self._current_device_label}."
        )
        self._trigger_live_capture()

    def _stop_live_capture(self) -> None:
        if self._live_timer.isActive():
            self._live_timer.stop()
        self._live_toggle.setText("Start Live Capture")
        self._pending_capture = False
        self._capture_button.setEnabled(self._current_serial is not None and self._current_game is not None)
        if self._current_game and self._current_serial:
            self._status_label.setText(
                f"Live capture stopped for {self._current_game}{self._scenario_suffix()}. "
                f"Ready for manual capture on {self._current_device_label}."
            )
        else:
            self._status_label.setText("Select a game and device to begin data collection.")

    def _trigger_live_capture(self) -> None:
        if self._pending_capture:
            return
        if not self._current_serial or not self._current_game:
            self._stop_live_capture()
            return

        self._pending_capture = True
        self._device_manager.capture_screenshot(self._current_serial)
        self._live_timer.start(getattr(self, "_live_interval_ms", 2000))

    def _on_preview_clicked(self, x: int, y: int) -> None:
        if not self._current_serial:
            QMessageBox.information(self, "Tap", "Select a device before sending taps.")
            return

        mapped = self._map_preview_point(x, y)
        if not mapped:
            return
        adj_x, adj_y, raw_x, raw_y = mapped

        self._awaiting_tap = True
        self._click_position_label.setText(
            f"Tapped preview at scaled=({adj_x}, {adj_y}) • raw=({raw_x}, {raw_y})"
        )
        self._status_label.setText(f"Sending tap to ({raw_x}, {raw_y}) on {self._current_device_label}…")
        self._device_manager.send_tap(self._current_serial, raw_x, raw_y)

    def _on_tap_sent(self, serial: str, x: int, y: int) -> None:
        if serial != self._current_serial:
            return
        self._awaiting_tap = False
        self._status_label.setText(
            f"Tap sent to ({x}, {y}) on {self._current_device_label}. Capture more or enable live mode."
        )

    def _on_preview_dragged(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if not self._current_serial:
            QMessageBox.information(self, "Swipe", "Select a device before sending swipes.")
            return
        if not self._last_raw_pixmap or not self._last_scaled_size:
            return

        start_mapped = self._map_preview_point(x1, y1)
        end_mapped = self._map_preview_point(x2, y2)
        if not start_mapped or not end_mapped:
            return

        start_adj_x, start_adj_y, start_raw_x, start_raw_y = start_mapped
        end_adj_x, end_adj_y, end_raw_x, end_raw_y = end_mapped

        self._awaiting_swipe = True
        self._click_position_label.setText(
            "Swipe preview scaled=({start}->{end}) • raw=({raw_start}->{raw_end})".format(
                start=(start_adj_x, start_adj_y),
                end=(end_adj_x, end_adj_y),
                raw_start=(start_raw_x, start_raw_y),
                raw_end=(end_raw_x, end_raw_y),
            )
        )
        self._status_label.setText(
            f"Sending swipe from ({start_raw_x}, {start_raw_y}) to ({end_raw_x}, {end_raw_y}) on {self._current_device_label}…"
        )
        self._device_manager.send_swipe(self._current_serial, start_raw_x, start_raw_y, end_raw_x, end_raw_y)

    def _on_swipe_sent(self, serial: str, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
        if serial != self._current_serial:
            return
        self._awaiting_swipe = False
        self._status_label.setText(
            f"Swipe sent from ({x1}, {y1}) to ({x2}, {y2}) on {self._current_device_label}."
        )

    def _map_preview_point(self, x: int, y: int) -> Optional[tuple[int, int, int, int]]:
        if not self._last_raw_pixmap or not self._last_scaled_size:
            return None

        scaled_w, scaled_h = self._last_scaled_size
        label_pixmap = self._preview_label.pixmap()
        if not label_pixmap:
            return None

        label_w = self._preview_label.width()
        label_h = self._preview_label.height()
        offset_x = max(0, (label_w - scaled_w) // 2)
        offset_y = max(0, (label_h - scaled_h) // 2)

        adj_x = x - offset_x
        adj_y = y - offset_y

        if not (0 <= adj_x <= scaled_w and 0 <= adj_y <= scaled_h):
            return None

        raw_w = self._last_raw_pixmap.width()
        raw_h = self._last_raw_pixmap.height()
        scale_x = raw_w / scaled_w
        scale_y = raw_h / scaled_h
        raw_x = int(adj_x * scale_x)
        raw_y = int(adj_y * scale_y)
        return adj_x, adj_y, raw_x, raw_y

    def _update_state_board_values(self, png_bytes: bytes) -> None:
        if not self._ocr_enabled:
            return

        if easyocr is None or Image is None:
            self._set_state_board_message(
                "OCR dependencies not available. Install easyocr and Pillow to enable this feature."
            )
            return

        if not self._current_game:
            self._set_state_board_message("Select a game to associate OCR regions.")
            return

        regions = [
            region
            for region in self._layout_registry.regions_for(self._current_game)
            if region.category.lower() == "state boards"
        ]

        if not regions:
            self._set_state_board_message(
                "No State Boards configured for this game. Create them in the Layout Designer."
            )
            return

        reader = self._get_ocr_reader()
        if reader is None:
            return

        try:
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load screenshot for OCR: {}", exc)
            self._set_state_board_message("Unable to load screenshot for OCR processing.")
            return

        width, height = image.size
        results: Dict[str, List[str]] = defaultdict(list)

        for region in regions:
            box = self._clamp_box(region.x, region.y, region.width, region.height, width, height)
            if box is None:
                logger.debug("Skipping region {} due to invalid bounds.", region.name)
                continue

            region_image = image.crop(box)
            np_region = np.array(region_image)
            try:
                texts = reader.readtext(
                    np_region,
                    detail=0,
                    allowlist="0123456789.,/()%+-",
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("OCR failed for region {}: {}", region.name, exc)
                continue

            cleaned = [text.strip() for text in texts if text and text.strip()]
            cleaned = self._apply_state_board_format(region, cleaned)
            if cleaned:
                results[region.name].extend(cleaned)

        if not results:
            self._set_state_board_message("No numeric values detected in State Boards.")
            return

        self._populate_state_board_results(results)

    def _apply_state_board_format(self, region, texts: List[str]) -> List[str]:
        filtered = [text for text in texts if text]
        if not filtered:
            return []

        format_hint = (getattr(region, "value_format", None) or "").strip()
        if not format_hint:
            return filtered if len(filtered) == 1 else [" ".join(filtered)]

        merged = "".join(filtered).replace(" ", "")
        if not merged:
            return []

        fmt_lower = format_hint.lower()
        if "/" in fmt_lower:
            ratio_source = merged.replace(" ", "")
            if ratio_source.count("/") == 1:
                numerator, denominator = ratio_source.split("/")
                num_text = self._format_integer_string(numerator)
                den_text = self._format_integer_string(denominator)
                formatted = f"{num_text}/{den_text}"
                if fmt_lower.startswith("(") and fmt_lower.endswith(")"):
                    formatted = f"({formatted})"
                return [formatted]
            return [ratio_source]

        if "%" in fmt_lower:
            normalized = merged.rstrip("%")
            number = self._coerce_number(normalized)
            if number is not None:
                return [f"{number:.2f}%"]
            if normalized:
                return [f"{normalized}%"]
            return [merged]

        if "currency" in fmt_lower or "," in merged:
            digits_only = merged.replace(",", "")
            formatted_currency = self._format_integer_string(digits_only)
            return [formatted_currency]

        return [merged]

    @staticmethod
    def _coerce_number(value: str) -> Optional[float]:
        try:
            cleaned = value.replace(",", "")
            return float(cleaned)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_integer_string(value: str) -> str:
        trimmed = value.strip()
        sign = ""
        if trimmed.startswith(("+", "-")):
            sign = trimmed[0]
            trimmed = trimmed[1:]
        stripped = "".join(ch for ch in trimmed if ch.isdigit())
        if not stripped:
            return value
        try:
            as_int = int(stripped)
            return f"{sign}{as_int:,}"
        except ValueError:
            return value

    def _set_state_board_message(self, text: str) -> None:
        self._state_board_message.setText(text)
        self._state_board_message.show()
        self._state_board_results_widget.hide()
        self._state_board_inputs.clear()
        self._state_board_captured.clear()
        self._clear_state_board_results()

    def _clear_state_board_results(self) -> None:
        while self._state_board_results_layout.count():
            item = self._state_board_results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _populate_state_board_results(self, results: Dict[str, List[str]]) -> None:
        previous_inputs = {name: edit.text() for name, edit in self._state_board_inputs.items()}
        self._state_board_inputs.clear()
        self._state_board_captured = {name: list(texts) for name, texts in results.items()}
        self._clear_state_board_results()

        self._state_board_message.hide()
        self._state_board_results_widget.show()

        for name, texts in results.items():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(12)

            name_label = QLabel(name)
            name_label.setMinimumWidth(120)
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            captured_label = QLabel(", ".join(texts))
            captured_label.setWordWrap(True)
            captured_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            correct_prompt = QLabel("Correct:")
            correct_prompt.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            input_edit = QLineEdit(previous_inputs.get(name, ""))
            input_edit.setPlaceholderText("Leave blank if captured value is correct")

            row_layout.addWidget(name_label)
            row_layout.addWidget(captured_label, stretch=1)
            row_layout.addWidget(correct_prompt)
            row_layout.addWidget(input_edit)

            self._state_board_results_layout.addWidget(row_widget)
            self._state_board_inputs[name] = input_edit

        self._state_board_results_layout.addStretch(1)

    def get_state_board_values(self) -> Dict[str, str]:
        collected: Dict[str, str] = {}
        for name, captured_values in self._state_board_captured.items():
            override_widget = self._state_board_inputs.get(name)
            override_value = override_widget.text().strip() if override_widget else ""
            if override_value:
                collected[name] = override_value
            else:
                collected[name] = ", ".join(captured_values)
        return collected

    def _get_ocr_reader(self) -> Optional["easyocr.Reader"]:
        if easyocr is None:
            return None

        if self._ocr_reader is not None:
            return self._ocr_reader

        try:
            self._ocr_reader = easyocr.Reader(["en"], gpu=False)
            return self._ocr_reader
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize EasyOCR Reader: {}", exc)
            self._set_state_board_message(f"OCR initialization failed: {exc}")
            self._ocr_enabled = False
            return None

    @staticmethod
    def _clamp_box(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> Optional[Tuple[int, int, int, int]]:
        if w <= 0 or h <= 0:
            return None
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(max_w, x + w)
        y2 = min(max_h, y + h)
        if x1 >= x2 or y1 >= y2:
            return None
        return x1, y1, x2, y2

