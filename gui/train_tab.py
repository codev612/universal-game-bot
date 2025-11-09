from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
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

from core.device_manager import DeviceManager
from core.layout_registry import LayoutRegistry


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(int(event.position().x()), int(event.position().y()))
        super().mousePressEvent(event)


class TrainTab(QWidget):
    """
    Training workflow tab: captures labeled screenshots for dataset creation.

    Screenshots are saved under the configured dataset directory, organized by game name.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        layout_registry: LayoutRegistry,  # reserved for future use (e.g., auto-annotating regions)
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._device_manager = device_manager
        self._layout_registry = layout_registry

        self._current_serial: Optional[str] = None
        self._current_device_label = "No device selected"
        self._current_game: Optional[str] = None

        self._pending_capture: bool = False
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._trigger_live_capture)
        self._last_raw_pixmap: Optional[QPixmap] = None
        self._last_scaled_size: Optional[tuple[int, int]] = None
        self._awaiting_tap: bool = False

        self._build_ui()
        self._wire_signals()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_group = QGroupBox("Context")
        info_form = QFormLayout(info_group)

        self._device_value = QLabel(self._current_device_label)
        self._game_value = QLabel("No game selected")

        info_form.addRow("Device:", self._device_value)
        info_form.addRow("Game:", self._game_value)

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

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumSize(720, 440)
        scroll_area.setWidget(self._preview_label)

        preview_layout.addWidget(scroll_area)
        layout.addWidget(self._preview_group, stretch=1)

    def _wire_signals(self) -> None:
        self._device_manager.error_occurred.connect(self._on_error)
        self._device_manager.screenshot_captured.connect(self._on_screenshot_captured)
        self._device_manager.tap_sent.connect(self._on_tap_sent)

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
        self._current_game = game_name
        self._game_value.setText(game_name or "No game selected")
        self._update_capture_state()

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
        self._status_label.setText(f"Capturing screenshot from {self._current_device_label}…")
        self._device_manager.capture_screenshot(self._current_serial)

    # Signal handlers ----------------------------------------------------------

    def _on_screenshot_captured(self, serial: str, data: bytes) -> None:
        if not self._pending_capture or serial != self._current_serial:
            return

        self._pending_capture = False
        self._capture_button.setEnabled(True)
        if self._live_toggle.isChecked():
            self._status_label.setText(f"Live capture active for {self._current_game}…")

        dataset_dir = self._dataset_directory(create=True)
        if dataset_dir is None:
            return

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        game_dir = dataset_dir / (self._current_game or "unknown_game")
        game_dir.mkdir(parents=True, exist_ok=True)
        file_path = game_dir / f"{timestamp}.png"

        try:
            file_path.write_bytes(data)
        except OSError as exc:
            logger.error("Failed to write screenshot {}: {}", file_path, exc)
            QMessageBox.critical(self, "Capture Error", f"Failed to save screenshot:\n{exc}")
            return

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
        logger.info("Training screenshot saved to {}", file_path)

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
                f"{self._current_game} selected. Choose a device in the Devices tab to capture screenshots."
            )
        elif self._live_toggle.isChecked():
            self._status_label.setText(f"Live capture active for {self._current_game} on {self._current_device_label}.")
        else:
            self._status_label.setText(
                f"Ready to capture screenshots for {self._current_game} on {self._current_device_label}."
            )

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
        self._status_label.setText(f"Live capture active for {self._current_game} on {self._current_device_label}.")
        self._trigger_live_capture()

    def _stop_live_capture(self) -> None:
        if self._live_timer.isActive():
            self._live_timer.stop()
        self._live_toggle.setText("Start Live Capture")
        self._pending_capture = False
        self._capture_button.setEnabled(self._current_serial is not None and self._current_game is not None)
        if self._current_game and self._current_serial:
            self._status_label.setText(
                f"Live capture stopped. Ready for manual capture on {self._current_device_label}."
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
        if not self._last_raw_pixmap or not self._last_scaled_size:
            return
        if not self._current_serial:
            QMessageBox.information(self, "Tap", "Select a device before sending taps.")
            return

        scaled_w, scaled_h = self._last_scaled_size
        label_pixmap = self._preview_label.pixmap()
        if not label_pixmap:
            return

        # Ensure clicks within the displayed pixmap bounds
        if not (0 <= x <= scaled_w and 0 <= y <= scaled_h):
            return

        raw_w = self._last_raw_pixmap.width()
        raw_h = self._last_raw_pixmap.height()
        scale_x = raw_w / scaled_w
        scale_y = raw_h / scaled_h
        raw_x = int(x * scale_x)
        raw_y = int(y * scale_y)

        self._awaiting_tap = True
        self._status_label.setText(f"Sending tap to ({raw_x}, {raw_y}) on {self._current_device_label}…")
        self._device_manager.send_tap(self._current_serial, raw_x, raw_y)

    def _on_tap_sent(self, serial: str, x: int, y: int) -> None:
        if serial != self._current_serial:
            return
        self._awaiting_tap = False
        self._status_label.setText(
            f"Tap sent to ({x}, {y}) on {self._current_device_label}. Capture more or enable live mode."
        )

