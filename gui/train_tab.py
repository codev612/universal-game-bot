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
        self._last_screenshot_bytes: Optional[bytes] = None
        self._awaiting_tap: bool = False
        self._awaiting_swipe: bool = False
        self._ocr_reader: Optional["easyocr.Reader"] = None
        self._ocr_enabled: bool = True

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
        self._ocr_results_label = QLabel(
            "Configure State Boards in the Layout Designer to see OCR results here."
        )
        self._ocr_results_label.setWordWrap(True)
        ocr_layout.addWidget(self._ocr_results_label)
        layout.addWidget(self._ocr_results_group)

    def _wire_signals(self) -> None:
        self._device_manager.error_occurred.connect(self._on_error)
        self._device_manager.screenshot_captured.connect(self._on_screenshot_captured)
        self._device_manager.tap_sent.connect(self._on_tap_sent)
        self._device_manager.swipe_sent.connect(self._on_swipe_sent)

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
        if self._last_screenshot_bytes:
            self._update_state_board_values(self._last_screenshot_bytes)
        else:
            self._ocr_results_label.setText(
                "Capture a screenshot to populate State Board values."
            )

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
            self._ocr_results_label.setText(
                "OCR dependencies not available. Install easyocr and Pillow to enable this feature."
            )
            return

        if not self._current_game:
            self._ocr_results_label.setText("Select a game to associate OCR regions.")
            return

        regions = [
            region
            for region in self._layout_registry.regions_for(self._current_game)
            if region.category.lower() == "state boards"
        ]

        if not regions:
            self._ocr_results_label.setText(
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
            self._ocr_results_label.setText("Unable to load screenshot for OCR processing.")
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
                    allowlist="0123456789.,",
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("OCR failed for region {}: {}", region.name, exc)
                continue

            cleaned = [text.strip() for text in texts if text and text.strip()]
            if cleaned:
                results[region.name].extend(cleaned)

        if not results:
            self._ocr_results_label.setText("No numeric values detected in State Boards.")
            return

        lines = []
        for name, texts in results.items():
            lines.append(f"{name}: {', '.join(texts)}")
        self._ocr_results_label.setText("\n".join(lines))

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
            self._ocr_results_label.setText(f"OCR initialization failed: {exc}")
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

