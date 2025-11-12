from __future__ import annotations

import datetime as dt
import io
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PyQt6.QtCore import QEvent, QObject, QProcess, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
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

from core.gpu_manager import get_gpu_manager, is_gpu_available
from core.gpu_image_processor import get_gpu_image_processor

easyocr = None
Image = ImageEnhance = ImageFilter = ImageOps = None
torch = None

DEFAULT_DETECTION_THRESHOLDS = {
    "exact": 0.95,
    "template": 0.88,
    "multiscale": 0.85,
    "orb": 0.80,
}
DEFAULT_COLOR_TOLERANCE = 35.0
def _lazy_imports() -> None:
    global easyocr, Image, ImageEnhance, ImageFilter, ImageOps, torch
    if easyocr is None:
        try:
            import easyocr as _easyocr  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            easyocr = None  # type: ignore[assignment]
            logger.warning("EasyOCR not available: {}", exc)
        else:
            easyocr = _easyocr
    if Image is None:
        try:
            from PIL import Image as _Image, ImageEnhance as _ImageEnhance, ImageFilter as _ImageFilter, ImageOps as _ImageOps  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            Image = None  # type: ignore[assignment]
            ImageEnhance = None  # type: ignore[assignment]
            ImageFilter = None  # type: ignore[assignment]
            ImageOps = None  # type: ignore[assignment]
            logger.warning("Pillow not available: {}", exc)
        else:
            Image = _Image
            ImageEnhance = _ImageEnhance
            ImageFilter = _ImageFilter
            ImageOps = _ImageOps
    if torch is None:
        try:
            import torch as _torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            torch = None
            logger.warning("PyTorch not available: {}", exc)
        else:
            torch = _torch

from core.device_manager import DeviceManager
from core.layout_registry import LayoutRegistry
from core.scenario_registry import ScenarioRegistry
from core.player_state_registry import PlayerStateRegistry
from core.policy_runner import PolicyRunner, PolicyActionPrediction
from core.rule_engine import RuleEngine, RuleSnapshot, SnippetObservation
from core.training_sample import ActionRecord, TrainingSample, TrainingSampleLogger
from gui.rule_editor_dialog import RuleEditorDialog
from gui.player_state_dialog import PlayerStateDialog


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
        player_state_registry: PlayerStateRegistry,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        _lazy_imports()
        self._device_manager = device_manager
        self._layout_registry = layout_registry
        self._scenario_registry = scenario_registry
        self._player_state_registry = player_state_registry

        self._current_serial: Optional[str] = None
        self._current_device_label = "No device selected"
        self._current_game: Optional[str] = None
        self._current_scenario: Optional[str] = None
        self._next_scenario: Optional[str] = None
        self._scenario_history: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._player_state_presets: List[str] = list(self._player_state_registry.player_states())
        self._player_state_history: Dict[str, Optional[str]] = {}
        self._current_player_state: Optional[str] = None
        self._updating_player_state = False
        self._current_action_target: Optional[str] = None
        self._auto_target_guess: Optional[str] = None
        self._updating_target_combo = False
        self._last_detection_outcomes: Dict[str, bool] = {}
        self._last_detection_scores: Dict[str, float] = {}
        self._available_scenarios: List[str] = []
        self._gpu_supported = bool(torch and torch.cuda.is_available())
        self._use_gpu = self._gpu_supported

        self._pending_capture: bool = False
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._trigger_live_capture)
        self._last_raw_pixmap: Optional[QPixmap] = None
        self._last_scaled_size: Optional[tuple[int, int]] = None
        self._last_screenshot_bytes: Optional[bytes] = None
        self._awaiting_tap: bool = False
        self._awaiting_swipe: bool = False
        self._ocr_reader: Optional[Any] = None
        self._ocr_enabled: bool = True
        self._state_board_inputs: Dict[str, QLineEdit] = {}
        self._state_board_captured: Dict[str, List[str]] = {}
        self._sample_logger = TrainingSampleLogger(Path("data/training/training_samples.jsonl"))
        self._last_screenshot_path: Optional[Path] = None
        self._training_process: Optional[QProcess] = None
        self._snippet_matches: List[dict] = []
        self._snippet_mask_cache: dict[str, Optional[np.ndarray]] = {}
        (
            self._detection_thresholds,
            self._color_tolerance_default,
        ) = self._load_detection_config()
        
        # Initialize GPU acceleration
        self._gpu_manager = get_gpu_manager()
        self._gpu_processor = get_gpu_image_processor()
        self._gpu_enabled = is_gpu_available()
        logger.info("GPU acceleration enabled: {}", self._gpu_enabled)
        
        try:
            self._orb_detector = cv2.ORB_create()
            self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ORB detector unavailable: {}", exc)
            self._orb_detector = None
            self._bf_matcher = None
        cuda_mod = getattr(cv2, "cuda", None)
        if cuda_mod is None:
            self._cuda_available = False
        else:
            try:
                self._cuda_available = cuda_mod.getCudaEnabledDeviceCount() > 0
            except Exception as exc:  # noqa: BLE001
                logger.debug("CUDA detection failed: {}", exc)
                self._cuda_available = False

        self._policy_runner: Optional[PolicyRunner] = None
        self._policy_enabled: bool = False
        self._policy_model_path: Optional[Path] = None
        self._policy_metadata_path: Optional[Path] = None
        self._last_policy_prediction: Optional[PolicyActionPrediction] = None
        self._rule_engine = RuleEngine(self._scenario_registry)
        self._rule_snapshot_path = Path("rules") / "last_snapshot.json"
        self._last_rule_snapshot: Optional[RuleSnapshot] = None

        self._build_ui()
        self._wire_signals()
        self._reload_scenarios()
        self._restore_player_state()
        self._scenario_registry.scenarios_changed.connect(self._on_scenarios_changed)
        self._player_state_registry.states_changed.connect(self._on_player_states_changed)
        self._on_player_states_changed(self._player_state_registry.player_states())
        self._refresh_gpu_support()
        self._update_training_controls()
        self._update_policy_controls()
        self._update_rule_preview()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_group = QGroupBox("Context")
        info_form = QFormLayout(info_group)

        self._device_value = QLabel(self._current_device_label)
        self._game_value = QLabel("No game selected")
        self._gpu_status_value = QLabel("Checking...")

        info_form.addRow("Device:", self._device_value)
        info_form.addRow("Game:", self._game_value)
        info_form.addRow("GPU Acceleration:", self._gpu_status_value)

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

        self._player_state_combo = QComboBox()
        self._player_state_combo.setEditable(True)
        player_state_row_widget = QWidget()
        player_state_row_layout = QHBoxLayout(player_state_row_widget)
        player_state_row_layout.setContentsMargins(0, 0, 0, 0)
        player_state_row_layout.setSpacing(6)
        self._player_state_combo.addItem("No player state")
        for option in self._player_state_presets:
            self._player_state_combo.addItem(option)
        self._manage_player_states_button = QPushButton("Manage…")
        self._manage_player_states_button.clicked.connect(self._on_manage_player_states_clicked)
        player_state_row_layout.addWidget(self._player_state_combo, stretch=1)
        player_state_row_layout.addWidget(self._manage_player_states_button)
        info_form.addRow("Player state:", player_state_row_widget)

        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_group)

        dataset_row = QHBoxLayout()
        self._dataset_path_edit = QLineEdit(str(Path("data") / "training").replace("\\", "/"))
        self._dataset_path_edit.setPlaceholderText("Directory to store captured screenshots")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._on_browse_dataset)
        dataset_row.addWidget(self._dataset_path_edit, stretch=1)
        dataset_row.addWidget(browse_button)

        self._reset_dataset_button = QPushButton("Reset Dataset")
        self._save_images_checkbox = QCheckBox("Save captured images to dataset")
        self._save_images_checkbox.setChecked(True)

        dataset_layout.addLayout(dataset_row)
        dataset_layout.addWidget(self._reset_dataset_button)
        dataset_layout.addWidget(self._save_images_checkbox)

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

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        header_row = QHBoxLayout()
        header_row.addWidget(info_group, stretch=1)
        header_row.addWidget(dataset_group, stretch=1)
        content_layout.addLayout(header_row)
        content_layout.addLayout(action_row)
        content_layout.addWidget(self._status_label)
        self._detection_status_group = QGroupBox("Snippet Detection")
        detection_layout = QVBoxLayout(self._detection_status_group)
        self._detection_status_message = QLabel("No snippets processed yet.")
        self._detection_status_message.setWordWrap(True)
        detection_layout.addWidget(self._detection_status_message)
        self._detection_results_widget = QWidget()
        self._detection_results_layout = QVBoxLayout(self._detection_results_widget)
        self._detection_results_layout.setContentsMargins(0, 0, 0, 0)
        self._detection_results_layout.setSpacing(6)
        detection_layout.addWidget(self._detection_results_widget)
        self._detection_results_widget.hide()
        target_row = QHBoxLayout()
        target_label = QLabel("Action target snippet:")
        target_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._target_snippet_combo = QComboBox()
        self._target_snippet_combo.setEditable(True)
        target_row.addWidget(target_label)
        target_row.addWidget(self._target_snippet_combo, stretch=1)
        detection_layout.addLayout(target_row)
        self._update_target_snippet_combo([])

        self._rule_group = QGroupBox("Rule Engine")
        rule_layout = QVBoxLayout(self._rule_group)
        rule_layout.setSpacing(6)
        self._rule_summary_label = QLabel(
            "Capture snippets or OCR values to preview matching rules."
        )
        self._rule_summary_label.setWordWrap(True)
        self._rule_summary_label.setStyleSheet("color: #444444;")
        rule_layout.addWidget(self._rule_summary_label)

        self._rule_action_label = QLabel("")
        self._rule_action_label.setWordWrap(True)
        rule_layout.addWidget(self._rule_action_label)

        rule_buttons = QHBoxLayout()
        self._rule_reload_button = QPushButton("Reload")
        self._rule_reload_button.clicked.connect(self._on_reload_rules_clicked)
        self._rule_edit_button = QPushButton("Edit…")
        self._rule_edit_button.clicked.connect(self._on_edit_rules_clicked)
        self._rule_test_button = QPushButton("Test Now")
        self._rule_test_button.clicked.connect(self._on_test_rules_clicked)
        self._rule_run_button = QPushButton("Run Action")
        self._rule_run_button.clicked.connect(self._on_run_rules_clicked)
        rule_buttons.addWidget(self._rule_reload_button)
        rule_buttons.addWidget(self._rule_edit_button)
        rule_buttons.addWidget(self._rule_test_button)
        rule_buttons.addWidget(self._rule_run_button)
        rule_buttons.addStretch(1)
        rule_layout.addLayout(rule_buttons)

        self._preview_group = QGroupBox("Last Capture Preview")
        preview_layout = QVBoxLayout(self._preview_group)

        self._preview_label = ClickableLabel("No capture yet.")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.clicked.connect(self._on_preview_clicked)
        self._preview_label.dragged.connect(self._on_preview_dragged)

        self._preview_scroll = QScrollArea()
        self._preview_scroll.setWidgetResizable(True)
        self._preview_scroll.setMinimumSize(720, 440)
        self._preview_scroll.setWidget(self._preview_label)
        self._preview_scroll.viewport().installEventFilter(self)

        self._click_position_label = QLabel("Click on the preview to send a tap. Coordinates will appear here.")
        self._click_position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        preview_layout.addWidget(self._preview_scroll)
        preview_layout.addWidget(self._click_position_label)

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

        self._training_group = QGroupBox("AI Training")
        training_layout = QVBoxLayout(self._training_group)

        self._gpu_info_label = QLabel("")
        self._gpu_info_label.setStyleSheet("color: #666666; font-size: 11px;")
        training_layout.addWidget(self._gpu_info_label)

        self._gpu_checkbox = QCheckBox("Use GPU")
        self._gpu_checkbox.setChecked(self._use_gpu)
        if not self._gpu_supported:
            self._gpu_checkbox.setEnabled(False)
            self._gpu_checkbox.setToolTip("CUDA GPU not detected in this environment.")
        training_layout.addWidget(self._gpu_checkbox)

        detection_row = QHBoxLayout()
        detection_row.addWidget(QLabel("Snippet detection:"))
        self._detection_method_combo = QComboBox()
        self._detection_method_combo.addItems(
            [
                "Template Matching",
                "Multi-scale Template",
                "ORB Features",
                "All Methods (best)",
            ]
        )
        detection_row.addWidget(self._detection_method_combo, stretch=1)
        training_layout.addLayout(detection_row)

        training_row = QVBoxLayout()
        self._start_training_button = QPushButton("Start Training")
        self._stop_training_button = QPushButton("Stop Training")
        self._stop_training_button.setEnabled(False)
        training_row.addWidget(self._start_training_button)
        training_row.addWidget(self._stop_training_button)
        training_layout.addLayout(training_row)

        self._training_status_label = QLabel("Training idle.")
        self._training_status_label.setWordWrap(True)
        training_layout.addWidget(self._training_status_label)

        note_label = QLabel(
            "Runs training/train_policy.py with collected samples. Ensure training_samples.jsonl exists."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666666; font-size: 11px;")
        training_layout.addWidget(note_label)

        training_layout.addSpacing(8)

        policy_header = QLabel("Policy model:")
        policy_header.setStyleSheet("font-weight: bold;")
        training_layout.addWidget(policy_header)

        self._policy_model_label = QLabel("No checkpoint loaded.")
        self._policy_model_label.setWordWrap(True)
        self._policy_model_label.setStyleSheet("color: #555555;")
        training_layout.addWidget(self._policy_model_label)

        policy_buttons = QHBoxLayout()
        self._policy_load_button = QPushButton("Load Checkpoint…")
        self._policy_unload_button = QPushButton("Unload")
        self._policy_unload_button.setEnabled(False)
        policy_buttons.addWidget(self._policy_load_button)
        policy_buttons.addWidget(self._policy_unload_button)
        training_layout.addLayout(policy_buttons)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Min confidence:"))
        self._policy_confidence_spin = QDoubleSpinBox()
        self._policy_confidence_spin.setRange(0.0, 1.0)
        self._policy_confidence_spin.setSingleStep(0.05)
        self._policy_confidence_spin.setValue(0.6)
        self._policy_confidence_spin.setDecimals(2)
        threshold_row.addWidget(self._policy_confidence_spin, stretch=1)
        training_layout.addLayout(threshold_row)

        self._policy_toggle_button = QPushButton("Enable Policy Control")
        self._policy_toggle_button.setCheckable(True)
        self._policy_toggle_button.setEnabled(False)
        training_layout.addWidget(self._policy_toggle_button)

        self._policy_status_label = QLabel("Policy control idle.")
        self._policy_status_label.setWordWrap(True)
        training_layout.addWidget(self._policy_status_label)

        if torch is None:
            self._policy_load_button.setEnabled(False)
            self._policy_toggle_button.setEnabled(False)
            self._policy_confidence_spin.setEnabled(False)
            self._policy_status_label.setText("PyTorch is required for policy inference.")

        left_column_widget = QWidget()
        left_column_layout = QVBoxLayout(left_column_widget)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(12)
        left_column_layout.addWidget(self._ocr_results_group)
        left_column_layout.addWidget(self._detection_status_group)
        left_column_layout.addWidget(self._rule_group)
        left_column_layout.addStretch(1)

        panels_row = QHBoxLayout()
        panels_row.addWidget(left_column_widget, stretch=1)
        panels_row.addWidget(self._preview_group, stretch=2)
        panels_row.addWidget(self._training_group, stretch=1)
        content_layout.addLayout(panels_row)

        outer_scroll = QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setWidget(content_widget)
        layout.addWidget(outer_scroll, stretch=1)

    def _wire_signals(self) -> None:
        self._device_manager.error_occurred.connect(self._on_error)
        self._device_manager.screenshot_captured.connect(self._on_screenshot_captured)
        self._device_manager.tap_sent.connect(self._on_tap_sent)
        self._device_manager.swipe_sent.connect(self._on_swipe_sent)
        self._scenario_combo_current.currentIndexChanged.connect(self._on_current_scenario_changed)
        self._scenario_combo_next.currentIndexChanged.connect(self._on_next_scenario_changed)
        self._manage_scenarios_button.clicked.connect(self._on_manage_scenarios_clicked)
        self._start_training_button.clicked.connect(self._on_start_training_clicked)
        self._stop_training_button.clicked.connect(self._on_stop_training_clicked)
        self._gpu_checkbox.toggled.connect(self._on_gpu_checkbox_changed)
        self._reset_dataset_button.clicked.connect(self._on_reset_dataset_clicked)
        self._player_state_combo.currentTextChanged.connect(self._on_player_state_changed)
        self._target_snippet_combo.currentTextChanged.connect(self._on_target_snippet_changed)
        if torch is not None:
            self._policy_load_button.clicked.connect(self._on_load_policy_model_clicked)
            self._policy_unload_button.clicked.connect(self._on_unload_policy_model_clicked)
            self._policy_toggle_button.toggled.connect(self._on_policy_toggle_changed)

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
        self._update_policy_controls()

    def set_active_game(self, game_name: Optional[str]) -> None:
        previous_game = self._current_game
        if previous_game != game_name:
            key = previous_game or self._GLOBAL_SCENARIO_KEY
            self._scenario_history[key] = (self._current_scenario, self._next_scenario)
            self._player_state_history[key] = self._current_player_state

        self._current_game = game_name
        self._game_value.setText(game_name or "No game selected")
        self._reload_scenarios()
        self._restore_player_state()
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

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        game_name = self._current_game or "unknown_game"
        file_path: Optional[Path] = None

        if self._save_images_checkbox.isChecked():
            dataset_dir = self._dataset_directory(create=True)
            if dataset_dir is None:
                return
            game_dir = dataset_dir / game_name
            game_dir.mkdir(parents=True, exist_ok=True)
            target_dir = game_dir
            scenario_dir = self._scenario_directory_name()
            if scenario_dir:
                target_dir = game_dir / scenario_dir
                target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / f"{timestamp}{self._scenario_filename_suffix()}.png"

        if file_path is not None:
            try:
                file_path.write_bytes(data)
            except OSError as exc:
                logger.error("Failed to write screenshot {}: {}", file_path, exc)
                QMessageBox.critical(self, "Capture Error", f"Failed to save screenshot:\n{exc}")
                return
            self._last_screenshot_path = file_path
            status_message = f"Screenshot saved to {file_path}"
        else:
            self._last_screenshot_path = None
            status_message = "Screenshot captured (not saved to disk)."

        self._last_screenshot_bytes = bytes(data)
        pixmap = QPixmap()
        if pixmap.loadFromData(data, "PNG"):
            self._last_raw_pixmap = pixmap
            self._refresh_preview_display()
        else:
            self._preview_label.setText("Unable to load preview image.")
            self._last_raw_pixmap = None
            self._last_scaled_size = None

        snippet_summary = self._detect_snippets_in_screenshot(data)
        status_output = status_message
        if snippet_summary:
            status_output = f"{status_output} | {snippet_summary}"

        self._refresh_preview_display()
        self._status_label.setText(status_output)
        self._click_position_label.setText("Click on the preview to send a tap. Coordinates will appear here.")
        if file_path is not None:
            logger.info("Training screenshot saved to {}", file_path)
        else:
            logger.info("Training screenshot captured without saving to disk.")
        self._update_state_board_values(data)
        self._detect_snippets_in_screenshot(data)
        self._maybe_run_policy_control(data)
        if file_path is not None:
            self._log_idle_sample()

    def _detect_snippets_in_screenshot(self, png_bytes: bytes) -> str:
        self._snippet_matches = []
        if not self._current_game:
            return ""

        metadata_path = self._snippet_metadata_path()
        if metadata_path is None or not metadata_path.exists():
            return ""

        try:
            entries = self._load_snippet_entries(metadata_path)
        except OSError as exc:
            logger.warning("Failed to read snippet metadata: {}", exc)
            return ""
        if not entries:
            return ""

        np_bytes = np.frombuffer(png_bytes, np.uint8)
        screenshot_color = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
        if screenshot_color is None:
            return ""
        screenshot_gray = cv2.cvtColor(screenshot_color, cv2.COLOR_BGR2GRAY)
        img_height, img_width = screenshot_gray.shape[:2]

        summary_parts: List[str] = []
        detection_outcomes: Dict[str, bool] = {}
        detection_scores: Dict[str, float] = {}
        selected_method = (
            self._detection_method_combo.currentText()
            if hasattr(self, "_detection_method_combo")
            else "Template Matching"
        )

        for entry in entries:
            snippet_path = entry.get("file")
            if not snippet_path:
                continue
            snippet_raw = cv2.imread(snippet_path, cv2.IMREAD_UNCHANGED)
            if snippet_raw is None:
                continue
            snippet_alpha: Optional[np.ndarray] = None
            if snippet_raw.ndim == 3:
                if snippet_raw.shape[2] == 4:
                    snippet_alpha = snippet_raw[:, :, 3]
                    snippet_img = cv2.cvtColor(snippet_raw, cv2.COLOR_BGRA2BGR)
                else:
                    snippet_img = snippet_raw
            else:
                snippet_img = cv2.cvtColor(snippet_raw, cv2.COLOR_GRAY2BGR)
            snippet_gray = cv2.cvtColor(snippet_img, cv2.COLOR_BGR2GRAY)
            sh, sw = snippet_gray.shape[:2]
            search = entry.get("search") or {}
            mode = (search.get("mode") or "global").lower()
            region = search.get("region") or {}
            target_color = self._parse_hex_color(search.get("color"))
            use_mask = bool(search.get("use_mask"))
            thresholds = self._thresholds_for_entry(search)
            color_tolerance = self._color_tolerance_for_entry(search)
            mask = None
            if use_mask:
                if snippet_alpha is not None and cv2.countNonZero(snippet_alpha) > 0:
                    mask = snippet_alpha
                else:
                    mask = self._get_snippet_mask(snippet_path, snippet_img)
                    if mask is None:
                        logger.debug("Mask generation failed for snippet {}; falling back to no mask.", snippet_path)
                        use_mask = False

            fallback_rect: Optional[Tuple[int, int, int, int]] = None
            if mode == "exact":
                rect = entry.get("rect") or {}
                x1 = int(rect.get("x", 0))
                y1 = int(rect.get("y", 0))
                w = int(rect.get("width", sw))
                h = int(rect.get("height", sh))
                x2 = x1 + w
                y2 = y1 + h
                fallback_rect = (x1, y1, w, h)
            elif mode == "custom":
                x1 = int(region.get("x1", 0))
                y1 = int(region.get("y1", 0))
                x2 = int(region.get("x2", img_width))
                y2 = int(region.get("y2", img_height))
            else:  # global
                x1, y1, x2, y2 = 0, 0, img_width, img_height

            x1 = max(0, min(img_width, x1))
            y1 = max(0, min(img_height, y1))
            x2 = max(x1 + 1, min(img_width, x2))
            y2 = max(y1 + 1, min(img_height, y2))

            roi_gray = screenshot_gray[y1:y2, x1:x2]
            roi_color = screenshot_color[y1:y2, x1:x2]
            skip_detection = roi_gray.shape[0] < sh or roi_gray.shape[1] < sw

            match_info: Optional[dict] = None
            chosen_method: Optional[str] = None
            if not skip_detection:
                match_info, chosen_method = self._run_detection_methods(
                    selected_method,
                    mode,
                    roi_gray,
                    snippet_gray,
                    x1,
                    y1,
                    thresholds,
                    mask,
                    roi_color,
                    snippet_img,
                )

            debug_name = entry.get("export_name") or entry.get("name") or "snippet"
            chosen_method = match_info.get("method") if match_info else None
            if match_info and target_color is not None:
                accepted = self._matches_target_color(
                    screenshot_color,
                    match_info,
                    target_color,
                    color_tolerance,
                )
            else:
                accepted = bool(match_info)
            try:
                self._save_debug_snippet_crop(
                    match_info,
                    accepted,
                    screenshot_color,
                    mask,
                    debug_name,
                    fallback_rect,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to save debug snippet crop: {}", exc)

            if match_info and accepted:
                match_info["name"] = debug_name
                match_info["method"] = chosen_method or (
                self._detection_method_combo.currentText()
                if hasattr(self, "_detection_method_combo")
                else None
            )
                self._snippet_matches.append(match_info)
                label = match_info["name"]
                if match_info.get("method"):
                    label = f"{label} [{match_info['method']}]"
                summary_parts.append(f"{label} ({match_info['score']:.2f})")
                detection_outcomes[debug_name] = True
                detection_scores[debug_name] = float(match_info.get("score", 0.0))
            else:
                detection_outcomes[debug_name] = False
                if match_info:
                    detection_scores[debug_name] = float(match_info.get("score", 0.0))

        self._last_detection_outcomes = dict(detection_outcomes)
        self._last_detection_scores = dict(detection_scores)
        self._update_detection_results(detection_outcomes, detection_scores)
        self._update_rule_preview()

        if summary_parts:
            return "Snippets detected."

        if detection_outcomes:
            any_hits = any(detection_outcomes.values())
            return "Snippets detected." if any_hits else "Snippets detected: none matched."
        return ""

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

    def _get_snippet_mask(
        self,
        snippet_path: str,
        snippet_color: np.ndarray,
    ) -> Optional[np.ndarray]:
        key = str(snippet_path)
        if key in self._snippet_mask_cache:
            return self._snippet_mask_cache[key]
        mask = self._generate_snippet_mask(snippet_color)
        self._snippet_mask_cache[key] = mask
        return mask

    def _generate_snippet_mask(self, snippet_color: np.ndarray) -> Optional[np.ndarray]:
        # Try GPU-accelerated mask creation first
        if self._gpu_enabled:
            mask = self._gpu_processor.create_snippet_mask(snippet_color, method="combined")
            if mask is not None:
                return mask
        
        # Fallback to original CPU implementation
        try:
            gray = cv2.cvtColor(snippet_color, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return None
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask is None:
            return None
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        if cv2.countNonZero(mask) < 25:
            edges = cv2.Canny(gray, 50, 150)
            if edges is None:
                return None
            mask = cv2.dilate(edges, kernel, iterations=1)
        if cv2.countNonZero(mask) < 25:
            return None
        return mask

    def _load_detection_config(self) -> tuple[dict[str, float], float]:
        config_path = Path("config/snippet_detection.json")
        if not config_path.exists():
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with config_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            **DEFAULT_DETECTION_THRESHOLDS,
                            "color_tolerance": DEFAULT_COLOR_TOLERANCE,
                        },
                        handle,
                        indent=2,
                    )
            except OSError as exc:
                logger.warning("Failed to create snippet detection config: {}", exc)
            return dict(DEFAULT_DETECTION_THRESHOLDS), DEFAULT_COLOR_TOLERANCE
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read snippet detection config: {}", exc)
            return dict(DEFAULT_DETECTION_THRESHOLDS), DEFAULT_COLOR_TOLERANCE

        thresholds: dict[str, float] = dict(DEFAULT_DETECTION_THRESHOLDS)
        for key in ("exact", "template", "multiscale", "orb"):
            value = data.get(key)
            if value is None:
                continue
            try:
                thresholds[key] = float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid threshold {}={} in config; using default.", key, value)
        color_tol = DEFAULT_COLOR_TOLERANCE
        if "color_tolerance" in data:
            try:
                color_tol = float(data["color_tolerance"])
            except (TypeError, ValueError):
                logger.warning("Invalid color_tolerance={} in config; using default.", data["color_tolerance"])
        return thresholds, color_tol

    def _thresholds_for_entry(self, search: dict) -> dict[str, float]:
        base = search.get("threshold")

        def resolve(key: str, default_key: str) -> float:
            if key in search:
                try:
                    return float(search[key])
                except (TypeError, ValueError):
                    logger.warning("Invalid threshold value for {}: {}", key, search[key])
            if base is not None:
                try:
                    return float(base)
                except (TypeError, ValueError):
                    logger.warning("Invalid base threshold: {}", base)
            return self._detection_thresholds.get(default_key, DEFAULT_DETECTION_THRESHOLDS[default_key])

        return {
            "exact": resolve("threshold_exact", "exact"),
            "template": resolve("threshold_template", "template"),
            "multiscale": resolve("threshold_multiscale", "multiscale"),
            "orb": resolve("threshold_orb", "orb"),
        }

    def _color_tolerance_for_entry(self, search: dict) -> float:
        value = search.get("color_tolerance")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid color_tolerance override: {}", value)
        return self._color_tolerance_default

    def _run_detection_methods(
        self,
        selected_method: str,
        mode: str,
        roi_gray: np.ndarray,
        snippet_gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        thresholds: dict[str, float],
        mask: Optional[np.ndarray],
        roi_color: Optional[np.ndarray],
        snippet_color: Optional[np.ndarray],
    ) -> tuple[Optional[dict], Optional[str]]:
        candidates: list[dict] = []

        def consider(label: str, result: Optional[dict]) -> None:
            if result and isinstance(result, dict):
                entry = dict(result)
                entry["method"] = label
                candidates.append(entry)

        method_map = {
            "Template Matching": {"template"},
            "Multi-scale Template": {"multiscale"},
            "ORB Features": {"orb"},
            "All Methods (best)": {"template", "multiscale", "orb"},
        }
        requested = method_map.get(selected_method, {"template"})

        if mode == "exact":
            result = self._match_exact_snippet(
                roi_gray,
                snippet_gray,
                origin_x,
                origin_y,
                thresholds.get("exact", DEFAULT_DETECTION_THRESHOLDS["exact"]),
                mask,
                roi_color,
                snippet_color,
            )
            consider("Exact Match", result)

        template_result: Optional[dict] = None
        if "template" in requested:
            template_result = self._match_template(
                roi_gray,
                snippet_gray,
                origin_x,
                origin_y,
                thresholds.get("template", DEFAULT_DETECTION_THRESHOLDS["template"]),
                mask,
            )
            consider("Template Matching", template_result)

        if "multiscale" in requested:
            result = self._match_multiscale_template(
                roi_gray,
                snippet_gray,
                origin_x,
                origin_y,
                thresholds.get("multiscale", DEFAULT_DETECTION_THRESHOLDS["multiscale"]),
                mask,
            )
            consider("Multi-scale Template", result)

        if "orb" in requested:
            result = self._match_orb_snippet(
                roi_gray,
                snippet_gray,
                origin_x,
                origin_y,
                thresholds.get("orb", DEFAULT_DETECTION_THRESHOLDS["orb"]),
            )
            consider("ORB Features", result)
            if (
                result is None
                and template_result is None
                and "template" not in requested
            ):
                template_result = self._match_template(
                    roi_gray,
                    snippet_gray,
                    origin_x,
                    origin_y,
                    thresholds.get("template", DEFAULT_DETECTION_THRESHOLDS["template"]),
                    mask,
                )
                consider("Template Matching", template_result)

        if not candidates:
            return None, None
        best = max(candidates, key=lambda item: item.get("score", float("-inf")))
        return best, best.get("method")

    def _save_debug_snippet_crop(
        self,
        match_info: Optional[dict],
        accepted: bool,
        screenshot_color: np.ndarray,
        mask: Optional[np.ndarray],
        snippet_name: str,
        fallback_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if match_info:
            x = int(match_info.get("x", 0))
            y = int(match_info.get("y", 0))
            w = int(match_info.get("w", 0))
            h = int(match_info.get("h", 0))
        elif fallback_rect:
            x, y, w, h = fallback_rect
        else:
            return

        if w <= 0 or h <= 0:
            return

        img_h, img_w = screenshot_color.shape[:2]
        x1 = max(0, min(img_w, x))
        y1 = max(0, min(img_h, y))
        x2 = max(x1 + 1, min(img_w, x + w))
        y2 = max(y1 + 1, min(img_h, y + h))
        if x2 <= x1 or y2 <= y1:
            return

        crop = screenshot_color[y1:y2, x1:x2]
        if crop.size == 0:
            return

        alpha_channel: Optional[np.ndarray] = None
        if mask is not None:
            resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            alpha_channel = resized_mask

        if alpha_channel is None:
            alpha_channel = np.full((h, w), 255, dtype=np.uint8)

        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha_channel

        debug_dir = Path("data/debug_snippets")
        if self._current_game:
            debug_dir = debug_dir / self._slugify(self._current_game)
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        status = "hit" if accepted else "miss"
        filename = f"{timestamp}_{status}_{self._slugify(snippet_name)}.png"
        cv2.imwrite(str((debug_dir / filename).resolve()), rgba)

    @staticmethod
    def _slugify(value: str) -> str:
        text = value.strip().lower()
        text = re.sub(r"[^\w.-]+", "-", text)
        return text or "snippet"

    @staticmethod
    def _parse_hex_color(value: Optional[str]) -> Optional[Tuple[int, int, int]]:
        if not value:
            return None
        text = value.strip().lower()
        if not text:
            return None
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 3:
            try:
                text = "".join(ch * 2 for ch in text)
            except TypeError:
                return None
        if len(text) != 6:
            return None
        try:
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
        except ValueError:
            return None
        return (b, g, r)

    def _matches_target_color(
        self,
        screenshot_color: np.ndarray,
        match_info: dict,
        target_bgr: Tuple[int, int, int],
        tolerance: float = DEFAULT_COLOR_TOLERANCE,
    ) -> bool:
        x = int(match_info.get("x", 0))
        y = int(match_info.get("y", 0))
        w = int(match_info.get("w", 0))
        h = int(match_info.get("h", 0))
        if w <= 0 or h <= 0:
            return False
        img_h, img_w = screenshot_color.shape[:2]
        x1 = max(0, min(img_w, x))
        y1 = max(0, min(img_h, y))
        x2 = max(x1 + 1, min(img_w, x + w))
        y2 = max(y1 + 1, min(img_h, y + h))
        roi = screenshot_color[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        mean_bgr = roi.reshape(-1, 3).mean(axis=0)
        diff = float(np.linalg.norm(mean_bgr - np.array(target_bgr, dtype=np.float32)))
        return diff <= tolerance

    def _update_detection_results(
        self,
        outcomes: Dict[str, bool],
        scores: Dict[str, float],
    ) -> None:
        # Clear previous results
        while self._detection_results_layout.count():
            item = self._detection_results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        names = list(outcomes.keys())
        self._update_target_snippet_combo(names)

        target_name = self._current_action_target
        if not target_name:
            target_name = self._auto_target_guess

        if not outcomes:
            self._detection_status_message.setText("No snippets processed yet.")
            self._detection_results_widget.hide()
            self._auto_target_guess = None
            return

        self._detection_status_message.setText("Latest snippet detection results:")
        for name, outcome in outcomes.items():
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(12)
            name_label = QLabel(name)
            name_label.setMinimumWidth(140)
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            result_label = QLabel("Yes" if outcome else "No")
            result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            result_label.setStyleSheet(
                "color: #2e7d32;" if outcome else "color: #c62828;"
            )
            if target_name and name == target_name:
                result_label.setText(result_label.text() + " (target)")
                result_label.setStyleSheet(
                    "color: #1b5e20; font-weight: bold;"
                    if outcome
                    else "color: #b71c1c; font-weight: bold;"
                )
            score = scores.get(name)
            if score is not None:
                score_label = QLabel(f"Score: {score:.2f}")
                score_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                score_label = QLabel("")
            layout.addWidget(name_label, stretch=1)
            layout.addWidget(result_label)
            layout.addWidget(score_label)
            self._detection_results_layout.addWidget(row)

        self._detection_results_layout.addStretch(1)
        self._detection_results_widget.show()

    def _update_target_snippet_combo(self, names: List[str]) -> None:
        if not hasattr(self, "_target_snippet_combo"):
            return
        self._updating_target_combo = True
        try:
            current_manual = self._current_action_target
            current_auto = self._auto_target_guess
            current_text = self._target_snippet_combo.currentText().strip() if self._target_snippet_combo.count() else ""
            self._target_snippet_combo.clear()
            self._target_snippet_combo.addItem("Auto (nearest)")
            seen = set()
            for name in names:
                if name and name not in seen:
                    self._target_snippet_combo.addItem(name)
                    seen.add(name)
            if current_manual and current_manual in seen:
                self._target_snippet_combo.setCurrentText(current_manual)
            elif current_manual and current_manual not in seen:
                self._current_action_target = None
                if current_auto and current_auto in seen:
                    self._target_snippet_combo.setCurrentText(current_auto)
                else:
                    self._target_snippet_combo.setCurrentIndex(0)
            elif current_auto and current_auto in seen and not current_manual:
                self._target_snippet_combo.setCurrentText(current_auto)
            elif current_text and current_text not in {"", "Auto (nearest)"} and current_text in seen:
                self._target_snippet_combo.setCurrentText(current_text)
            else:
                self._target_snippet_combo.setCurrentIndex(0)
        finally:
            self._updating_target_combo = False

    def _refresh_detection_display(self) -> None:
        self._update_detection_results(self._last_detection_outcomes, self._last_detection_scores)

    def _set_auto_target_guess(self, name: Optional[str]) -> None:
        if name and name not in self._last_detection_outcomes:
            name = None
        self._auto_target_guess = name
        if self._current_action_target:
            self._refresh_detection_display()
            return
        if hasattr(self, "_target_snippet_combo"):
            self._updating_target_combo = True
            try:
                if name and self._target_snippet_combo.findText(name) != -1:
                    self._target_snippet_combo.setCurrentText(name)
                else:
                    self._target_snippet_combo.setCurrentIndex(0)
            finally:
                self._updating_target_combo = False
        self._refresh_detection_display()

    def _snippet_name_detected(self, name: Optional[str]) -> bool:
        return bool(name and name in self._last_detection_outcomes)

    def _known_snippet_names(self) -> List[str]:
        names: set[str] = set()
        path = self._snippet_metadata_path()
        if path and path.exists():
            try:
                entries = self._load_snippet_entries(path)
            except OSError:
                entries = []
            for entry in entries:
                candidate = entry.get("export_name") or entry.get("name")
                if candidate:
                    names.add(str(candidate))
        for detected_name in self._last_detection_outcomes.keys():
            if detected_name:
                names.add(detected_name)
        return sorted(names, key=str.lower)

    def _snippet_position_map(self) -> Dict[str, Tuple[int, int, int, int]]:
        positions: Dict[str, Tuple[int, int, int, int]] = {}
        path = self._snippet_metadata_path()
        if not path or not path.exists():
            return positions
        try:
            entries = self._load_snippet_entries(path)
        except OSError:
            entries = []
        for entry in entries:
            name = entry.get("export_name") or entry.get("name")
            rect = entry.get("rect") or {}
            x = rect.get("x")
            y = rect.get("y")
            width = rect.get("width")
            height = rect.get("height")
            if (
                name
                and isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and isinstance(width, (int, float))
                and isinstance(height, (int, float))
            ):
                positions[str(name)] = (int(x), int(y), int(width), int(height))
        for match in self._snippet_matches:
            name = match.get("name")
            x = match.get("x")
            y = match.get("y")
            w = match.get("w")
            h = match.get("h")
            if (
                name
                and isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and isinstance(w, (int, float))
                and isinstance(h, (int, float))
            ):
                positions[str(name)] = (int(x), int(y), int(w), int(h))
        return positions

    def _snippet_swipe_templates(self) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
        cache = getattr(self, "_swipe_template_cache", None)
        if cache is not None:
            return dict(cache)

        templates: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
        samples_path = Path("data") / "training" / "training_samples.jsonl"
        if not samples_path.exists():
            self._swipe_template_cache = templates
            return templates

        sums: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}

        try:
            with samples_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    action = entry.get("action") or {}
                    if action.get("type") != "swipe":
                        continue
                    target = action.get("target_snippet")
                    start = action.get("start")
                    end = action.get("end")
                    if (
                        not target
                        or not isinstance(start, list)
                        or not isinstance(end, list)
                        or len(start) != 2
                        or len(end) != 2
                    ):
                        continue
                    sx, sy = start
                    ex, ey = end
                    if not all(isinstance(v, (int, float)) for v in (sx, sy, ex, ey)):
                        continue
                    bucket = sums.setdefault(
                        target,
                        {"sx": 0.0, "sy": 0.0, "ex": 0.0, "ey": 0.0},
                    )
                    bucket["sx"] += float(sx)
                    bucket["sy"] += float(sy)
                    bucket["ex"] += float(ex)
                    bucket["ey"] += float(ey)
                    counts[target] = counts.get(target, 0) + 1
        except OSError:
            self._swipe_template_cache = templates
            return templates

        for name, bucket in sums.items():
            count = counts.get(name, 0)
            if count <= 0:
                continue
            start = (int(round(bucket["sx"] / count)), int(round(bucket["sy"] / count)))
            end = (int(round(bucket["ex"] / count)), int(round(bucket["ey"] / count)))
            templates[name] = (start, end)

        self._swipe_template_cache = dict(templates)
        return dict(templates)

    def _on_reload_rules_clicked(self) -> None:
        try:
            self._rule_engine.reload()
            self._rule_summary_label.setText("Rules reloaded from disk.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to reload rules: {}", exc)
            QMessageBox.warning(self, "Rule Engine", f"Unable to reload rules:\n{exc}")
        self._update_rule_preview()

    def _on_edit_rules_clicked(self) -> None:
        snippet_names = self._known_snippet_names()
        snippet_map = self._snippet_position_map()
        snippet_swipes = self._snippet_swipe_templates()
        dialog = RuleEditorDialog(
            self._scenario_registry,
            snippet_names=snippet_names,
            snippet_positions=snippet_map,
            snippet_swipes=snippet_swipes,
            snapshot_path=self._rule_snapshot_path,
            parent=self,
        )
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            self._rule_engine.reload()
            self._update_rule_preview()

    def _on_test_rules_clicked(self) -> None:
        try:
            snapshot = self._build_rule_snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to build rule snapshot: {}", exc)
            QMessageBox.warning(self, "Rule Test", f"Unable to build snapshot:\n{exc}")
            return

        has_data = (
            snapshot.snippets
            or snapshot.state_values
            or snapshot.player_state
            or snapshot.scenario_current
            or snapshot.scenario_next
        )
        if not has_data:
            QMessageBox.information(
                self,
                "Rule Test",
                "Capture a screenshot or OCR values first, then try again.",
            )
            return

        try:
            match = self._rule_engine.evaluate(snapshot, respect_cooldowns=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Rule evaluation failed: {}", exc)
            QMessageBox.warning(self, "Rule Test", f"Rule evaluation failed:\n{exc}")
            return

        if match is None:
            QMessageBox.information(
                self,
                "Rule Test",
                "No rule matched the latest snapshot.",
            )
            return

        action_text = self._rule_engine.describe_action(match.action)
        QMessageBox.information(
            self,
            "Rule Test",
            f"Matched rule '{match.rule.name}'\nAction: {action_text}",
        )

    def _on_run_rules_clicked(self) -> None:
        if not self._current_serial:
            QMessageBox.information(
                self,
                "Rule Action",
                "Connect a device before running actions.",
            )
            return
        if not self._device_manager:
            QMessageBox.warning(
                self,
                "Rule Action",
                "Device manager unavailable.",
            )
            return

        try:
            snapshot = self._build_rule_snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to build rule snapshot: {}", exc)
            QMessageBox.warning(self, "Rule Action", f"Unable to build snapshot:\n{exc}")
            return

        has_data = (
            snapshot.snippets
            or snapshot.state_values
            or snapshot.player_state
            or snapshot.scenario_current
            or snapshot.scenario_next
        )
        if not has_data:
            QMessageBox.information(
                self,
                "Rule Action",
                "Capture a screenshot or OCR values first, then try again.",
            )
            return

        try:
            match = self._rule_engine.evaluate(snapshot, respect_cooldowns=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Rule evaluation failed: {}", exc)
            QMessageBox.warning(self, "Rule Action", f"Rule evaluation failed:\n{exc}")
            return

        if match is None:
            QMessageBox.information(
                self,
                "Rule Action",
                "No rule matched the latest snapshot.",
            )
            return

        action = match.action
        action_type = action.type
        params = action.params
        summary = self._rule_engine.describe_action(action)

        try:
            if action_type == "tap":
                coords = params.get("coordinates")
                if not coords:
                    raise ValueError("Tap action missing coordinates")
                x, y = coords
                self._device_manager.send_tap(self._current_serial, int(x), int(y))
            elif action_type == "swipe":
                start = params.get("start") or params.get("coordinates")
                end = params.get("end")
                if not start or not end:
                    raise ValueError("Swipe action missing start/end coordinates")
                duration = int(params.get("duration_ms", 300))
                self._device_manager.send_swipe(
                    self._current_serial,
                    int(start[0]),
                    int(start[1]),
                    int(end[0]),
                    int(end[1]),
                    duration,
                )
            elif action_type == "scenario":
                scenario_name = params.get("name")
                if not scenario_name:
                    raise ValueError("Scenario action missing name")
                QMessageBox.information(
                    self,
                    "Rule Action",
                    f"Scenario '{scenario_name}' selected. Apply it manually if needed.",
                )
                return
            else:
                QMessageBox.information(
                    self,
                    "Rule Action",
                    f"Action type '{action_type}' not supported for direct execution.",
                )
                return
        except Exception as exc:  # noqa: BLE001
            logger.error("Rule action execution failed: {}", exc)
            QMessageBox.warning(self, "Rule Action", f"Failed to execute action:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Rule Action",
            f"Executed: {summary}",
        )

    def _build_rule_snapshot(self) -> RuleSnapshot:
        snippets: Dict[str, SnippetObservation] = {}
        for name, detected in self._last_detection_outcomes.items():
            score = self._last_detection_scores.get(name)
            snippets[name] = SnippetObservation(detected=bool(detected), score=score)

        try:
            state_values = self.get_state_board_values() if self._state_board_captured else {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Falling back to last captured state board snapshot: {}", exc)
            state_values = self._state_board_snapshot()

        metadata = {
            "auto_target": self._auto_target_guess,
            "manual_target": self._current_action_target,
        }

        return RuleSnapshot(
            snippets=snippets,
            state_values=state_values,
            scenario_current=self._current_scenario,
            scenario_next=self._next_scenario,
            player_state=self._current_player_state,
            metadata=metadata,
        )

    def _update_rule_preview(self) -> None:
        snapshot = self._build_rule_snapshot()
        self._last_rule_snapshot = snapshot
        self._rule_action_label.setStyleSheet("")

        has_snippets = bool(snapshot.snippets)
        has_state = bool(snapshot.state_values)
        if not (has_snippets or has_state or snapshot.player_state or snapshot.scenario_current or snapshot.scenario_next):
            self._rule_summary_label.setText("Capture snippets or OCR values to preview matching rules.")
            self._rule_action_label.clear()
            self._persist_rule_snapshot(snapshot)
            return

        try:
            match = self._rule_engine.evaluate(snapshot, respect_cooldowns=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Rule evaluation failed: {}", exc)
            self._rule_summary_label.setText(f"Rule evaluation failed: {exc}")
            self._rule_action_label.setStyleSheet("color: #c62828;")
            self._rule_action_label.setText("")
            self._persist_rule_snapshot(snapshot)
            return

        if match is None:
            self._rule_summary_label.setText("No rules matched the current snapshot.")
            self._rule_action_label.setStyleSheet("color: #555555;")
            self._rule_action_label.setText("")
        else:
            description = match.rule.description or ""
            detail = f"Matched rule: {match.rule.name}"
            if description:
                detail += f" — {description}"
            self._rule_summary_label.setText(detail)
            action_text = self._rule_engine.describe_action(match.action)
            self._rule_action_label.setStyleSheet("color: #1b5e20; font-weight: 600;")
            self._rule_action_label.setText(action_text)

        self._persist_rule_snapshot(snapshot)

    def _persist_rule_snapshot(self, snapshot: RuleSnapshot) -> None:
        payload = self._serialize_rule_snapshot(snapshot)
        try:
            self._rule_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self._rule_snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            logger.debug("Unable to write rule snapshot to {}: {}", self._rule_snapshot_path, exc)

    @staticmethod
    def _serialize_rule_snapshot(snapshot: RuleSnapshot) -> Dict[str, Any]:
        return {
            "snippets": {
                name: {"detected": obs.detected, "score": obs.score}
                for name, obs in snapshot.snippets.items()
            },
            "state_values": dict(snapshot.state_values),
            "scenario_current": snapshot.scenario_current,
            "scenario_next": snapshot.scenario_next,
            "player_state": snapshot.player_state,
            "metadata": dict(snapshot.metadata),
        }

    def _update_capture_state(self) -> None:
        has_device = self._current_serial is not None
        has_game = self._current_game is not None
        ready_for_capture = has_device and has_game
        can_capture = ready_for_capture and not self._pending_capture

        self._capture_button.setEnabled(can_capture and not self._live_toggle.isChecked())
        self._live_toggle.setEnabled(ready_for_capture)

        if self._live_toggle.isChecked() and not ready_for_capture:
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
        self._update_rule_preview()

    def _on_next_scenario_changed(self, index: int) -> None:
        data = self._scenario_combo_next.itemData(index)
        scenario = data if isinstance(data, str) and data.strip() else None
        self._next_scenario = scenario
        self._store_scenario_history()
        self._update_capture_state()
        self._update_rule_preview()

    def _on_manage_scenarios_clicked(self) -> None:
        self.scenario_manage_requested.emit()

    def _on_scenarios_changed(self, _: List[str]) -> None:
        self._reload_scenarios()

    def _on_player_states_changed(self, states: List[str]) -> None:
        self._player_state_presets = list(states)
        current_selection = self._current_player_state
        self._updating_player_state = True
        self._player_state_combo.blockSignals(True)
        try:
            existing_text = self._player_state_combo.currentText().strip()
            self._player_state_combo.clear()
            self._player_state_combo.addItem("No player state")
            for state in self._player_state_presets:
                self._player_state_combo.addItem(state)
            target = current_selection or existing_text
            if target and target in self._player_state_presets:
                index = self._player_state_combo.findText(target, Qt.MatchFlag.MatchExactly)
                if index != -1:
                    self._player_state_combo.setCurrentIndex(index)
                    self._current_player_state = target
                else:
                    self._player_state_combo.setCurrentIndex(0)
                    self._current_player_state = None
            else:
                self._player_state_combo.setCurrentIndex(0)
                if current_selection:
                    self._current_player_state = None
        finally:
            self._player_state_combo.blockSignals(False)
            self._updating_player_state = False
        self._store_player_state_history()

    def _on_player_state_changed(self, text: str) -> None:
        if self._updating_player_state:
            return
        value = text.strip()
        if not value or value.lower() == "no player state":
            self._current_player_state = None
        else:
            if value not in self._player_state_presets:
                self._player_state_registry.add_state(value)
            self._current_player_state = value
        self._store_player_state_history()
        self._update_rule_preview()

    def _on_manage_player_states_clicked(self) -> None:
        dialog = PlayerStateDialog(self._player_state_registry, self)
        dialog.exec()

    def _on_target_snippet_changed(self, text: str) -> None:
        if self._updating_target_combo:
            return
        trimmed = text.strip()
        if not trimmed or trimmed.lower() in {"auto (nearest)", "auto"}:
            self._current_action_target = None
        else:
            self._current_action_target = trimmed
            self._auto_target_guess = None
        self._refresh_detection_display()

    def _log_training_sample(self, action: ActionRecord) -> None:
        if not self._last_screenshot_path:
            logger.warning("Skipping training sample; no screenshot has been captured yet.")
            return

        self._annotate_action_with_snippet(action)

        sample = TrainingSample(
            screenshot_path=str(self._last_screenshot_path),
            game=self._current_game,
            scenario_current=self._current_scenario,
            scenario_next=self._next_scenario,
            player_state=self._current_player_state,
            state_board_values=self.get_state_board_values(),
            action=action,
            timestamp=TrainingSample.timestamp_now(),
        )
        self._sample_logger.append(sample)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if (
            hasattr(self, "_preview_scroll")
            and obj is self._preview_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._refresh_preview_display()
        return super().eventFilter(obj, event)

    def _log_idle_sample(self) -> None:
        self._log_training_sample(ActionRecord(type="idle"))

    def _annotate_action_with_snippet(self, action: ActionRecord) -> None:
        if action.type not in {"tap", "swipe"}:
            self._set_auto_target_guess(None)
            action.target_snippet = None
            return

        manual_target = self._current_action_target
        if manual_target and not self._snippet_name_detected(manual_target):
            manual_target = None

        if manual_target:
            action.target_snippet = manual_target
            self._set_auto_target_guess(None)
            return

        point: Optional[Tuple[int, int]] = None
        if action.type == "tap":
            point = action.position
        elif action.type == "swipe":
            point = action.start

        guess = self._guess_snippet_from_point(point) if point else None
        self._set_auto_target_guess(guess)
        action.target_snippet = guess if self._snippet_name_detected(guess) else None

    def _guess_snippet_from_point(
        self,
        point: Tuple[int, int],
    ) -> Optional[str]:
        if not self._snippet_matches:
            return None
        px, py = point
        best_name = None
        best_distance = float("inf")
        for match in self._snippet_matches:
            name = match.get("name")
            x = int(match.get("x", 0))
            y = int(match.get("y", 0))
            w = int(match.get("w", 0))
            h = int(match.get("h", 0))
            if w <= 0 or h <= 0:
                continue
            within = (x <= px <= x + w) and (y <= py <= y + h)
            if within:
                distance = 0.0
            else:
                cx = min(max(px, x), x + w)
                cy = min(max(py, y), y + h)
                dx = px - cx
                dy = py - cy
                distance = math.hypot(dx, dy)
            if distance < best_distance or (math.isclose(distance, best_distance) and name):
                best_distance = distance
                best_name = name
        return best_name

    def _format_ratio_token(self, token: str) -> str:
        normalized = token.strip()
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1]
        numerator, denominator = normalized.split("/", 1)
        numerator = numerator.strip()
        denominator = denominator.strip()
        try:
            num_val = float(numerator.replace(",", ""))
            den_val = float(denominator.replace(",", ""))
            if abs(den_val) > 1e-9:
                return f"{num_val / den_val:.4f}"
        except ValueError:
            pass
        num_text = self._format_integer_string(numerator)
        den_text = self._format_integer_string(denominator)
        return f"{num_text}/{den_text}"

    def _refresh_preview_display(self) -> None:
        if not self._last_raw_pixmap:
            return
        target_size = self._preview_scroll.viewport().size()
        target_width = max(1, target_size.width())
        target_height = max(1, target_size.height())
        if target_width == 1 and target_height == 1:
            target_width, target_height = 960, 540
        scaled = self._last_raw_pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._last_scaled_size = (scaled.width(), scaled.height())
        if self._snippet_matches:
            draw_pixmap = QPixmap(scaled)
            painter = QPainter(draw_pixmap)
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(3)
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(max(8, font.pointSize()))
            painter.setFont(font)
            orig_w = self._last_raw_pixmap.width()
            orig_h = self._last_raw_pixmap.height()
            scale_x = scaled.width() / orig_w if orig_w else 1.0
            scale_y = scaled.height() / orig_h if orig_h else 1.0
            for match in self._snippet_matches:
                rx = int(match["x"] * scale_x)
                ry = int(match["y"] * scale_y)
                rw = int(match["w"] * scale_x)
                rh = int(match["h"] * scale_y)
                painter.drawRect(rx, ry, rw, rh)
                label = match.get("name") or "snippet"
                method_label = match.get("method")
                if method_label:
                    label = f"{label} [{method_label}]"
                painter.drawText(rx + 4, ry + 16, f"{label}")
            painter.end()
            self._preview_label.setPixmap(draw_pixmap)
        else:
            self._preview_label.setPixmap(scaled)
        self._preview_label.resize(scaled.size())

    def _refresh_gpu_support(self) -> None:
        """Update GPU acceleration status display."""
        if hasattr(self, '_gpu_status_value'):
            gpu_info = self._gpu_processor.get_processing_stats()
            status_parts = []
            
            if gpu_info["cuda_available"]:
                status_parts.append("CUDA")
            if gpu_info["opencv_cuda_available"]:
                status_parts.append("OpenCV-CUDA")
            if gpu_info["cupy_available"]:
                status_parts.append("CuPy")
                
            if status_parts:
                gpu_name = gpu_info.get("gpu_name", "Unknown")
                memory_gb = gpu_info.get("gpu_memory_gb", 0)
                status_text = f"{', '.join(status_parts)} ({gpu_name}, {memory_gb:.1f}GB)"
            else:
                status_text = "CPU Only"
                
            self._gpu_status_value.setText(status_text)
        
    def _original_refresh_gpu_support(self) -> None:
        _lazy_imports()
        info_parts: List[str] = []
        available = False

        if torch is None:
            info_parts.append("PyTorch not installed")
        else:
            try:
                available = torch.cuda.is_available()
                info_parts.append(f"PyTorch {torch.__version__}")
                info_parts.append(f"CUDA available: {available}")
                info_parts.append(f"Python: {sys.executable}")
                cuda_version = getattr(torch.version, "cuda", None)
                if cuda_version:
                    info_parts.append(f"CUDA build: {cuda_version}")
                if available:
                    try:
                        info_parts.append(f"Device: {torch.cuda.get_device_name(0)}")
                    except Exception as exc:  # noqa: BLE001
                        info_parts.append(f"Device query failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                info_parts.append(f"cuda detection error: {exc}")

        self._gpu_supported = available
        display_text = "\n".join(info_parts) if info_parts else "No GPU information available."
        self._gpu_info_label.setText(display_text)

        self._gpu_checkbox.setToolTip(
            "CUDA GPU detected." if available else "CUDA GPU not detected in this environment."
        )
        self._gpu_checkbox.setEnabled(available)
        if available:
            self._gpu_checkbox.blockSignals(True)
            self._gpu_checkbox.setChecked(self._use_gpu)
            self._gpu_checkbox.blockSignals(False)

    def _training_samples_path(self) -> Path:
        return Path("data/training/training_samples.jsonl")

    def _on_reset_dataset_clicked(self) -> None:
        target_text = self._dataset_path_edit.text().strip()
        if not target_text:
            QMessageBox.warning(self, "Reset Dataset", "Please specify the dataset directory first.")
            return

        target_path = Path(target_text).expanduser().resolve()
        samples_path = self._training_samples_path()

        message = (
            "This will permanently delete all captured screenshots under:\n"
            f"{target_path}\n\n"
            "and clear the training_samples.jsonl log.\n\n"
            "Are you sure you want to continue?"
        )
        confirm = QMessageBox.question(
            self,
            "Reset Dataset",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        errors: List[str] = []

        if target_path.exists():
            try:
                shutil.rmtree(target_path)
            except OSError as exc:
                errors.append(f"Failed to remove dataset directory: {exc}")

        if samples_path.exists():
            try:
                samples_path.unlink()
            except OSError as exc:
                errors.append(f"Failed to remove {samples_path}: {exc}")

        try:
            target_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"Unable to recreate dataset directory: {exc}")

        if errors:
            QMessageBox.critical(
                self,
                "Reset Dataset",
                "\n".join(errors),
            )
            return

        self._sample_logger = TrainingSampleLogger(samples_path)
        self._last_screenshot_path = None
        self._status_label.setText("Dataset cleared. Ready to capture new screenshots.")
        self._training_status_label.setText("Training idle.")
        self._snippet_mask_cache.clear()

    def _on_start_training_clicked(self) -> None:
        if self._training_process and self._training_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Training", "Training is already running.")
            return

        samples_path = self._training_samples_path()
        if not samples_path.exists() or samples_path.stat().st_size == 0:
            QMessageBox.warning(
                self,
                "Training",
                "No training samples found. Capture gameplay actions so that training_samples.jsonl is populated.",
            )
            return

        script_path = Path("training/train_policy.py")
        if not script_path.exists():
            QMessageBox.critical(
                self,
                "Training",
                f"Training script not found at {script_path}.",
            )
            return

        output_dir = Path("models/policy")
        device_arg = "cuda" if self._use_gpu else "cpu"
        args = [
            str(script_path),
            "--samples",
            str(samples_path),
            "--output-dir",
            str(output_dir),
            "--device",
            device_arg,
        ]

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_training_output_ready)
        process.finished.connect(self._on_training_finished)
        process.start(sys.executable, args)

        if not process.waitForStarted(5000):
            QMessageBox.critical(
                self,
                "Training",
                "Failed to start training process.",
            )
            process.deleteLater()
            return

        self._training_process = process
        self._training_status_label.setText("Training started…")
        self._update_training_controls()

    def _on_stop_training_clicked(self) -> None:
        if not self._training_process or self._training_process.state() == QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Training", "No training process is running.")
            return
        self._training_status_label.setText("Stopping training…")
        self._training_process.terminate()

    def _on_gpu_checkbox_changed(self, checked: bool) -> None:
        self._use_gpu = bool(checked)
        if self._use_gpu and not self._gpu_supported:
            QMessageBox.information(
                self,
                "GPU Forcing",
                "PyTorch did not detect a CUDA GPU. We'll attempt to use GPU anyway; if it fails the app will fall back to CPU.",
            )
        self._ocr_reader = None
        if self._last_screenshot_bytes:
            self._update_state_board_values(self._last_screenshot_bytes)
        self._refresh_gpu_support()
        if self._policy_runner and self._policy_runner.is_ready:
            self._policy_runner.to(self._policy_device())
            if self._policy_model_path:
                try:
                    display_path = self._policy_model_path.relative_to(Path.cwd())
                except ValueError:
                    display_path = self._policy_model_path
                self._policy_model_label.setText(f"{display_path} (device: {self._policy_device()})")
        self._update_policy_controls()

    def _on_training_output_ready(self) -> None:
        if not self._training_process:
            return
        text = bytes(self._training_process.readAllStandardOutput()).decode(errors="ignore")
        if not text:
            return
        last_line = text.strip().splitlines()[-1]
        self._training_status_label.setText(f"Training: {last_line}")

    def _on_training_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if exit_status == QProcess.ExitStatus.NormalExit:
            if exit_code == 0:
                self._training_status_label.setText("Training completed successfully.")
            else:
                self._training_status_label.setText(
                    f"Training finished with exit code {exit_code} (status {exit_status.name})."
                )
        else:
            self._training_status_label.setText(
                f"Training failed with exit code {exit_code} (status {exit_status.name})."
            )
        if self._training_process:
            self._training_process.deleteLater()
        self._training_process = None
        self._update_training_controls()

    def _update_training_controls(self) -> None:
        process = getattr(self, "_training_process", None)
        running = process is not None and process.state() != QProcess.ProcessState.NotRunning
        self._start_training_button.setEnabled(not running)
        self._stop_training_button.setEnabled(running)

    # Policy control ----------------------------------------------------------

    def _policy_device(self) -> str:
        if self._use_gpu and torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_policy_runner(self) -> PolicyRunner:
        if self._policy_runner is None:
            self._policy_runner = PolicyRunner(device=self._policy_device())
        else:
            self._policy_runner.to(self._policy_device())
        return self._policy_runner

    def _set_policy_status(self, message: str) -> None:
        if hasattr(self, "_policy_status_label"):
            self._policy_status_label.setText(message)

    def _update_policy_controls(self) -> None:
        if not hasattr(self, "_policy_load_button"):
            return
        has_torch = torch is not None
        has_model = bool(self._policy_runner and self._policy_runner.is_ready)
        self._policy_load_button.setEnabled(has_torch)
        self._policy_unload_button.setEnabled(has_model)
        allow_toggle = has_model and self._current_serial is not None
        self._policy_toggle_button.setEnabled(allow_toggle)
        if not allow_toggle and self._policy_toggle_button.isChecked():
            self._policy_toggle_button.blockSignals(True)
            self._policy_toggle_button.setChecked(False)
            self._policy_toggle_button.blockSignals(False)
            self._policy_enabled = False
        self._policy_toggle_button.setText(
            "Disable Policy Control" if self._policy_enabled else "Enable Policy Control"
        )

    def _on_load_policy_model_clicked(self) -> None:
        if torch is None:
            QMessageBox.warning(self, "Policy Model", "PyTorch is required to load policy checkpoints.")
            return
        start_dir = (
            str(self._policy_model_path.parent)
            if self._policy_model_path
            else str(Path("models/policy").resolve())
        )
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Policy Checkpoint",
            start_dir,
            "PyTorch models (*.pt *.pth);;All files (*)",
        )
        if not filename:
            return
        checkpoint_path = Path(filename)
        metadata_path = checkpoint_path.with_name("policy_metadata.json")
        if not metadata_path.exists():
            meta_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select Policy Metadata",
                str(checkpoint_path.parent),
                "JSON files (*.json);;All files (*)",
            )
            if not meta_name:
                QMessageBox.warning(
                    self,
                    "Policy Model",
                    "Metadata file is required to load the policy.",
                )
                return
            metadata_path = Path(meta_name)
        try:
            runner = self._ensure_policy_runner()
            runner.load(checkpoint_path, metadata_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load policy model: {}", exc)
            QMessageBox.critical(self, "Policy Model", f"Failed to load policy model:\n{exc}")
            return

        self._policy_model_path = checkpoint_path
        self._policy_metadata_path = metadata_path
        try:
            display_path = checkpoint_path.relative_to(Path.cwd())
        except ValueError:
            display_path = checkpoint_path
        self._policy_model_label.setText(f"{display_path} (device: {self._policy_device()})")
        self._policy_enabled = False
        self._policy_toggle_button.blockSignals(True)
        self._policy_toggle_button.setChecked(False)
        self._policy_toggle_button.blockSignals(False)
        self._set_policy_status("Policy model loaded. Enable control to let it act.")
        self._update_policy_controls()

    def _on_unload_policy_model_clicked(self) -> None:
        self._policy_runner = None
        self._policy_model_path = None
        self._policy_metadata_path = None
        self._last_policy_prediction = None
        self._policy_enabled = False
        self._policy_model_label.setText("No checkpoint loaded.")
        self._policy_toggle_button.blockSignals(True)
        self._policy_toggle_button.setChecked(False)
        self._policy_toggle_button.blockSignals(False)
        self._set_policy_status("Policy model unloaded.")
        self._update_policy_controls()

    def _on_policy_toggle_changed(self, enabled: bool) -> None:
        if not (self._policy_runner and self._policy_runner.is_ready):
            QMessageBox.information(self, "Policy Control", "Load a policy checkpoint first.")
            self._policy_toggle_button.blockSignals(True)
            self._policy_toggle_button.setChecked(False)
            self._policy_toggle_button.blockSignals(False)
            return
        if enabled and not self._current_serial:
            QMessageBox.information(self, "Policy Control", "Select a connected device first.")
            self._policy_toggle_button.blockSignals(True)
            self._policy_toggle_button.setChecked(False)
            self._policy_toggle_button.blockSignals(False)
            return
        self._policy_enabled = bool(enabled)
        self._policy_toggle_button.setText(
            "Disable Policy Control" if self._policy_enabled else "Enable Policy Control"
        )
        self._set_policy_status("Policy control enabled." if self._policy_enabled else "Policy control disabled.")

    def _state_board_snapshot(self) -> Dict[str, str]:
        if not self._state_board_captured:
            return {}
        return {name: ", ".join(values) for name, values in self._state_board_captured.items()}

    def _maybe_run_policy_control(self, png_bytes: bytes) -> None:
        if (
            not self._policy_enabled
            or self._policy_runner is None
            or not self._policy_runner.is_ready
            or torch is None
        ):
            return
        if self._current_serial is None:
            self._set_policy_status("Policy disabled: no device selected.")
            return
        if self._awaiting_tap or self._awaiting_swipe:
            self._set_policy_status("Waiting for previous action to complete.")
            return

        try:
            state_values = self.get_state_board_values() if self._state_board_captured else {}
        except Exception:  # noqa: BLE001
            state_values = self._state_board_snapshot()

        try:
            prediction = self._policy_runner.predict(
                png_bytes=png_bytes,
                state_board=state_values,
                scenario_current=self._current_scenario,
                scenario_next=self._next_scenario,
                player_state=self._current_player_state,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Policy inference failed: {}", exc)
            self._set_policy_status(f"Policy error: {exc}")
            self._policy_toggle_button.blockSignals(True)
            self._policy_toggle_button.setChecked(False)
            self._policy_toggle_button.blockSignals(False)
            self._policy_enabled = False
            return

        if prediction is None:
            self._set_policy_status("Policy not ready.")
            return

        self._last_policy_prediction = prediction

        confidence = prediction.confidence
        threshold = float(self._policy_confidence_spin.value())
        if confidence < threshold:
            self._set_policy_status(f"Low confidence ({confidence:.2f} < {threshold:.2f}); skipping action.")
            return

        action = prediction.action
        coords = prediction.coordinates
        width, height = prediction.image_size

        if action == "idle":
            self._set_policy_status(f"Policy chose idle ({confidence:.2f}).")
            return

        def clamp01(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        if action == "tap":
            x = int(round(clamp01(coords[0]) * width))
            y = int(round(clamp01(coords[1]) * height))
            if self._snippet_matches:
                guess = self._guess_snippet_from_point((x, y))
                if guess:
                    self._set_auto_target_guess(guess)
            self._awaiting_tap = True
            self._click_position_label.setText(f"Policy tap raw=({x}, {y})")
            self._status_label.setText(f"Policy sending tap to ({x}, {y}) on {self._current_device_label}…")
            self._device_manager.send_tap(self._current_serial, x, y)
            self._set_policy_status(f"Sent tap ({x}, {y}) (confidence {confidence:.2f}).")
        elif action == "swipe":
            x1 = int(round(clamp01(coords[0]) * width))
            y1 = int(round(clamp01(coords[1]) * height))
            x2 = int(round(clamp01(coords[2]) * width))
            y2 = int(round(clamp01(coords[3]) * height))
            duration_ms = max(50, int(round(max(0.0, coords[4]) * 1000.0)))
            if self._snippet_matches:
                guess = self._guess_snippet_from_point((x1, y1))
                if guess:
                    self._set_auto_target_guess(guess)
            self._awaiting_swipe = True
            self._click_position_label.setText(
                f"Policy swipe raw=({x1}, {y1})→({x2}, {y2}) {duration_ms}ms"
            )
            self._status_label.setText(
                f"Policy sending swipe ({x1}, {y1})→({x2}, {y2}) on {self._current_device_label}…"
            )
            self._device_manager.send_swipe(self._current_serial, x1, y1, x2, y2, duration_ms)
            self._set_policy_status(
                f"Sent swipe ({x1}, {y1})→({x2}, {y2}) (confidence {confidence:.2f})."
            )
        else:
            self._set_policy_status(f"Policy action '{action}' is not supported.")

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

    def _store_player_state_history(self) -> None:
        key = self._current_game or self._GLOBAL_SCENARIO_KEY
        self._player_state_history[key] = self._current_player_state

    def _restore_player_state(self) -> None:
        key = self._current_game or self._GLOBAL_SCENARIO_KEY
        saved = self._player_state_history.get(key)
        self._set_player_state(saved)

    def _set_player_state(self, state: Optional[str]) -> None:
        self._updating_player_state = True
        block_restored = False
        try:
            if state and state not in self._player_state_presets:
                self._player_state_registry.add_state(state)
            self._player_state_combo.blockSignals(True)
            if state:
                index = self._player_state_combo.findText(state, Qt.MatchFlag.MatchExactly)
                if index != -1:
                    self._player_state_combo.setCurrentIndex(index)
                else:
                    self._player_state_combo.setCurrentIndex(0)
                self._current_player_state = state if index != -1 else None
            else:
                self._player_state_combo.setCurrentIndex(0)
                self._current_player_state = None
            self._player_state_combo.blockSignals(False)
            block_restored = True
        finally:
            if not block_restored:
                self._player_state_combo.blockSignals(False)
            self._updating_player_state = False
        self._store_player_state_history()

    def _snippet_metadata_path(self) -> Optional[Path]:
        if not self._current_game:
            return None
        slug = self._sanitize_game_slug(self._current_game)
        return Path("data") / "layout_snippets" / slug / "metadata.jsonl"

    @staticmethod
    def _sanitize_game_slug(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
        slug = slug.strip("_")
        return slug or "unspecified"

    @staticmethod
    def _load_snippet_entries(path: Path) -> List[dict]:
        entries: List[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "search" not in entry:
                    entry["search"] = {"mode": "global"}
                entries.append(entry)
        return entries

    def _match_exact_snippet(
        self,
        roi_gray: np.ndarray,
        snippet_gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        threshold: float,
        mask: Optional[np.ndarray] = None,
        roi_color: Optional[np.ndarray] = None,
        snippet_color: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        sh, sw = snippet_gray.shape[:2]
        roi_section = roi_gray[:sh, :sw]
        if roi_section.shape[0] != sh or roi_section.shape[1] != sw:
            return None
        effective_mask: Optional[np.ndarray] = None
        if mask is not None and mask.size:
            if mask.shape[:2] != (sh, sw):
                effective_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            else:
                effective_mask = mask
        if (
            effective_mask is not None
            and roi_color is not None
            and roi_color.shape[0] >= sh
            and roi_color.shape[1] >= sw
            and snippet_color is not None
            and snippet_color.shape[0] >= sh
            and snippet_color.shape[1] >= sw
        ):
            mask_indices = effective_mask[:sh, :sw] > 0
            if not np.any(mask_indices):
                return None
            roi_color_section = roi_color[:sh, :sw]
            snippet_color_section = snippet_color[:sh, :sw]
            diff_color = cv2.absdiff(roi_color_section, snippet_color_section)
            diff_gray = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
            score = 1.0 - float(diff_gray[mask_indices].mean()) / 255.0
        else:
            diff = cv2.absdiff(roi_section, snippet_gray)
            score = 1.0 - float(diff.mean()) / 255.0
        if score >= threshold:
            return {"x": origin_x, "y": origin_y, "w": sw, "h": sh, "score": score}
        return None

    def _match_template(
        self,
        roi_gray: np.ndarray,
        snippet_gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        threshold: float,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        sh, sw = snippet_gray.shape[:2]
        if roi_gray.shape[0] < sh or roi_gray.shape[1] < sw:
            return None
            
        # Try GPU acceleration first
        if self._gpu_enabled:
            result = self._gpu_processor.template_match_gpu(roi_gray, snippet_gray, threshold, mask)
            if result is not None:
                # Adjust coordinates to global position
                result["x"] += origin_x
                result["y"] += origin_y
                return result
        
        # Fallback to original CPU implementation
        method = cv2.TM_CCORR_NORMED if mask is not None else cv2.TM_CCOEFF_NORMED
        try:
            if mask is None and self._use_gpu and self._cuda_available:
                roi_gpu = cv2.cuda_GpuMat()
                tmpl_gpu = cv2.cuda_GpuMat()
                roi_gpu.upload(roi_gray)
                tmpl_gpu.upload(snippet_gray)
                res_gpu = cv2.cuda.matchTemplate(roi_gpu, tmpl_gpu, method)
                result_img = res_gpu.download()
            else:
                raise AttributeError
        except Exception:
            if mask is not None:
                result_img = cv2.matchTemplate(roi_gray, snippet_gray, method, mask=mask)
            else:
                result_img = cv2.matchTemplate(roi_gray, snippet_gray, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(result_img)
        if max_val >= threshold:
            x = origin_x + max_loc[0]
            y = origin_y + max_loc[1]
            return {"x": x, "y": y, "w": sw, "h": sh, "score": float(max_val)}
        return None

    def _match_multiscale_template(
        self,
        roi_gray: np.ndarray,
        snippet_gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        threshold: float,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        # Try GPU-accelerated multiscale matching first
        if self._gpu_enabled:
            result = self._gpu_processor.multiscale_template_match_gpu(
                roi_gray, snippet_gray, threshold, 
                scales=[0.8, 0.9, 1.0, 1.1, 1.2], 
                mask=mask
            )
            if result is not None:
                # Adjust coordinates to global position
                result["x"] += origin_x
                result["y"] += origin_y
                return result
        
        # Fallback to original CPU implementation
        sh, sw = snippet_gray.shape[:2]
        best = None
        best_val = -1.0
        method = cv2.TM_CCORR_NORMED if mask is not None else cv2.TM_CCOEFF_NORMED
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            scaled_w = max(1, int(sw * scale))
            scaled_h = max(1, int(sh * scale))
            if roi_gray.shape[0] < scaled_h or roi_gray.shape[1] < scaled_w:
                continue
            scaled_template = cv2.resize(snippet_gray, (scaled_w, scaled_h))
            scaled_mask = None
            if mask is not None:
                scaled_mask = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                if cv2.countNonZero(scaled_mask) == 0:
                    scaled_mask = None
            try:
                if mask is None and self._use_gpu and self._cuda_available:
                    roi_gpu = cv2.cuda_GpuMat()
                    tmpl_gpu = cv2.cuda_GpuMat()
                    roi_gpu.upload(roi_gray)
                    tmpl_gpu.upload(scaled_template)
                    res_gpu = cv2.cuda.matchTemplate(roi_gpu, tmpl_gpu, method)
                    result = res_gpu.download()
                else:
                    raise AttributeError
            except Exception:
                if scaled_mask is not None:
                    result = cv2.matchTemplate(roi_gray, scaled_template, method, mask=scaled_mask)
                else:
                    result = cv2.matchTemplate(roi_gray, scaled_template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = float(max_val)
                best = (max_loc, scaled_w, scaled_h)
        if best is None or best_val < threshold:
            return None
        max_loc, scaled_w, scaled_h = best
        x = origin_x + max_loc[0]
        y = origin_y + max_loc[1]
        return {"x": x, "y": y, "w": scaled_w, "h": scaled_h, "score": best_val}

    def _match_orb_snippet(
        self,
        roi_gray: np.ndarray,
        snippet_gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        threshold: float,
    ) -> Optional[dict]:
        if self._orb_detector is None or self._bf_matcher is None:
            return None
            
        # Enhance images for better feature detection
        if self._gpu_enabled:
            enhanced_snippet = self._gpu_processor.enhance_for_feature_detection(snippet_gray)
            enhanced_roi = self._gpu_processor.enhance_for_feature_detection(roi_gray)
        else:
            enhanced_snippet = snippet_gray
            enhanced_roi = roi_gray
            
        kp1, des1 = self._orb_detector.detectAndCompute(enhanced_snippet, None)
        if des1 is None or len(kp1) < 4:
            return None
        kp2, des2 = self._orb_detector.detectAndCompute(enhanced_roi, None)
        if des2 is None or len(kp2) < 4:
            return None

        try:
            matches = self._bf_matcher.match(des1, des2)
        except cv2.error:
            return None
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = matches[:40]
        if len(good_matches) < 8:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None
        inliers = int(mask.sum())
        if inliers < 6:
            return None

        sh, sw = snippet_gray.shape[:2]
        corners = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H)
        xs = projected[:, 0, 0]
        ys = projected[:, 0, 1]
        x_min = max(0, min(xs)) + origin_x
        y_min = max(0, min(ys)) + origin_y
        x_max = max(xs) + origin_x
        y_max = max(ys) + origin_y
        w = max(1, int(x_max - x_min))
        h = max(1, int(y_max - y_min))
        score = inliers / max(len(good_matches), 1)
        if score < threshold:
            return None
        return {"x": int(x_min), "y": int(y_min), "w": w, "h": h, "score": float(score)}

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
        self._log_training_sample(ActionRecord(type="tap", position=(x, y)))

    def _on_preview_dragged(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if not self._last_raw_pixmap or not self._last_scaled_size:
            return

        start_mapped = self._map_preview_point(x1, y1)
        end_mapped = self._map_preview_point(x2, y2)
        if not start_mapped or not end_mapped:
            return

        start_adj_x, start_adj_y, start_raw_x, start_raw_y = start_mapped
        end_adj_x, end_adj_y, end_raw_x, end_raw_y = end_mapped

        if not self._current_serial:
            self._status_label.setText(
                "Swipe recorded locally (no device connected)."
            )
            self._log_training_sample(
                ActionRecord(
                    type="swipe",
                    start=(start_raw_x, start_raw_y),
                    end=(end_raw_x, end_raw_y),
                    duration_ms=0,
                )
            )
            return

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
        self._log_training_sample(
            ActionRecord(
                type="swipe",
                start=(x1, y1),
                end=(x2, y2),
                duration_ms=duration_ms,
            )
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
            
            # Preprocess image for better OCR results with GPU acceleration
            if self._gpu_enabled and np_region.size > 0:
                enhanced_region = self._gpu_processor.preprocess_for_ocr(np_region, enhance=True)
            else:
                enhanced_region = np_region
                
            try:
                texts = reader.readtext(
                    enhanced_region,
                    detail=0,
                    allowlist="0123456789.,/()%+-",
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("OCR failed for region {}: {}", region.name, exc)
                continue

            cleaned = [text.strip() for text in texts if text and text.strip()]
            cleaned = self._apply_state_board_format(region, cleaned)
            if cleaned:
                format_hint = (getattr(region, "value_format", None) or "").strip().lower()
                is_ratio_hint = format_hint.startswith("(") and format_hint.endswith(")")
                if len(cleaned) > 1 and is_ratio_hint:
                    quest_labels = [
                        "first_quest",
                        "second_quest",
                        "third_quest",
                        "fourth_quest",
                    ]
                    for idx, value in enumerate(cleaned):
                        label = quest_labels[idx] if idx < len(quest_labels) else f"quest {idx + 1}"
                        results[label].append(value)
                else:
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

        combined = "".join(filtered)
        tokens = combined.split()
        if not tokens:
            tokens = [combined.replace(" ", "")]

        fmt_lower = format_hint.lower()
        if fmt_lower.startswith("(") and fmt_lower.endswith(")"):
            ratio_matches = re.findall(r"\(\s*([0-9.,]+/[0-9.,]+)\s*\)", combined)
            if ratio_matches:
                return [self._format_ratio_token(match) for match in ratio_matches]

        merged = combined.replace(" ", "")
        if not merged:
            return []

        if fmt_lower in {"text", "string"}:
            return [" ".join(filtered)]
        if "/" in fmt_lower:
            ratio_source = merged.replace(" ", "")
            if ratio_source.count("/") == 1:
                numerator, denominator = ratio_source.split("/")
                return [self._format_ratio_token(f"{numerator}/{denominator}")]
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
            decimal_value = self._coerce_number(digits_only)
            if decimal_value is not None:
                return [f"{decimal_value:.0f}"]
            numeric_fallback = "".join(ch for ch in digits_only if ch.isdigit())
            return [numeric_fallback] if numeric_fallback else [formatted_currency]

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
        self._update_rule_preview()

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
        self._update_rule_preview()

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

    def _get_ocr_reader(self) -> Optional[Any]:
        if easyocr is None:
            return None

        if self._ocr_reader is not None:
            return self._ocr_reader

        try:
            # Use GPU manager for optimal EasyOCR configuration
            try:
                self._ocr_reader = self._gpu_manager.create_easyocr_reader(["en"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("EasyOCR GPU init failed: %s", exc)
                # Fallback to direct creation
                try:
                    self._ocr_reader = easyocr.Reader(["en"], gpu=self._use_gpu)
                except Exception as exc2:  # noqa: BLE001
                    logger.warning("EasyOCR CPU fallback failed: %s", exc2)
                    if self._use_gpu:
                        QMessageBox.warning(
                            self,
                            "EasyOCR",
                            "Failed to initialize EasyOCR with GPU. Falling back to CPU.",
                    )
                    self._use_gpu = False
                    self._gpu_checkbox.blockSignals(True)
                    self._gpu_checkbox.setChecked(False)
                    self._gpu_checkbox.blockSignals(False)
                    self._ocr_reader = easyocr.Reader(["en"], gpu=False)
                    self._refresh_gpu_support()
                else:
                    return None
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

