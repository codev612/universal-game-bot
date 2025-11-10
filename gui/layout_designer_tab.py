from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QInputDialog,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from core.device_manager import DeviceManager
from core.layout_registry import LayoutRegistry, RegionConfig
from gui.screenshot_canvas import Region, ScreenshotCanvas
from gui.train_tab import TrainTab


class Category:
    CONTROLS = "Controls"
    STATE_BOARDS = "State Boards"
    DIALOGUES = "Dialogues"
    SNIPPETS = "Snippets"

    ALL = [CONTROLS, STATE_BOARDS, DIALOGUES, SNIPPETS]


STATE_BOARD_VALUE_FORMATS = [
    "text",
    "(xxx/yyy)",
    "xxxx/yyyy",
    "xx%",
    "xxxx",
    "currency(x,yyy,zzz)",
]


@dataclass(slots=True)
class ControlMapping:
    name: str
    rect: Tuple[int, int, int, int]  # x, y, width, height
    category: str = Category.CONTROLS
    notes: str = ""
    value_format: str = ""


class _StateBoardFormatDialog(QDialog):
    def __init__(
        self,
        parent: Optional[QWidget],
        initial: str,
        known_formats: list[str],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Value Format")

        self._combo = QComboBox()
        self._combo.setEditable(True)
        self._combo.addItem("Automatic (no format)", "")
        for entry in known_formats:
            if entry:
                self._combo.addItem(entry, entry)
        if initial:
            if self._combo.findData(initial) == -1:
                self._combo.addItem(initial, initial)
            index = self._combo.findData(initial)
            if index != -1:
                self._combo.setCurrentIndex(index)
            else:
                self._combo.setCurrentIndex(0)
                self._combo.setEditText(initial)
        else:
            self._combo.setCurrentIndex(0)

        form = QFormLayout()
        form.addRow("Format:", self._combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def selected_format(self) -> str:
        return self._combo.currentText().strip()

    @staticmethod
    def get_format(
        parent: QWidget,
        initial: str,
        known_formats: list[str],
    ) -> Optional[str]:
        dialog = _StateBoardFormatDialog(parent, initial, known_formats)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_format()
        return None


class _SnippetPreviewLabel(QLabel):
    colorPicked = pyqtSignal(QColor)

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._original_pixmap = pixmap
        self._display_pixmap = self._prepare_display_pixmap(pixmap)
        if not self._display_pixmap.isNull():
            self.setPixmap(self._display_pixmap)
            self.setFixedSize(self._display_pixmap.size())
        self.setStyleSheet("border: 1px solid #555555;")
        self.setCursor(Qt.CursorShape.CrossCursor)

    @staticmethod
    def _prepare_display_pixmap(pixmap: QPixmap) -> QPixmap:
        if pixmap.isNull():
            return QPixmap()
        max_dim = 256
        if pixmap.width() <= max_dim and pixmap.height() <= max_dim:
            return pixmap
        return pixmap.scaled(
            max_dim,
            max_dim,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if (
            self._original_pixmap is None
            or self._original_pixmap.isNull()
            or self._display_pixmap is None
            or self._display_pixmap.isNull()
        ):
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        display_width = self._display_pixmap.width()
        display_height = self._display_pixmap.height()
        if display_width <= 0 or display_height <= 0:
            return
        pos = event.position()
        x_ratio = self._original_pixmap.width() / display_width
        y_ratio = self._original_pixmap.height() / display_height
        x = int(max(0, min(self._original_pixmap.width() - 1, pos.x() * x_ratio)))
        y = int(max(0, min(self._original_pixmap.height() - 1, pos.y() * y_ratio)))
        qimage = self._original_pixmap.toImage()
        color = QColor(qimage.pixel(x, y))
        self.colorPicked.emit(color)


class _SnippetOptionsDialog(QDialog):
    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        parent: Optional[QWidget] = None,
        default: Optional[dict] = None,
        snippet_pixmap: Optional[QPixmap] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Snippet Options")
        self._rect = rect
        self._color_display: Optional[QLabel] = None

        self._combo = QComboBox()
        self._combo.addItems(["Global", "Exact", "Custom Region"])

        self._x1 = QLineEdit(str(rect[0]))
        self._y1 = QLineEdit(str(rect[1]))
        self._x2 = QLineEdit(str(rect[0] + rect[2]))
        self._y2 = QLineEdit(str(rect[1] + rect[3]))
        self._color_edit = QLineEdit()
        self._color_edit.setPlaceholderText("#rrggbb")
        self._mask_checkbox = QCheckBox("Ignore background using mask")

        form = QFormLayout()
        form.addRow("Search mode:", self._combo)
        form.addRow("X1:", self._x1)
        form.addRow("Y1:", self._y1)
        form.addRow("X2:", self._x2)
        form.addRow("Y2:", self._y2)
        form.addRow("Target color (optional):", self._color_edit)
        form.addRow(self._mask_checkbox)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        if snippet_pixmap and not snippet_pixmap.isNull():
            preview_group = QGroupBox("Snippet Preview")
            preview_group_layout = QVBoxLayout(preview_group)
            preview_group_layout.setContentsMargins(8, 8, 8, 8)
            instruction = QLabel("Click the image to sample a color.")
            instruction.setWordWrap(True)
            preview_group_layout.addWidget(instruction)
            preview_label = _SnippetPreviewLabel(snippet_pixmap, self)
            preview_label.colorPicked.connect(self._on_preview_color_picked)
            preview_group_layout.addWidget(preview_label, alignment=Qt.AlignmentFlag.AlignCenter)
            color_row = QHBoxLayout()
            color_row.addWidget(QLabel("Picked color:"))
            self._color_display = QLabel("None")
            self._color_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._color_display.setMinimumWidth(72)
            self._color_display.setStyleSheet("border: 1px solid #555555; padding: 2px; color: #666666;")
            color_row.addWidget(self._color_display)
            color_row.addStretch()
            preview_group_layout.addLayout(color_row)
            layout.addWidget(preview_group)
        layout.addWidget(self._buttons)

        if default:
            mode = (default.get("mode") or "global").lower()
            if mode == "exact":
                self._combo.setCurrentIndex(1)
            elif mode == "custom":
                self._combo.setCurrentIndex(2)
                region = default.get("region") or {}
                self._x1.setText(str(region.get("x1", rect[0])))
                self._y1.setText(str(region.get("y1", rect[1])))
                self._x2.setText(str(region.get("x2", rect[0] + rect[2])))
                self._y2.setText(str(region.get("y2", rect[1] + rect[3])))
            else:
                self._combo.setCurrentIndex(0)
            if default.get("color"):
                self._color_edit.setText(str(default["color"]))
            if default.get("use_mask"):
                self._mask_checkbox.setChecked(bool(default["use_mask"]))

        self._combo.currentIndexChanged.connect(self._update_enabled_state)
        self._update_enabled_state(self._combo.currentIndex())
        self._color_edit.textChanged.connect(self._on_color_text_changed)
        self._on_color_text_changed(self._color_edit.text())

    def _update_enabled_state(self, index: int) -> None:
        custom = index == 2
        for widget in (self._x1, self._y1, self._x2, self._y2):
            widget.setEnabled(custom)

    def get_options(self) -> Optional[dict]:
        mode = self._combo.currentText().lower()
        color = self._sanitize_color(self._color_edit.text())
        if mode == "global":
            return {"mode": "global", "color": color, "use_mask": self._mask_checkbox.isChecked()}
        if mode == "exact":
            return {
                "mode": "exact",
                "region": {
                    "x1": self._rect[0],
                    "y1": self._rect[1],
                    "x2": self._rect[0] + self._rect[2],
                    "y2": self._rect[1] + self._rect[3],
                },
                "color": color,
                "use_mask": self._mask_checkbox.isChecked(),
            }
        try:
            x1 = int(self._x1.text().strip())
            y1 = int(self._y1.text().strip())
            x2 = int(self._x2.text().strip())
            y2 = int(self._y2.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Snippet Options", "Custom region values must be integers.")
            return None
        if x2 <= x1 or y2 <= y1:
            QMessageBox.warning(self, "Snippet Options", "Custom region coordinates are invalid.")
            return None
        return {
            "mode": "custom",
            "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "color": color,
            "use_mask": self._mask_checkbox.isChecked(),
        }

    @staticmethod
    def exec_options(
        rect: Tuple[int, int, int, int],
        parent: QWidget,
        default: Optional[dict] = None,
        snippet_pixmap: Optional[QPixmap] = None,
    ) -> Optional[dict]:
        dialog = _SnippetOptionsDialog(
            rect=rect,
            parent=parent,
            default=default,
            snippet_pixmap=snippet_pixmap,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_options()
        return None

    @staticmethod
    def _sanitize_color(value: str) -> Optional[str]:
        if not value:
            return None
        text = value.strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) not in (3, 6):
            return None
        try:
            int(text, 16)
        except ValueError:
            return None
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        return "#" + text.lower()

    def _on_preview_color_picked(self, color: QColor) -> None:
        hex_color = f"#{color.red():02x}{color.green():02x}{color.blue():02x}"
        sanitized = self._sanitize_color(hex_color)
        if not sanitized:
            return
        self._color_edit.blockSignals(True)
        self._color_edit.setText(sanitized)
        self._color_edit.blockSignals(False)
        self._update_color_preview(sanitized, has_input=True)

    def _on_color_text_changed(self, text: str) -> None:
        stripped = text.strip()
        sanitized = self._sanitize_color(stripped)
        self._update_color_preview(sanitized, has_input=bool(stripped))

    def _update_color_preview(self, hex_color: Optional[str], has_input: bool) -> None:
        if self._color_display is None:
            return
        if hex_color:
            text_color = self._contrast_text_color(hex_color)
            self._color_display.setText(hex_color.upper())
            self._color_display.setStyleSheet(
                f"border: 1px solid #555555; padding: 2px; background-color: {hex_color}; color: {text_color};"
            )
        elif has_input:
            self._color_display.setText("Invalid")
            self._color_display.setStyleSheet(
                "border: 1px solid #aa0000; padding: 2px; color: #aa0000;"
            )
        else:
            self._color_display.setText("None")
            self._color_display.setStyleSheet(
                "border: 1px solid #555555; padding: 2px; color: #666666;"
            )

    @staticmethod
    def _contrast_text_color(hex_color: str) -> str:
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
        except (ValueError, IndexError):
            return "#000000"
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return "#000000" if luminance > 0.5 else "#ffffff"


class _NameInputDialog(QDialog):
    """Custom dialog that enforces unique, non-empty names."""

    def __init__(
        self,
        title: str,
        label: str,
        existing_names: set[str],
        default: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._existing_names = existing_names

        self._line_edit = QLineEdit(default)
        self._line_edit.selectAll()
        self._info_label = QLabel(label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._info_label)
        layout.addWidget(self._line_edit)
        layout.addWidget(self._buttons)

        self._line_edit.textChanged.connect(self._on_text_changed)
        self._on_text_changed(self._line_edit.text())

    def _on_text_changed(self, text: str) -> None:
        trimmed = text.strip()
        valid = bool(trimmed) and trimmed not in self._existing_names
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
        if trimmed in self._existing_names:
            self._info_label.setText("Name already exists. Choose a different name.")
        else:
            self._info_label.setText("Control name:")

    @staticmethod
    def get_name(
        parent: QWidget,
        title: str,
        existing_names: set[str],
        default: str = "",
    ) -> Optional[str]:
        dialog = _NameInputDialog(
            title=title,
            label="Control name:",
            existing_names=existing_names,
            default=default,
            parent=parent,
        )
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return dialog._line_edit.text().strip()
        return None


class LayoutDesignerTab(QWidget):
    """Tab that allows configuring control positions on a screenshot."""

    def __init__(
        self,
        device_manager: DeviceManager,
        layout_registry: LayoutRegistry,
        train_tab: TrainTab,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._device_manager = device_manager
        self._layout_registry = layout_registry
        self._train_tab = train_tab
        self._current_serial: Optional[str] = None
        self._current_game: Optional[str] = None
        self._controls_per_category: Dict[str, List[ControlMapping]] = {cat: [] for cat in Category.ALL}
        self._pixmap: Optional[QPixmap] = None
        self._snippets: List[Dict[str, object]] = []
        self._updating_value_format = False
        self._known_value_formats: list[str] = list(STATE_BOARD_VALUE_FORMATS)

        self._build_ui()
        self._wire_signals()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header_box = QGroupBox("Layout Controls")
        header_layout = QHBoxLayout(header_box)

        self._device_label = QLabel("No device selected")
        self._game_label = QLabel("No game selected")
        self._category_combo = QComboBox()
        for category in Category.ALL:
            self._category_combo.addItem(category)
        self._capture_button = QPushButton("Capture Screenshot")
        self._clear_regions_button = QPushButton("Clear Regions")

        header_layout.addWidget(QLabel("Device:"))
        header_layout.addWidget(self._device_label, stretch=1)
        header_layout.addWidget(QLabel("Game:"))
        header_layout.addWidget(self._game_label, stretch=1)
        header_layout.addWidget(QLabel("Category:"))
        header_layout.addWidget(self._category_combo)
        header_layout.addWidget(self._capture_button)
        header_layout.addWidget(self._clear_regions_button)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)

        self._canvas = ScreenshotCanvas()
        self._canvas.setMinimumSize(1024, 576)
        canvas_scroll = QScrollArea()
        canvas_scroll.setWidgetResizable(True)
        canvas_scroll.setBackgroundRole(QPalette.ColorRole.Dark)
        canvas_scroll.setMinimumSize(1040, 620)
        canvas_scroll.setWidget(self._canvas)

        self._control_list = QListWidget()
        self._control_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        control_box = QGroupBox("Control Regions")
        control_layout = QVBoxLayout(control_box)
        control_layout.addWidget(self._control_list)

        self._rename_button = QPushButton("Rename")
        self._delete_button = QPushButton("Delete")
        self._export_button = QPushButton("Export Snippet")

        button_row = QHBoxLayout()
        button_row.addWidget(self._rename_button)
        button_row.addWidget(self._delete_button)
        button_row.addWidget(self._export_button)

        control_layout.addLayout(button_row)

        format_row = QHBoxLayout()
        self._value_format_label = QLabel("Value Format:")
        self._value_format_combo = QComboBox()
        self._value_format_combo.addItem("Automatic (no format)", "")
        for option in self._known_value_formats:
            self._value_format_combo.addItem(option, option)
        self._value_format_combo.addItem("Custom…", "__custom__")
        self._value_format_combo.setEnabled(False)
        format_row.addWidget(self._value_format_label)
        format_row.addWidget(self._value_format_combo, stretch=1)
        control_layout.addLayout(format_row)
        self._value_format_label.setVisible(False)
        self._value_format_combo.setVisible(False)

        splitter.addWidget(canvas_scroll)
        splitter.addWidget(control_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1100, 320])

        self._status_label = QLabel("Capture a screenshot to begin marking controls.")
        self._status_label.setWordWrap(True)

        layout.addWidget(header_box)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self._status_label)

    def _wire_signals(self) -> None:
        self._category_combo.currentIndexChanged.connect(self._on_category_changed)
        self._capture_button.clicked.connect(self._on_capture_clicked)
        self._clear_regions_button.clicked.connect(self._on_clear_regions_clicked)
        self._rename_button.clicked.connect(self._on_rename_clicked)
        self._delete_button.clicked.connect(self._on_delete_clicked)
        self._export_button.clicked.connect(self._on_export_clicked)
        self._control_list.itemDoubleClicked.connect(self._on_rename_item)
        self._control_list.currentItemChanged.connect(self._on_control_selection_changed)
        self._canvas.region_drawn.connect(self._on_region_drawn)
        self._value_format_combo.currentIndexChanged.connect(self._on_value_format_changed)

    def bind_signals(self) -> None:
        self._device_manager.error_occurred.connect(self._on_error)
        self._device_manager.screenshot_captured.connect(self._on_screenshot_captured)

    def set_active_device(self, serial: Optional[str], display_name: Optional[str]) -> None:
        self._current_serial = serial
        self._device_label.setText(display_name or "No device selected")
        self._update_status_for_context()
        self._update_buttons()
        self._train_tab.set_active_device(serial, display_name)

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
                "Please select a game from the Devices tab before capturing a layout.",
            )
            return
        self._status_label.setText(f"Capturing screenshot for '{self._current_game}'...")
        self._device_manager.capture_screenshot(self._current_serial)

    def _on_clear_regions_clicked(self) -> None:
        for controls in self._controls_per_category.values():
            controls.clear()
        self._control_list.clear()
        self._canvas.clear_regions()
        if self._current_game:
            self._status_label.setText(f"All regions cleared for {self._current_game}.")
        else:
            self._status_label.setText("Regions cleared.")
        self._update_canvas_regions()
        self._persist_current_regions()
        self._update_buttons()

    def _on_rename_clicked(self) -> None:
        item = self._control_list.currentItem()
        if not item:
            return
        index = self._control_list.row(item)
        if self.current_category == Category.SNIPPETS:
            if not (0 <= index < len(self._snippets)):
                return
            snippet = self._snippets[index]
            original_name = snippet.get("export_name") or snippet.get("name") or ""
            new_name, ok = QInputDialog.getText(
                self,
                "Rename Snippet",
                "Snippet name:",
                text=original_name,
            )
            if not ok:
                return
            new_name = new_name.strip()
            if not new_name:
                QMessageBox.warning(self, "Rename Snippet", "Name cannot be empty.")
                return
            snippet["export_name"] = new_name
            self._persist_snippets()
            self._refresh_control_list()
            self._status_label.setText(f"Snippet renamed to '{new_name}'.")
            return

        controls = self._controls_per_category[self.current_category]
        control = controls[index]
        existing = self._collect_existing_names(exclude=control.name)
        name = _NameInputDialog.get_name(
            parent=self,
            title="Rename Control",
            existing_names=existing,
            default=control.name,
        )
        if name:
            control.name = name
            item.setText(self._format_control_text(control))
            self._update_canvas_regions()
            self._persist_current_regions()
            self._update_buttons()

    def _on_delete_clicked(self) -> None:
        item = self._control_list.currentItem()
        if not item:
            return
        index = self._control_list.row(item)
        if self.current_category == Category.SNIPPETS:
            if 0 <= index < len(self._snippets):
                entry = self._snippets.pop(index)
                file_path = entry.get("file")
                if file_path:
                    try:
                        Path(file_path).unlink(missing_ok=True)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                self._persist_snippets()
                self._refresh_control_list()
                self._status_label.setText("Snippet deleted.")
            return

        controls = self._controls_per_category[self.current_category]
        del controls[index]
        self._control_list.takeItem(index)
        self._update_canvas_regions()
        self._status_label.setText("Control removed.")
        self._persist_current_regions()
        self._update_buttons()
        self._update_value_format_ui()

    def _on_export_clicked(self) -> None:
        if not self._current_game:
            QMessageBox.warning(self, "Export Snippet", "Select a game before exporting regions.")
            return
        if not self._pixmap:
            QMessageBox.warning(self, "Export Snippet", "Capture a screenshot before exporting regions.")
            return

        control = self._get_selected_control()
        if control is None:
            QMessageBox.information(self, "Export Snippet", "Select a region to export.")
            return

        export_name, ok = QInputDialog.getText(
            self,
            "Export Region",
            "Export name:",
            text=control.name,
        )
        if not ok:
            return

        export_name = export_name.strip() or control.name.strip()
        if not export_name:
            QMessageBox.warning(self, "Export Snippet", "Export name cannot be empty.")
            return

        x, y, w, h = control.rect
        source_rect = QRect(x, y, w, h).intersected(self._pixmap.rect())
        if source_rect.width() <= 0 or source_rect.height() <= 0:
            QMessageBox.warning(
                self,
                "Export Snippet",
                "Selected region lies outside of the captured screenshot.",
            )
            return

        cropped = self._pixmap.copy(source_rect)
        if cropped.isNull():
            QMessageBox.warning(self, "Export Snippet", "Failed to crop the selected region.")
            return

        search_options = _SnippetOptionsDialog.exec_options(
            rect_tuple := (x, y, w, h),
            parent=self,
            default={"mode": "exact"},
            snippet_pixmap=cropped,
        )
        if not search_options:
            return

        processed_pixmap = self._apply_snippet_mask(cropped, search_options)

        game_slug = self._sanitize_export_name(self._current_game)
        snippet_dir = Path("data") / "layout_snippets" / game_slug
        snippet_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self._sanitize_export_name(export_name)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{safe_name}_{x}_{y}_{w}_{h}.png"
        export_path = snippet_dir / filename

        if not processed_pixmap.save(str(export_path), "PNG"):
            QMessageBox.critical(
                self,
                "Export Snippet",
                f"Unable to save the cropped image to:\n{export_path}",
            )
            return

        metadata_entry = {
            "timestamp": timestamp,
            "game": self._current_game,
            "name": control.name,
            "export_name": export_name,
            "category": control.category,
            "value_format": control.value_format or None,
            "rect": {"x": x, "y": y, "width": w, "height": h},
            "file": str(export_path),
            "search": search_options,
        }
        self._snippets.append(metadata_entry)
        self._persist_snippets()

        self._status_label.setText(f"Exported '{control.name}' to {export_path}")
        QMessageBox.information(
            self,
            "Export Snippet",
            f"Saved cropped image to:\n{export_path}",
        )
        self._load_snippets_for_game()
        self._refresh_control_list()
        self._load_snippets_for_game()
        self._persist_snippets()

    def _on_rename_item(self, item: QListWidgetItem) -> None:
        self._on_rename_clicked()

    def _on_region_drawn(self, rect) -> None:
        if not self._pixmap:
            return
        name = _NameInputDialog.get_name(
            parent=self,
            title="New Snippet" if self.current_category == Category.SNIPPETS else "New Control",
            existing_names=self._collect_existing_names(),
        )
        if not name:
            self._status_label.setText("Region creation cancelled.")
            return

        rect_tuple = (rect.x(), rect.y(), rect.width(), rect.height())

        if self.current_category == Category.SNIPPETS:
            success = self._create_snippet(name, rect_tuple)
            if success:
                self._status_label.setText(f"Snippet '{name}' saved.")
                self._load_snippets_for_game()
                self._refresh_control_list()
            else:
                self._status_label.setText("Failed to save snippet.")
            return

        control = ControlMapping(
            name=name,
            rect=rect_tuple,
            category=self.current_category,
        )
        if control.category == Category.STATE_BOARDS:
            fmt = _StateBoardFormatDialog.get_format(
                self,
                "",
                self._known_value_formats,
            )
            if fmt is not None:
                fmt = fmt.strip()
                if fmt:
                    control.value_format = fmt
                    self._ensure_value_format_option(fmt)
                else:
                    control.value_format = ""
        self._controls_per_category[self.current_category].append(control)
        self._control_list.addItem(self._format_control_text(control))
        self._control_list.setCurrentRow(self._control_list.count() - 1)
        self._update_canvas_regions()
        self._status_label.setText(f"Added control '{control.name}'.")
        self._persist_current_regions()
        self._update_buttons()

    def _on_screenshot_captured(self, serial: str, data: bytes) -> None:
        if serial != self._current_serial:
            return
        pixmap = QPixmap()
        if not pixmap.loadFromData(data, "PNG"):
            self._status_label.setText("Failed to load screenshot data.")
            return
        self._pixmap = pixmap
        self._canvas.set_pixmap(pixmap)
        self._status_label.setText("Screenshot loaded. Draw regions by dragging on the image.")
        self._update_buttons()

    def _on_error(self, message: str) -> None:
        if "Screenshot" in message:
            self._status_label.setText(message)

    def _update_canvas_regions(self) -> None:
        regions: List[Region] = []
        if self.current_category != Category.SNIPPETS:
            for controls in self._controls_per_category.values():
                for control in controls:
                    rect = QRect(control.rect[0], control.rect[1], control.rect[2], control.rect[3])
                    label = f"[{control.category}] {control.name}"
                    if control.category == Category.STATE_BOARDS and control.value_format:
                        label = f"{label} [{control.value_format}]"
                    regions.append(Region(name=label, rect=rect))
        self._canvas.set_regions(regions)

    def _format_control_text(self, control: ControlMapping) -> str:
        x, y, w, h = control.rect
        base = f"{control.name} — ({x}, {y}, {w}, {h})"
        if control.category == Category.STATE_BOARDS and control.value_format:
            return f"{base} [{control.value_format}]"
        return base

    def _on_control_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        previous: Optional[QListWidgetItem],
    ) -> None:
        self._update_buttons()
        self._update_value_format_ui()

    def _select_value_format(self, value: str) -> None:
        self._updating_value_format = True
        try:
            target_value = value or ""
            if target_value:
                self._ensure_value_format_option(target_value)
            index = self._value_format_combo.findData(target_value)
            if index == -1 and target_value:
                self._ensure_value_format_option(target_value)
                index = self._value_format_combo.findData(target_value)
            self._value_format_combo.setCurrentIndex(index if index != -1 else 0)
        finally:
            self._updating_value_format = False

    def _ensure_value_format_option(self, value: str) -> None:
        trimmed = value.strip()
        if not trimmed:
            return
        if trimmed not in self._known_value_formats:
            self._known_value_formats.append(trimmed)
        if self._value_format_combo.findData(trimmed) == -1:
            insert_index = max(0, self._value_format_combo.count() - 1)
            self._value_format_combo.insertItem(insert_index, trimmed, trimmed)

    def _update_value_format_ui(self) -> None:
        is_state_board = self.current_category == Category.STATE_BOARDS
        self._value_format_label.setVisible(is_state_board)
        self._value_format_combo.setVisible(is_state_board)

        if not is_state_board:
            self._value_format_combo.setEnabled(False)
            self._select_value_format("")
            return

        current_row = self._control_list.currentRow()
        controls = self._controls_per_category[self.current_category]
        if current_row < 0 or current_row >= len(controls):
            self._value_format_combo.setEnabled(False)
            self._select_value_format("")
            return

        control = controls[current_row]
        self._value_format_combo.setEnabled(True)
        self._select_value_format(control.value_format)

    def _on_value_format_changed(self, index: int) -> None:
        if self._updating_value_format:
            return
        if self.current_category != Category.STATE_BOARDS:
            return

        current_row = self._control_list.currentRow()
        controls = self._controls_per_category[self.current_category]
        if current_row < 0 or current_row >= len(controls):
            return

        control = controls[current_row]
        selected_value = self._value_format_combo.currentData()
        if selected_value == "__custom__":
            initial = control.value_format or ""
            result = _StateBoardFormatDialog.get_format(
                self,
                initial,
                self._known_value_formats,
            )
            if result is None:
                # Revert selection
                self._select_value_format(control.value_format)
                return
            result = result.strip()
            self._ensure_value_format_option(result)
            control.value_format = result
            self._select_value_format(result)
        else:
            control.value_format = str(selected_value) if selected_value else ""
        item = self._control_list.item(current_row)
        if item:
            item.setText(self._format_control_text(control))
        self._persist_current_regions()

    def _update_buttons(self) -> None:
        has_device = self._current_serial is not None
        has_game = bool(self._current_game)
        controls = self._controls_per_category[self.current_category] if self.current_category != Category.SNIPPETS else []
        self._capture_button.setEnabled(has_device and has_game)
        any_controls = has_game and any(self._controls_per_category[c] for c in Category.ALL if c != Category.SNIPPETS)
        self._clear_regions_button.setEnabled(any_controls and self.current_category != Category.SNIPPETS)

        if self.current_category == Category.SNIPPETS:
            selected = self._control_list.currentRow() >= 0
            self._rename_button.setEnabled(has_game and selected and bool(self._snippets))
            self._delete_button.setEnabled(has_game and selected and bool(self._snippets))
            self._export_button.setEnabled(False)
        else:
            selected = self._control_list.currentItem() is not None
            self._rename_button.setEnabled(has_game and bool(controls) and selected)
            self._delete_button.setEnabled(has_game and bool(controls) and selected)
            self._export_button.setEnabled(has_game and bool(controls) and selected and self._pixmap is not None)
            self._update_value_format_ui()

    @staticmethod
    def _pixmap_to_bgr_array(pixmap: QPixmap) -> Optional[np.ndarray]:
        if pixmap.isNull():
            return None
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width = image.width()
        height = image.height()
        if width <= 0 or height <= 0:
            return None
        ptr = image.bits()
        ptr.setsize(width * height * 4)
        array = np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()
        bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        return bgr

    @staticmethod
    def _generate_snippet_mask(snippet_color: np.ndarray) -> Optional[np.ndarray]:
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

    def _apply_snippet_mask(self, pixmap: QPixmap, search_options: dict) -> QPixmap:
        if not search_options.get("use_mask"):
            return pixmap
        color_array = self._pixmap_to_bgr_array(pixmap)
        if color_array is None:
            return pixmap
        mask = self._generate_snippet_mask(color_array)
        if mask is None or cv2.countNonZero(mask) == 0:
            return pixmap
        rgba = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGBA)
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return pixmap
        rgba[~mask_bool] = (0, 0, 0, 0)
        height, width = rgba.shape[:2]
        image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(image.copy())

    @property
    def current_category(self) -> str:
        return self._category_combo.currentText() or Category.CONTROLS

    def _on_category_changed(self, index: int) -> None:
        category = self._category_combo.itemText(index)
        if not category:
            return
        self._refresh_control_list()
        self._update_status_for_context()
        self._update_buttons()

    def _refresh_control_list(self) -> None:
        if not self._current_game:
            self._control_list.clear()
            self._update_canvas_regions()
            return

        category = self.current_category
        self._control_list.clear()
        if category == Category.SNIPPETS:
            if self._snippets:
                for entry in self._snippets:
                    name = entry.get("export_name") or entry.get("name") or "snippet"
                    file_path = entry.get("file", "")
                    search_mode = (entry.get("search") or {}).get("mode", "global")
                    item_text = f"{name} [{search_mode}] — {file_path}"
                    self._control_list.addItem(item_text)
            self._update_canvas_regions()
            self._value_format_label.setVisible(False)
            self._value_format_combo.setVisible(False)
        else:
            for control in self._controls_per_category[category]:
                self._control_list.addItem(self._format_control_text(control))
            self._update_canvas_regions()
            self._update_value_format_ui()

    def _update_status_for_context(self) -> None:
        if self._current_game and self._current_serial:
            message = f"Ready to map controls for {self._current_game} using {self._device_label.text()}."
        elif self._current_game:
            message = (
                f"Viewing saved regions for {self._current_game}. Select a device in the Devices tab to capture new screenshots."
            )
        else:
            message = "Select a game in the Devices tab to begin mapping controls."

        if self.current_category == Category.SNIPPETS:
            message = (
                f"Snippets mode for {self._current_game or 'no game'}. Draw a region to save a cropped image snippet."
            )
        self._status_label.setText(message)

    def _reset_category_store(self) -> None:
        self._controls_per_category = {cat: [] for cat in Category.ALL}

    def _load_regions_for_game(self) -> None:
        self._reset_category_store()
        if not self._current_game:
            self._control_list.clear()
            self._canvas.clear_regions()
            self._status_label.setText("Select a game in the Devices tab to begin mapping controls.")
            return

        regions = self._layout_registry.regions_for(self._current_game)
        for region in regions:
            category = region.category if region.category in Category.ALL else Category.CONTROLS
            control = ControlMapping(
                name=region.name,
                rect=(region.x, region.y, region.width, region.height),
                category=category,
                value_format=region.value_format or "",
            )
            self._controls_per_category.setdefault(category, []).append(control)

        self._refresh_control_list()
        if regions:
            self._status_label.setText(
                f"Loaded {len(regions)} region(s) for {self._current_game}."
            )
        else:
            self._status_label.setText(
                f"No saved regions for {self._current_game}. Capture a screenshot and drag to add one."
            )

    def _load_snippets_for_game(self) -> None:
        self._snippets = []
        if not self._current_game:
            return
        metadata_path = self._snippet_metadata_path()
        if not metadata_path.exists():
            return
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "search" not in entry:
                            entry["search"] = {"mode": "global"}
                        self._snippets.append(entry)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return

    def _persist_current_regions(
        self,
        game_name: Optional[str] = None,
    ) -> None:
        target_game = game_name if game_name is not None else self._current_game
        if not target_game:
            return

        all_regions: List[RegionConfig] = []
        for category, controls in self._controls_per_category.items():
            if category == Category.SNIPPETS:
                continue
            for control in controls:
                all_regions.append(
                    RegionConfig(
                        name=control.name,
                        category=category,
                        x=control.rect[0],
                        y=control.rect[1],
                        width=control.rect[2],
                        height=control.rect[3],
                        value_format=control.value_format or None,
                    )
                )

        if all_regions:
            self._layout_registry.save_regions(target_game, all_regions)
        else:
            self._layout_registry.clear_regions(target_game)

    def _collect_existing_names(self, exclude: Optional[str] = None) -> set[str]:
        names: set[str] = set()
        for controls in self._controls_per_category.values():
            for control in controls:
                if control.name != exclude:
                    names.add(control.name)
        for snippet in self._snippets:
            snippet_name = snippet.get("export_name") or snippet.get("name")
            if snippet_name and snippet_name != exclude:
                names.add(str(snippet_name))
        return names

    def _create_snippet(self, name: str, rect: Tuple[int, int, int, int]) -> bool:
        if not self._pixmap or not self._current_game:
            QMessageBox.warning(self, "Export Snippet", "Capture a screenshot before creating snippets.")
            return False

        x, y, w, h = rect
        source_rect = QRect(x, y, w, h).intersected(self._pixmap.rect())
        if source_rect.width() <= 0 or source_rect.height() <= 0:
            QMessageBox.warning(self, "Export Snippet", "Selected region lies outside of the screenshot.")
            return False

        cropped = self._pixmap.copy(source_rect)
        if cropped.isNull():
            QMessageBox.critical(self, "Export Snippet", "Failed to crop the selected region.")
            return False

        search_options = _SnippetOptionsDialog.exec_options(
            rect=rect,
            parent=self,
            snippet_pixmap=cropped,
        )
        if not search_options:
            return False

        processed_pixmap = self._apply_snippet_mask(cropped, search_options)

        snippet_dir = self._snippet_directory()
        if snippet_dir is None:
            QMessageBox.critical(self, "Export Snippet", "Unable to determine snippet directory.")
            return False
        snippet_dir.mkdir(parents=True, exist_ok=True)

        sanitized = self._sanitize_export_name(name)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{sanitized}_{x}_{y}_{w}_{h}.png"
        export_path = snippet_dir / filename

        if not processed_pixmap.save(str(export_path), "PNG"):
            QMessageBox.critical(
                self,
                "Export Snippet",
                f"Unable to save the cropped image to:\n{export_path}",
            )
            return False

        metadata_entry = {
            "timestamp": timestamp,
            "game": self._current_game,
            "name": name,
            "category": self.current_category,
            "rect": {"x": x, "y": y, "width": w, "height": h},
            "file": str(export_path),
            "search": search_options,
        }
        self._snippets.append(metadata_entry)
        self._persist_snippets()

        return True

    def _persist_snippets(self) -> None:
        if not self._current_game:
            return
        metadata_path = self._snippet_metadata_path()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with metadata_path.open("w", encoding="utf-8") as handle:
                for entry in self._snippets:
                    json.dump(entry, handle, ensure_ascii=False)
                    handle.write("\n")
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Snippets",
                f"Failed to persist snippets metadata:\n{exc}",
            )

    def _snippet_directory(self) -> Optional[Path]:
        if not self._current_game:
            return None
        game_slug = self._sanitize_export_name(self._current_game)
        return Path("data") / "layout_snippets" / game_slug

    def _snippet_metadata_path(self) -> Path:
        directory = self._snippet_directory() or Path("data") / "layout_snippets" / "unspecified"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "metadata.jsonl"

    def _get_selected_control(self) -> Optional[ControlMapping]:
        item = self._control_list.currentItem()
        if not item:
            return None
        index = self._control_list.row(item)
        if self.current_category == Category.SNIPPETS:
            return None
        controls = self._controls_per_category[self.current_category]
        if 0 <= index < len(controls):
            return controls[index]
        return None

    @staticmethod
    def _sanitize_export_name(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
        return sanitized.strip("_") or "snippet"

    def set_active_game(self, game_name: Optional[str]) -> None:
        normalized = game_name or None
        if normalized == self._current_game:
            return

        previous_game = self._current_game
        if previous_game:
            self._persist_current_regions(game_name=previous_game)

        self._current_game = normalized
        self._game_label.setText(self._current_game or "No game selected")
        self._load_regions_for_game()
        self._load_snippets_for_game()
        self._update_buttons()
        self._update_status_for_context()
        self._train_tab.set_active_game(self._current_game)


