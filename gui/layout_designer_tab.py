from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QPalette, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.device_manager import DeviceManager
from core.layout_registry import LayoutRegistry, RegionConfig
from gui.screenshot_canvas import Region, ScreenshotCanvas


class Category:
    CONTROLS = "Controls"
    STATE_BOARDS = "State Boards"
    DIALOGUES = "Dialogues"

    ALL = [CONTROLS, STATE_BOARDS, DIALOGUES]


@dataclass(slots=True)
class ControlMapping:
    name: str
    rect: Tuple[int, int, int, int]  # x, y, width, height
    category: str = Category.CONTROLS
    notes: str = ""


class LayoutDesignerTab(QWidget):
    """Tab that allows configuring control positions on a screenshot."""

    def __init__(
        self,
        device_manager: DeviceManager,
        layout_registry: LayoutRegistry,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._device_manager = device_manager
        self._layout_registry = layout_registry
        self._current_serial: Optional[str] = None
        self._current_game: Optional[str] = None
        self._controls_per_category: Dict[str, List[ControlMapping]] = {cat: [] for cat in Category.ALL}
        self._pixmap: Optional[QPixmap] = None

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

        button_row = QHBoxLayout()
        button_row.addWidget(self._rename_button)
        button_row.addWidget(self._delete_button)

        control_layout.addLayout(button_row)

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
        self._control_list.itemDoubleClicked.connect(self._on_rename_item)
        self._control_list.currentItemChanged.connect(lambda *_: self._update_buttons())
        self._canvas.region_drawn.connect(self._on_region_drawn)

    def bind_signals(self) -> None:
        self._device_manager.error_occurred.connect(self._on_error)
        self._device_manager.screenshot_captured.connect(self._on_screenshot_captured)

    def set_active_device(self, serial: Optional[str], display_name: Optional[str]) -> None:
        self._current_serial = serial
        self._device_label.setText(display_name or "No device selected")
        self._update_status_for_context()
        self._update_buttons()

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
        controls = self._controls_per_category[self.current_category]
        control = controls[index]
        name, ok = QInputDialog.getText(self, "Rename Control", "Control name:", text=control.name)
        if ok and name.strip():
            control.name = name.strip()
            item.setText(self._format_control_text(control))
            self._update_canvas_regions()
            self._persist_current_regions()

    def _on_delete_clicked(self) -> None:
        item = self._control_list.currentItem()
        if not item:
            return
        index = self._control_list.row(item)
        controls = self._controls_per_category[self.current_category]
        del controls[index]
        self._control_list.takeItem(index)
        self._update_canvas_regions()
        self._status_label.setText("Control removed.")
        self._persist_current_regions()
        self._update_buttons()

    def _on_rename_item(self, item: QListWidgetItem) -> None:
        self._on_rename_clicked()

    def _on_region_drawn(self, rect) -> None:
        if not self._pixmap:
            return
        name, ok = QInputDialog.getText(self, "New Control", "Control name:")
        if not ok or not name.strip():
            self._status_label.setText("Region creation cancelled.")
            return
        control = ControlMapping(
            name=name.strip(),
            rect=(rect.x(), rect.y(), rect.width(), rect.height()),
            category=self.current_category,
        )
        self._controls_per_category[self.current_category].append(control)
        self._control_list.addItem(self._format_control_text(control))
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
        for controls in self._controls_per_category.values():
            for control in controls:
                rect = QRect(control.rect[0], control.rect[1], control.rect[2], control.rect[3])
                regions.append(Region(name=f"[{control.category}] {control.name}", rect=rect))
        self._canvas.set_regions(regions)

    def _format_control_text(self, control: ControlMapping) -> str:
        x, y, w, h = control.rect
        return f"{control.name} — ({x}, {y}, {w}, {h})"

    def _update_buttons(self) -> None:
        has_device = self._current_serial is not None
        has_game = bool(self._current_game)
        controls = self._controls_per_category[self.current_category]
        self._capture_button.setEnabled(has_device and has_game)
        any_controls = has_game and any(self._controls_per_category.values())
        self._clear_regions_button.setEnabled(any_controls)
        self._rename_button.setEnabled(
            has_game and bool(controls) and self._control_list.currentItem() is not None
        )
        self._delete_button.setEnabled(
            has_game and bool(controls) and self._control_list.currentItem() is not None
        )

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
        for control in self._controls_per_category[category]:
            self._control_list.addItem(self._format_control_text(control))
        self._update_canvas_regions()

    def _update_status_for_context(self) -> None:
        if self._current_game and self._current_serial:
            self._status_label.setText(
                f"Ready to map controls for {self._current_game} using {self._device_label.text()}."
            )
        elif self._current_game:
            self._status_label.setText(
                f"Viewing saved regions for {self._current_game}. Select a device in the Devices tab to capture new screenshots."
            )
        else:
            self._status_label.setText("Select a game in the Devices tab to begin mapping controls.")

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

    def _persist_current_regions(
        self,
        game_name: Optional[str] = None,
    ) -> None:
        target_game = game_name if game_name is not None else self._current_game
        if not target_game:
            return

        all_regions: List[RegionConfig] = []
        for category, controls in self._controls_per_category.items():
            for control in controls:
                all_regions.append(
                    RegionConfig(
                        name=control.name,
                        category=category,
                        x=control.rect[0],
                        y=control.rect[1],
                        width=control.rect[2],
                        height=control.rect[3],
                    )
                )

        if all_regions:
            self._layout_registry.save_regions(target_game, all_regions)
        else:
            self._layout_registry.clear_regions(target_game)

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
        self._update_buttons()
        self._update_status_for_context()


