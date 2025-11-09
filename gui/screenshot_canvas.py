from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter, QPen, QPixmap, QColor
from PyQt6.QtWidgets import QWidget


@dataclass(slots=True)
class Region:
    name: str
    rect: QRect


class ScreenshotCanvas(QWidget):
    """Canvas widget that displays a screenshot and allows drawing regions."""

    region_drawn = pyqtSignal(QRect)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._regions: List[Region] = []
        self._drawing = False
        self._start_point = QPoint()
        self._current_rect: Optional[QRect] = None
        self.setMouseTracking(True)

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._pixmap = pixmap
        if pixmap:
            self.setFixedSize(pixmap.size())
        else:
            self.setFixedSize(400, 300)
        self.update()

    def set_regions(self, regions: List[Region]) -> None:
        self._regions = regions
        self.update()

    def clear_regions(self) -> None:
        self._regions = []
        self._current_rect = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap:
            painter.drawPixmap(0, 0, self._pixmap)

        pen_existing = QPen(QColor(0, 255, 0))
        pen_existing.setWidth(2)
        painter.setPen(pen_existing)

        for region in self._regions:
            painter.drawRect(region.rect)
            painter.drawText(region.rect.topLeft() + QPoint(4, 14), region.name)

        if self._current_rect:
            pen_current = QPen(QColor(255, 255, 0))
            pen_current.setWidth(2)
            pen_current.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen_current)
            painter.drawRect(self._current_rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._pixmap or event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.rect().contains(event.position().toPoint()):
            return
        self._drawing = True
        self._start_point = event.position().toPoint()
        self._current_rect = QRect(self._start_point, self._start_point)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._drawing or not self._pixmap:
            return
        current_point = self._clamp_to_pixmap(event.position().toPoint())
        self._current_rect = QRect(self._start_point, current_point).normalized()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._drawing or not self._pixmap:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        current_point = self._clamp_to_pixmap(event.position().toPoint())
        rect = QRect(self._start_point, current_point).normalized()
        self._drawing = False
        self._current_rect = None
        if rect.width() > 5 and rect.height() > 5:
            self.region_drawn.emit(rect)
        self.update()

    def _clamp_to_pixmap(self, point: QPoint) -> QPoint:
        if not self._pixmap:
            return point
        x = max(0, min(point.x(), self._pixmap.width() - 1))
        y = max(0, min(point.y(), self._pixmap.height() - 1))
        return QPoint(x, y)

