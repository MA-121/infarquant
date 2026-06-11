from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui
from PyQt5.QtCore import Qt, QSize, QRect, QEventLoop
from PyQt5.QtGui import QColor, QCursor, QImage, QKeySequence, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analysis import (
    ChannelValidationError,
    DRAW_THICKNESS,
    channel_plane,
    draw_dashed_polyline,
    get_largest_contour,
    qt_process_images,
    segment_image_by_hsv,
    split_image,
    validate_channel_selection,
)
from .preprocess import process_folder


def _resolve_logo_path() -> str:
    """Resolve logo path for repo, editable installs, and frozen exe."""
    here = Path(__file__).resolve()
    candidates = []
    base = getattr(sys, "_MEIPASS", None)  # set by PyInstaller at runtime
    if base:
        candidates.append(Path(base) / "docs" / "infarquant_logo.png")
    candidates += [
        here.parents[2] / "docs" / "infarquant_logo.png",
        Path.cwd() / "docs" / "infarquant_logo.png",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


class CustomTabBar(QTabBar):
    def tabSizeHint(self, index: int) -> QSize:
        # Get default size
        size = super().tabSizeHint(index)
        text = self.tabText(index)
        # Use font metrics to compute required width
        fm = self.fontMetrics()
        # Add padding to width
        width = fm.width(text) + 40
        # Ensure a minimum width to avoid extremely small tabs
        width = max(width, 100)
        size.setWidth(width)
        return size

# -----------------------------------------------------------------------------
# UI styling and instruction dispatch (Qt dock replaces floating OpenCV windows)
# -----------------------------------------------------------------------------

APP_STYLE_SHEET = """
QMainWindow { background: #F6F7FB; }
QLabel { color: #1F2937; }
QFrame[card="true"] { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; }
QLineEdit, QSpinBox, QComboBox, QTextEdit {
    background: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 4px 6px;
}
QPushButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border-radius: 6px;
    padding: 6px 10px;
}
QPushButton:disabled { background-color: #93C5FD; }
QPushButton[acceptAction="true"] {
    background-color: #9CA3AF;
    color: #FFFFFF;
    border: 1px solid #6B7280;
    font-weight: 600;
}
QPushButton[acceptAction="true"][ready="true"] {
    background-color: #0F766E;
    border: 1px solid #0B5E57;
}
QPushButton[acceptAction="true"]:disabled {
    background-color: #D1D5DB;
    color: #6B7280;
    border: 1px solid #9CA3AF;
}
QPushButton#primaryAction { background-color: #0F766E; }
QPushButton#secondaryAction { background-color: #1D4ED8; }
QPushButton#accentAction { background-color: #D97706; color: #111827; }
QPushButton[nav="true"] {
    background-color: #E5E7EB;
    color: #1F2937;
    border: 1px solid #CBD5E1;
    border-bottom: 3px solid #9CA3AF;
    border-radius: 0;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13pt;
}
QPushButton[nav="true"]:hover { background-color: #DDE3EA; }
QPushButton[nav="true"]:checked {
    background-color: #FFFFFF;
    color: #1D4ED8;
    border-bottom: 3px solid #1D4ED8;
}
QToolButton {
    background-color:#E5E7EB;
    color:#111827;
    border-radius:8px;
    min-width:18px;
    min-height:18px;
}
QDockWidget::title {
    background: #E5E7EB;
    padding: 6px;
    font-weight: 600;
}
QStatusBar { background: #E5E7EB; color: #111827; }
"""

INSTRUCTION_SINK = None

def set_instruction_sink(sink) -> None:
    """Register a UI sink for instructions (used to keep shortcuts in-app)."""
    global INSTRUCTION_SINK
    INSTRUCTION_SINK = sink

def instructions_use_cv() -> bool:
    """Return True when OpenCV instruction windows should be used."""
    return INSTRUCTION_SINK is None

def update_instruction_list(title: str, lines: list) -> None:
    """Send a list of instruction lines to the UI sink if available."""
    if INSTRUCTION_SINK is not None:
        INSTRUCTION_SINK.show_list(title, lines)

def update_instruction_table(title: str, pairs: list) -> None:
    """Send a key/action table to the UI sink if available."""
    if INSTRUCTION_SINK is not None:
        INSTRUCTION_SINK.show_table(title, pairs)


class InstructionPanel(QWidget):
    """Dockable panel that displays context-sensitive shortcuts and hints."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = QLabel("Shortcuts")
        self._title.setStyleSheet("font-weight: 700; font-size: 12pt;")
        self._context = QLabel("Idle")
        self._context.setStyleSheet("color: #6B7280;")
        self._content = QTextEdit()
        self._content.setReadOnly(True)
        self._content.setMinimumWidth(260)
        self._content.setText("Shortcuts will appear here when analysis starts.")
        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addWidget(self._title)
        layout.addWidget(self._context)
        layout.addWidget(self._content)
        self.setLayout(layout)

    def set_context(self, text: str) -> None:
        self._context.setText(text)

    def show_list(self, title: str, lines: list) -> None:
        self.set_context(title)
        html = "<ul>" + "".join([f"<li>{line}</li>" for line in lines]) + "</ul>"
        self._content.setHtml(html)

    def show_table(self, title: str, pairs: list) -> None:
        self.set_context(title)
        rows = "".join([f"<tr><td><b>{k}</b></td><td>{a}</td></tr>" for k, a in pairs])
        html = (
            "<table style='width:100%; border-collapse: collapse;'>"
            "<tr><th align='left'>Key</th><th align='left'>Action</th></tr>"
            f"{rows}</table>"
        )
        self._content.setHtml(html)

    def clear(self) -> None:
        self.set_context("Idle")
        self._content.setText("Shortcuts will appear here when analysis starts.")


def create_card(title: str) -> Tuple[QFrame, QVBoxLayout]:
    """Create a styled container with a title for grouped UI controls."""
    card = QFrame()
    card.setProperty("card", True)
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    outer_layout = QVBoxLayout(card)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    outer_layout.setSpacing(0)

    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(20, 12, 20, 2)
    header_layout.setSpacing(12)
    title_label = QLabel(title)
    title_label.setStyleSheet("font-weight: 600; font-size: 12pt;")
    header_layout.addWidget(title_label)
    header_layout.addStretch(1)
    outer_layout.addWidget(header)

    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(20, 0, 20, 16)
    content_layout.setSpacing(12)
    outer_layout.addWidget(content)

    return card, content_layout


def save_csv_with_retry(parent, df, path) -> bool:
    """Save a DataFrame to CSV, prompting the user to close the file if it is
    locked (e.g. open in Excel) and retrying until it succeeds. Returns True
    once saved, or False if the user chooses to cancel.

    On Windows an open Excel file locks the CSV, so ``to_csv`` raises and -- in
    the windowed build with no console -- the failure would otherwise be silent.
    Shared by the Analyze and Quantify tabs so both behave identically.
    """
    while True:
        try:
            df.to_csv(path, index=False, na_rep="NA")
            return True
        except OSError as exc:
            choice = QMessageBox.warning(
                parent,
                "Cannot save results",
                "Please close your current results Excel file to save the new "
                f"results, then press Retry.\n\nFile:\n{path}\n\n({exc})",
                QMessageBox.Retry | QMessageBox.Cancel,
                QMessageBox.Retry,
            )
            if choice != QMessageBox.Retry:
                return False


def create_dot_cursor(size: int = 16) -> QCursor:
    """Create a small white-dot cursor for precise click actions."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#FFFFFF"))
    r = max(2, size // 6)
    painter.drawEllipse((size - r) // 2, (size - r) // 2, r, r)
    painter.end()
    return QCursor(pix, size // 2, size // 2)


class ImagePane(QWidget):
    """Widget that displays a single image with aspect-ratio preserving fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self._qimage = None
        self._display_rect = None
        self._click_handler = None
        self._hover_cursor = None
        self.setMouseTracking(True)

    def set_image(self, image: Optional[np.ndarray]) -> None:
        self._image = image
        if image is None:
            self._qimage = None
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self._qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def set_click_handler(self, handler) -> None:
        self._click_handler = handler

    def set_hover_cursor(self, cursor: Optional[QCursor]) -> None:
        self._hover_cursor = cursor

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#F3F4F6"))
        if self._qimage is None:
            painter.setPen(QColor("#9CA3AF"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image")
            return
        target = self.rect()
        img_w = self._qimage.width()
        img_h = self._qimage.height()
        scaled = self._qimage.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = target.x() + (target.width() - scaled.width()) // 2
        y = target.y() + (target.height() - scaled.height()) // 2
        self._display_rect = QRect(x, y, scaled.width(), scaled.height())
        painter.drawImage(self._display_rect, scaled)

    def mousePressEvent(self, event):
        if self._image is None or self._display_rect is None or self._click_handler is None:
            return
        if not self._display_rect.contains(event.pos()):
            return
        img_h, img_w = self._image.shape[:2]
        rel_x = event.pos().x() - self._display_rect.x()
        rel_y = event.pos().y() - self._display_rect.y()
        x = int(rel_x * (img_w / self._display_rect.width()))
        y = int(rel_y * (img_h / self._display_rect.height()))
        self._click_handler(x, y)

    def mouseMoveEvent(self, event):
        if self._hover_cursor and self._display_rect and self._display_rect.contains(event.pos()):
            self.setCursor(self._hover_cursor)
        else:
            self.unsetCursor()

    def enterEvent(self, event):
        if self._hover_cursor:
            self.setCursor(self._hover_cursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)


class QuadrantViewer(QWidget):
    """2x2 viewer that can show 1-4 images without warping."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_pane = ImagePane()
        self._grid_panes = [ImagePane() for _ in range(4)]
        self._stack = QStackedWidget()
        full_page = QWidget()
        full_layout = QVBoxLayout(full_page)
        full_layout.setContentsMargins(0, 0, 0, 0)
        full_layout.addWidget(self._full_pane)
        grid_page = QWidget()
        grid_layout = QGridLayout(grid_page)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(6)
        grid_layout.addWidget(self._grid_panes[0], 0, 0)
        grid_layout.addWidget(self._grid_panes[1], 0, 1)
        grid_layout.addWidget(self._grid_panes[2], 1, 0)
        grid_layout.addWidget(self._grid_panes[3], 1, 1)
        self._stack.addWidget(full_page)
        self._stack.addWidget(grid_page)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

    def set_click_handler(self, handler) -> None:
        self._full_pane.set_click_handler(handler)
        for pane in self._grid_panes:
            pane.set_click_handler(handler)

    def set_hover_cursor(self, cursor: Optional[QCursor]) -> None:
        self._full_pane.set_hover_cursor(cursor)
        for pane in self._grid_panes:
            pane.set_hover_cursor(cursor)

    def set_images(self, images: list) -> None:
        images = images or []
        if len(images) <= 1:
            self._stack.setCurrentIndex(0)
            self._full_pane.set_image(images[0] if images else None)
            return
        self._stack.setCurrentIndex(1)
        for idx, pane in enumerate(self._grid_panes):
            pane.setVisible(idx < len(images))
            pane.set_image(images[idx] if idx < len(images) else None)


class Stepper(QWidget):
    """Simple stepper for analysis flow."""

    def __init__(self, steps: list, parent=None):
        super().__init__(parent)
        self._labels = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for step in steps:
            label = QLabel(step)
            label.setStyleSheet("padding: 6px 12px; border: 1px solid #CBD5F5; background:#E5E7EB;")
            layout.addWidget(label)
            self._labels.append(label)
        layout.addStretch(1)

    def set_active(self, index: int) -> None:
        for idx, label in enumerate(self._labels):
            if idx == index:
                label.setStyleSheet("padding: 6px 12px; border: 1px solid #1D4ED8; background:#1D4ED8; color:#FFFFFF;")
            else:
                label.setStyleSheet("padding: 6px 12px; border: 1px solid #CBD5F5; background:#E5E7EB; color:#111827;")


class AnalysisWorkspace(QWidget):
    """Interactive analysis workspace with stepper and quadrant viewer."""

    def __init__(self, status_callback=None, parent=None):
        super().__init__(parent)
        self._status_callback = status_callback
        self._shortcuts = []
        self._action_buttons = {}
        self._pending_loop = None
        self._pending_result = None
        self._stepper = Stepper(["Draw Midline", "Brain Contour", "Infarct Contour"])
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #111827; font-size: 12pt; font-weight: 600;")
        self._action_row = QHBoxLayout()
        self._action_row.setSpacing(8)
        self._action_row_widget = QWidget()
        self._action_row_widget.setLayout(self._action_row)
        self._controls_row = QHBoxLayout()
        self._controls_row.setSpacing(8)
        self._controls_row_widget = QWidget()
        self._controls_row_widget.setLayout(self._controls_row)
        self._banner = QLabel("")
        self._banner.setStyleSheet("background:#FDE68A; color:#111827; padding:6px 10px; border-radius:6px;")
        self._banner.setWordWrap(True)
        self._viewer = QuadrantViewer()
        self._hover_cursor = create_dot_cursor(size=64)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(self._stepper, 0)
        header.addSpacing(12)
        header.addWidget(self._progress_label, 1)
        layout.addLayout(header)
        layout.addWidget(self._action_row_widget)
        layout.addWidget(self._controls_row_widget)
        layout.addWidget(self._banner)
        layout.addWidget(self._viewer, 1)

    def _set_status(self, text: str) -> None:
        if self._status_callback:
            self._status_callback(text)

    def set_progress(self, text: str) -> None:
        """Update the per-section progress label in the workspace header."""
        self._progress_label.setText(text)

    def _clear_shortcuts(self) -> None:
        for sc in self._shortcuts:
            sc.setEnabled(False)
        self._shortcuts = []

    def _register_shortcut(self, key: str, callback) -> None:
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(callback)
        self._shortcuts.append(sc)

    def _clear_action_row(self) -> None:
        while self._action_row.count():
            item = self._action_row.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
        self._action_buttons = {}

    def _clear_controls_row(self) -> None:
        while self._controls_row.count():
            item = self._controls_row.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

    def _set_actions(self, actions: list, controls: list, context_label: str, instruction: str, shortcuts: list, hover_cursor: bool) -> None:
        self._stepper.set_active({"Midline": 0, "Brain": 1, "Infarct": 2, "Pre-draw ROI": 0}.get(context_label, 0))
        self._banner.setText(instruction)
        self._viewer.set_hover_cursor(self._hover_cursor if hover_cursor else None)
        self._clear_action_row()
        for action in actions:
            btn = QPushButton(action["label"])
            btn.setToolTip(action.get("tooltip", ""))
            is_accept = action.get("role") == "accept" or action["label"] == "Accept"
            if is_accept:
                initial_enabled = bool(action.get("enabled", True))
                btn.setProperty("acceptAction", True)
                btn.setProperty("ready", bool(action.get("ready", initial_enabled)))
                btn.setEnabled(initial_enabled)
            elif "enabled" in action:
                btn.setEnabled(bool(action["enabled"]))
            btn.clicked.connect(action["callback"])
            self._action_row.addWidget(btn)
            self._action_buttons[action["label"]] = btn
            if is_accept:
                btn.style().unpolish(btn)
                btn.style().polish(btn)
                btn.update()
        self._action_row.addStretch(1)
        self._clear_controls_row()
        for widget in controls:
            self._controls_row.addWidget(widget)
        self._controls_row.addStretch(1)

    def _set_action_state(self, label: str, enabled: Optional[bool] = None, ready: Optional[bool] = None) -> None:
        btn = self._action_buttons.get(label)
        if btn is None:
            return
        if enabled is not None:
            btn.setEnabled(enabled)
        if ready is not None and bool(btn.property("acceptAction")):
            btn.setProperty("ready", bool(ready))
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        btn.update()

    def _is_action_enabled(self, label: str) -> bool:
        btn = self._action_buttons.get(label)
        return bool(btn and btn.isEnabled())

    def _wait_for_result(self):
        self._pending_loop = QEventLoop()
        self._pending_loop.exec_()
        result = self._pending_result
        self._pending_result = None
        return result

    def _finish_step(self, result):
        self._pending_result = result
        if self._pending_loop is not None:
            self._pending_loop.quit()

    # ------------------------------------------------------------------
    # Interactive steps implemented in Qt (no external OpenCV windows)
    # ------------------------------------------------------------------

    def run_midline(self, image: np.ndarray, display_info: str):
        """Interactive midline drawing using the workspace viewer."""
        self._set_status(f"Midline: {display_info}")
        points: list = []
        flip_h = False
        flip_v = False
        mod_image = image.copy()

        def _redraw():
            frame = mod_image.copy()
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], (255, 255, 255), DRAW_THICKNESS)
            for pt in points:
                cv2.circle(frame, pt, DRAW_THICKNESS, (255, 255, 255), -1)
            self._viewer.set_images([frame])
            ready = len(points) >= 2
            self._set_action_state("Accept", enabled=ready, ready=ready)

        def _apply_flip():
            nonlocal mod_image
            if flip_h and flip_v:
                mod_image = cv2.flip(image, -1)
            elif flip_h:
                mod_image = cv2.flip(image, 1)
            elif flip_v:
                mod_image = cv2.flip(image, 0)
            else:
                mod_image = image.copy()
            _redraw()

        def on_click(x, y):
            points.append((x, y))
            _redraw()

        def on_clear():
            points.clear()
            _redraw()

        def on_flip_h():
            nonlocal flip_h
            flip_h = not flip_h
            _apply_flip()

        def on_flip_v():
            nonlocal flip_v
            flip_v = not flip_v
            _apply_flip()

        def on_accept():
            if len(points) < 2:
                pyautogui.alert("Invalid midline: Please draw a midline with at least two points.")
                points.clear()
                _redraw()
                return
            self._finish_step((points, mod_image, (flip_h, flip_v)))

        def on_exit():
            self._finish_step(("EXIT", None, None))

        self._viewer.set_click_handler(on_click)
        self._clear_shortcuts()
        self._register_shortcut("C", on_clear)
        self._register_shortcut("H", on_flip_h)
        self._register_shortcut("V", on_flip_v)
        self._register_shortcut("A", lambda: on_accept() if self._is_action_enabled("Accept") else None)
        self._register_shortcut("Esc", on_exit)
        actions = [
            {"label": "Accept", "callback": on_accept, "tooltip": "Accept midline. Shortcut: A/Enter", "enabled": False, "ready": False},
            {"label": "Clear", "callback": on_clear, "tooltip": "Clear points. Shortcut: C"},
            {"label": "Flip H", "callback": on_flip_h, "tooltip": "Flip horizontally. Shortcut: H"},
            {"label": "Flip V", "callback": on_flip_v, "tooltip": "Flip vertically. Shortcut: V"},
            {"label": "Exit", "callback": on_exit, "tooltip": "Exit step. Shortcut: Esc"},
        ]
        shortcuts = [
            ("Left Click", "Add midline point"),
            ("C", "Clear points"),
            ("H", "Flip horizontal"),
            ("V", "Flip vertical"),
            ("A", "Accept"),
            ("Esc", "Exit"),
        ]
        self._set_actions(
            actions,
            [],
            "Midline",
            "Midline: click to draw the split line across the brain.",
            shortcuts,
            True,
        )
        _redraw()
        return self._wait_for_result()

    def run_brain_threshold(self, ref_image: np.ndarray, split_points: list, section_contour_threshold: int, display_info: str = ""):
        """Interactive brain contour threshold adjustment."""
        self._set_status(f"Brain threshold: {display_info}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(section_contour_threshold)
        value_box = QSpinBox()
        value_box.setRange(0, 255)
        value_box.setValue(section_contour_threshold)
        current_result = {"contours": None, "mask": None, "positions": None}

        def _compute(thresh_val: int):
            positions = {
                "Brain Lower H": 0,
                "Brain Lower S": 0,
                "Brain Lower V": thresh_val,
                "Brain Upper H": 255,
                "Brain Upper S": 255,
                "Brain Upper V": 255,
            }
            lower_keys = ["Brain Lower H", "Brain Lower S", "Brain Lower V"]
            upper_keys = ["Brain Upper H", "Brain Upper S", "Brain Upper V"]
            thresh_mask = segment_image_by_hsv(ref_image, lower_keys, upper_keys, positions)
            whole_contour, _ = get_largest_contour(thresh_mask)
            if whole_contour is None:
                current_result["contours"] = None
                current_result["mask"] = None
                current_result["positions"] = positions
                self._viewer.set_images([ref_image])
                self._set_action_state("Accept", enabled=False, ready=False)
                return
            whole_mask_filled = np.zeros_like(thresh_mask)
            cv2.drawContours(whole_mask_filled, [whole_contour], -1, 255, thickness=cv2.FILLED)
            left_region_mask, right_region_mask = split_image(ref_image, split_points, np.ones(ref_image.shape[:2], dtype=np.uint8))
            left_brain_mask = cv2.bitwise_and(whole_mask_filled, left_region_mask)
            right_brain_mask = cv2.bitwise_and(whole_mask_filled, right_region_mask)
            left_contour, _ = get_largest_contour(left_brain_mask)
            right_contour, _ = get_largest_contour(right_brain_mask)
            brain_contours = {"whole": whole_contour, "left": left_contour, "right": right_contour}
            whole = ref_image.copy()
            cv2.drawContours(whole, [whole_contour], -1, (255, 255, 255), DRAW_THICKNESS)
            left_preview = cv2.bitwise_and(ref_image, ref_image, mask=left_region_mask)
            if left_contour is not None:
                cv2.drawContours(left_preview, [left_contour], -1, (255, 255, 255), DRAW_THICKNESS)
            right_preview = cv2.bitwise_and(ref_image, ref_image, mask=right_region_mask)
            if right_contour is not None:
                cv2.drawContours(right_preview, [right_contour], -1, (255, 255, 255), DRAW_THICKNESS)
            self._viewer.set_images([whole, left_preview, right_preview])
            current_result["contours"] = brain_contours
            current_result["mask"] = thresh_mask
            current_result["positions"] = positions
            self._set_action_state("Accept", enabled=True, ready=True)

        def on_accept():
            if current_result["contours"] is None:
                pyautogui.alert("No brain contour found. Adjust the threshold.")
                return
            self._finish_step((current_result["contours"], current_result["mask"], current_result["positions"]))

        def on_exit():
            self._finish_step(("EXIT", None, None))

        def _sync_from_slider(val: int):
            if value_box.value() != val:
                value_box.blockSignals(True)
                value_box.setValue(val)
                value_box.blockSignals(False)
            _compute(val)

        def _sync_from_box(val: int):
            if slider.value() != val:
                slider.blockSignals(True)
                slider.setValue(val)
                slider.blockSignals(False)
            _compute(val)

        slider.valueChanged.connect(_sync_from_slider)
        value_box.valueChanged.connect(_sync_from_box)
        self._viewer.set_click_handler(lambda x, y: None)
        self._clear_shortcuts()
        self._register_shortcut("A", lambda: on_accept() if self._is_action_enabled("Accept") else None)
        self._register_shortcut("Esc", on_exit)
        actions = [
            {"label": "Accept", "callback": on_accept, "tooltip": "Accept brain contour. Shortcut: A", "enabled": False, "ready": False},
            {"label": "Exit", "callback": on_exit, "tooltip": "Exit step. Shortcut: Esc"},
        ]
        shortcuts = [
            ("Slider", "Adjust brain outline"),
            ("A", "Accept"),
            ("Esc", "Exit"),
        ]
        self._set_actions(
            actions,
            [QLabel("Brain threshold"), slider, value_box],
            "Brain",
            "Brain outline: adjust the slider until the whole brain is captured.",
            shortcuts,
            False,
        )
        _compute(section_contour_threshold)
        return self._wait_for_result()

    def run_infarct_threshold(
        self,
        infarct_img: np.ndarray,
        reference_img: np.ndarray,
        brain_contours: dict,
        split_points: list,
        infarct_contour_detection: str,
        display_info: str,
        cd68_color: str,
        exclude_color: str,
        cd68_start_val: int,
        exclude_start_val: int,
        fixed_roi_data: Optional[dict] = None,
    ):
        """Interactive infarct segmentation using in-app controls."""
        self._set_status(f"Infarct: {display_info}")
        roi_fixed = fixed_roi_data is not None
        fixed_poly_norm = None
        fixed_centroid_norm = None
        if roi_fixed and fixed_roi_data:
            poly_norm_orig = fixed_roi_data.get("poly_norm")
            centroid_norm_orig = fixed_roi_data.get("centroid_norm")
            flip_h_flag = bool(fixed_roi_data.get("flip_h", False))
            flip_v_flag = bool(fixed_roi_data.get("flip_v", False))
            if poly_norm_orig:
                adjusted_poly = []
                for xn, yn in poly_norm_orig:
                    x_adj = 1.0 - xn if flip_h_flag else xn
                    y_adj = 1.0 - yn if flip_v_flag else yn
                    adjusted_poly.append((x_adj, y_adj))
                fixed_poly_norm = adjusted_poly
            if centroid_norm_orig:
                cx, cy = centroid_norm_orig
                cx_adj = 1.0 - cx if flip_h_flag else cx
                cy_adj = 1.0 - cy if flip_v_flag else cy
                fixed_centroid_norm = (cx_adj, cy_adj)

        roi_points: list = []
        roi_mask: Optional[np.ndarray] = None
        roi_hull_points: Optional[np.ndarray] = None

        # Precompute default ROI position if fixed
        if roi_fixed and fixed_poly_norm is not None and fixed_centroid_norm is not None:
            img_h, img_w = infarct_img.shape[:2]
            pts = [(xn * img_w, yn * img_h) for xn, yn in fixed_poly_norm]
            pts_arr = np.array(pts, dtype=np.int32)
            min_x = np.min(pts_arr[:, 0])
            max_x = np.max(pts_arr[:, 0])
            min_y = np.min(pts_arr[:, 1])
            max_y = np.max(pts_arr[:, 1])
            shift_x = 0
            shift_y = 0
            if min_x < 0:
                shift_x = -min_x
            if max_x >= img_w:
                shift_x = min(shift_x, img_w - 1 - max_x) if shift_x != 0 else (img_w - 1 - max_x)
            if min_y < 0:
                shift_y = -min_y
            if max_y >= img_h:
                shift_y = min(shift_y, img_h - 1 - max_y) if shift_y != 0 else (img_h - 1 - max_y)
            pts_arr[:, 0] = np.clip(pts_arr[:, 0] + shift_x, 0, img_w - 1)
            pts_arr[:, 1] = np.clip(pts_arr[:, 1] + shift_y, 0, img_h - 1)
            roi_hull_points = pts_arr.reshape(-1, 1, 2)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [roi_hull_points], 255)
            roi_mask = mask

        # Sliders
        cd68_slider = QSlider(Qt.Horizontal)
        cd68_slider.setRange(0, 255)
        cd68_slider.setValue(cd68_start_val)
        cd68_box = QSpinBox()
        cd68_box.setRange(0, 255)
        cd68_box.setValue(cd68_start_val)
        exclude_slider = None
        exclude_box = None
        if exclude_color != "none":
            exclude_slider = QSlider(Qt.Horizontal)
            exclude_slider.setRange(0, 255)
            exclude_slider.setValue(exclude_start_val)
            exclude_box = QSpinBox()
            exclude_box.setRange(0, 255)
            exclude_box.setValue(exclude_start_val)

        current_index = 0
        selected_infarct_contour = None
        multi_contours: list = []
        top_contours: list = []
        views = [("infarct", infarct_img), ("reference", reference_img)]
        view_index = 0
        final_positions = None
        save_whole = None
        save_left = None
        save_right = None

        def on_click(x, y):
            nonlocal roi_hull_points, roi_mask
            if roi_fixed:
                if fixed_poly_norm is None or fixed_centroid_norm is None:
                    return
                img_h, img_w = infarct_img.shape[:2]
                orig_cx = fixed_centroid_norm[0] * img_w
                orig_cy = fixed_centroid_norm[1] * img_h
                dx = x - orig_cx
                dy = y - orig_cy
                pts = []
                for xn, yn in fixed_poly_norm:
                    px = xn * img_w + dx
                    py = yn * img_h + dy
                    pts.append((px, py))
                pts_arr = np.array(pts, dtype=np.int32)
                min_x = np.min(pts_arr[:, 0])
                max_x = np.max(pts_arr[:, 0])
                min_y = np.min(pts_arr[:, 1])
                max_y = np.max(pts_arr[:, 1])
                shift_x = 0
                shift_y = 0
                if min_x < 0:
                    shift_x = -min_x
                if max_x >= img_w:
                    shift_x = min(shift_x, img_w - 1 - max_x) if shift_x != 0 else (img_w - 1 - max_x)
                if min_y < 0:
                    shift_y = -min_y
                if max_y >= img_h:
                    shift_y = min(shift_y, img_h - 1 - max_y) if shift_y != 0 else (img_h - 1 - max_y)
                pts_arr[:, 0] = np.clip(pts_arr[:, 0] + shift_x, 0, img_w - 1)
                pts_arr[:, 1] = np.clip(pts_arr[:, 1] + shift_y, 0, img_h - 1)
                roi_hull_points = pts_arr.reshape(-1, 1, 2)
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask, [roi_hull_points], 255)
                roi_mask = mask
            else:
                roi_points.append((x, y))
            _update_display()

        def _positions():
            return {
                "Infarct Lower H": 0,
                "Infarct Lower S": 0,
                "Infarct Lower V": cd68_slider.value(),
                "Infarct Upper H": 255,
                "Infarct Upper S": 255,
                "Infarct Upper V": 255,
                "exclude threshold": exclude_slider.value() if exclude_slider else exclude_start_val,
            }

        def _update_display():
            nonlocal selected_infarct_contour, current_index, save_whole, save_left, save_right, top_contours
            positions = _positions()
            cd68_channel = channel_plane(infarct_img, cd68_color)
            _, mask_cd68 = cv2.threshold(cd68_channel, positions["Infarct Lower V"], 255, cv2.THRESH_BINARY)
            if exclude_color != "none":
                exclude_channel = channel_plane(reference_img, exclude_color)
                _, mask_exclude = cv2.threshold(exclude_channel, positions["exclude threshold"], 255, cv2.THRESH_BINARY_INV)
            else:
                mask_exclude = np.ones_like(mask_cd68, dtype=np.uint8) * 255
            merged_mask = cv2.bitwise_and(mask_cd68, mask_exclude)
            if roi_mask is not None:
                merged_mask = cv2.bitwise_and(merged_mask, roi_mask)
            infarct_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if infarct_contour_detection == "max" and infarct_contours:
                sorted_contours = sorted(infarct_contours, key=cv2.contourArea, reverse=True)
                top_contours = sorted_contours[:5]
                if current_index >= len(top_contours):
                    current_index = 0
                if top_contours:
                    selected_infarct_contour = top_contours[current_index]
            else:
                selected_infarct_contour = None
            view_label, base_img = views[view_index]
            display_frame = base_img.copy()
            if brain_contours is not None and brain_contours.get("whole") is not None:
                cv2.drawContours(display_frame, [brain_contours["whole"]], -1, (255, 255, 255), DRAW_THICKNESS)
            if infarct_contour_detection == "max" and selected_infarct_contour is not None:
                cv2.drawContours(display_frame, [selected_infarct_contour], -1, (0, 255, 0), DRAW_THICKNESS)
            elif infarct_contours:
                cv2.drawContours(display_frame, infarct_contours, -1, (0, 255, 0), max(1, DRAW_THICKNESS - 1))
            if multi_contours:
                for cnt in multi_contours:
                    cv2.drawContours(display_frame, [cnt], -1, (0, 0, 255), DRAW_THICKNESS)
            if roi_mask is not None and roi_hull_points is not None:
                dash_thickness = max(2, DRAW_THICKNESS * 2)
                draw_dashed_polyline(display_frame, roi_hull_points.reshape(-1, 2), (255, 255, 0), thickness=dash_thickness, dash_length=5, gap_length=10)
            elif roi_points:
                dash_thickness = max(2, DRAW_THICKNESS * 2)
                draw_dashed_polyline(display_frame, np.array(roi_points, np.int32).reshape(-1, 2), (255, 255, 0), thickness=dash_thickness, dash_length=5, gap_length=10)
            # Save channel overlays for output
            whole_out = infarct_img.copy()
            left_mask_split = split_image(infarct_img, split_points, np.ones(infarct_img.shape[:2], dtype=np.uint8))[0]
            left_out = cv2.bitwise_and(infarct_img, infarct_img, mask=left_mask_split)
            right_mask_split = split_image(infarct_img, split_points, np.ones(infarct_img.shape[:2], dtype=np.uint8))[1]
            right_out = cv2.bitwise_and(infarct_img, infarct_img, mask=right_mask_split)
            if brain_contours is not None:
                if brain_contours.get("whole") is not None:
                    cv2.drawContours(whole_out, [brain_contours["whole"]], -1, (255, 255, 255), DRAW_THICKNESS)
                if brain_contours.get("left") is not None:
                    cv2.drawContours(left_out, [brain_contours["left"]], -1, (255, 255, 255), DRAW_THICKNESS)
                if brain_contours.get("right") is not None:
                    cv2.drawContours(right_out, [brain_contours["right"]], -1, (255, 255, 255), DRAW_THICKNESS)
            if infarct_contour_detection == "max" and selected_infarct_contour is not None:
                cv2.drawContours(whole_out, [selected_infarct_contour], -1, (0, 255, 0), DRAW_THICKNESS)
                cv2.drawContours(left_out, [selected_infarct_contour], -1, (0, 255, 0), DRAW_THICKNESS)
                cv2.drawContours(right_out, [selected_infarct_contour], -1, (0, 255, 0), DRAW_THICKNESS)
            elif infarct_contours:
                cv2.drawContours(whole_out, infarct_contours, -1, (0, 255, 0), max(1, DRAW_THICKNESS - 1))
                cv2.drawContours(left_out, infarct_contours, -1, (0, 255, 0), max(1, DRAW_THICKNESS - 1))
                cv2.drawContours(right_out, infarct_contours, -1, (0, 255, 0), max(1, DRAW_THICKNESS - 1))
            if multi_contours:
                for cnt in multi_contours:
                    cv2.drawContours(whole_out, [cnt], -1, (0, 0, 255), DRAW_THICKNESS)
                    cv2.drawContours(left_out, [cnt], -1, (0, 0, 255), DRAW_THICKNESS)
                    cv2.drawContours(right_out, [cnt], -1, (0, 0, 255), DRAW_THICKNESS)
            save_whole = whole_out.copy()
            save_left = left_out.copy()
            save_right = right_out.copy()
            self._viewer.set_images([display_frame])
            accept_ready = True
            if infarct_contour_detection == "max":
                accept_ready = selected_infarct_contour is not None or bool(multi_contours)
            self._set_action_state("Accept", enabled=accept_ready, ready=accept_ready)

        def on_exit():
            self._finish_step("EXIT")

        def on_no_infarct():
            positions = _positions()
            self._finish_step({
                "segmented": {"whole": infarct_img.copy(), "left": infarct_img.copy(), "right": infarct_img.copy()},
                "final_positions": positions,
                "selected_infarct_contour": None,
                "no_infarct": True,
            })

        def on_reset_roi():
            nonlocal roi_mask, roi_hull_points
            if roi_fixed:
                return
            roi_points.clear()
            roi_mask = None
            roi_hull_points = None
            _update_display()

        def on_set_roi():
            nonlocal roi_mask, roi_hull_points
            if roi_fixed:
                return
            if len(roi_points) >= 3:
                pts = np.array(roi_points, np.int32)
                hull = cv2.convexHull(pts)
                roi_hull_points = hull
                roi_mask = np.zeros(infarct_img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(roi_mask, [hull], 255)
                roi_points.clear()
                _update_display()
            else:
                pyautogui.alert("Need at least 3 points to set ROI.")

        def on_cycle_contour():
            nonlocal current_index
            if top_contours:
                current_index = (current_index + 1) % len(top_contours)
                _update_display()

        def on_add_contour():
            if infarct_contour_detection == "max" and selected_infarct_contour is not None:
                multi_contours.append(selected_infarct_contour)
                _update_display()

        def on_clear_contours():
            multi_contours.clear()
            _update_display()

        def on_toggle_view():
            nonlocal view_index
            view_index = (view_index + 1) % len(views)
            _update_display()

        def on_accept():
            positions = _positions()
            selected = selected_infarct_contour
            if multi_contours:
                all_contours = multi_contours.copy()
                if selected_infarct_contour is not None:
                    all_contours.append(selected_infarct_contour)
                selected = all_contours
            self._finish_step({
                "segmented": {"whole": save_whole, "left": save_left, "right": save_right},
                "final_positions": positions,
                "selected_infarct_contour": selected,
                "fixed_roi_used": roi_fixed,
                "FIXED_ROI": roi_fixed,
                "roi_mask": roi_mask,
                "no_infarct": False,
            })

        def _sync_cd68_from_slider(val: int):
            if cd68_box.value() != val:
                cd68_box.blockSignals(True)
                cd68_box.setValue(val)
                cd68_box.blockSignals(False)
            _update_display()

        def _sync_cd68_from_box(val: int):
            if cd68_slider.value() != val:
                cd68_slider.blockSignals(True)
                cd68_slider.setValue(val)
                cd68_slider.blockSignals(False)
            _update_display()

        cd68_slider.valueChanged.connect(_sync_cd68_from_slider)
        cd68_box.valueChanged.connect(_sync_cd68_from_box)
        if exclude_slider:
            def _sync_exclude_from_slider(val: int):
                if exclude_box.value() != val:
                    exclude_box.blockSignals(True)
                    exclude_box.setValue(val)
                    exclude_box.blockSignals(False)
                _update_display()

            def _sync_exclude_from_box(val: int):
                if exclude_slider.value() != val:
                    exclude_slider.blockSignals(True)
                    exclude_slider.setValue(val)
                    exclude_slider.blockSignals(False)
                _update_display()

            exclude_slider.valueChanged.connect(_sync_exclude_from_slider)
            exclude_box.valueChanged.connect(_sync_exclude_from_box)
        self._viewer.set_click_handler(on_click)
        self._clear_shortcuts()
        self._register_shortcut("Esc", on_exit)
        self._register_shortcut("N", on_no_infarct)
        self._register_shortcut("R", on_reset_roi)
        self._register_shortcut("S", on_set_roi)
        self._register_shortcut("C", on_cycle_contour)
        self._register_shortcut("M", on_add_contour)
        self._register_shortcut("D", on_clear_contours)
        self._register_shortcut("Z", on_toggle_view)
        self._register_shortcut("A", lambda: on_accept() if self._is_action_enabled("Accept") else None)

        accept_initial_ready = infarct_contour_detection != "max"
        actions = [
            {
                "label": "Accept",
                "callback": on_accept,
                "tooltip": "Accept infarct. Shortcut: A",
                "enabled": accept_initial_ready,
                "ready": accept_initial_ready,
            },
            {"label": "No infarct", "callback": on_no_infarct, "tooltip": "Mark as no infarct. Shortcut: N"},
            {"label": "Cycle contour", "callback": on_cycle_contour, "tooltip": "Cycle contour. Shortcut: C"},
            {"label": "Add contour", "callback": on_add_contour, "tooltip": "Add contour. Shortcut: M"},
            {"label": "Clear contours", "callback": on_clear_contours, "tooltip": "Clear stored contours. Shortcut: D"},
            {"label": "Change view", "callback": on_toggle_view, "tooltip": "Toggle channel view. Shortcut: Z"},
        ]
        if not roi_fixed:
            actions.extend([
                {"label": "Set ROI", "callback": on_set_roi, "tooltip": "Set ROI. Shortcut: S"},
                {"label": "Reset ROI", "callback": on_reset_roi, "tooltip": "Reset ROI. Shortcut: R"},
            ])
        actions.append({"label": "Exit", "callback": on_exit, "tooltip": "Exit step. Shortcut: Esc"})
        shortcuts = [
            ("Left Click", "Add ROI point / reposition ROI"),
            ("C", "Cycle contour"),
            ("M", "Add contour"),
            ("D", "Clear stored contours"),
            ("Z", "Change channel"),
            ("N", "No infarct"),
            ("A", "Accept"),
            ("Esc", "Exit"),
        ]
        if not roi_fixed:
            shortcuts.insert(1, ("S", "Set ROI"))
            shortcuts.insert(2, ("R", "Reset ROI"))
        controls = [QLabel("CD68 threshold"), cd68_slider, cd68_box]
        if exclude_slider:
            controls.extend([QLabel("Exclude threshold"), exclude_slider, exclude_box])
        self._set_actions(
            actions,
            controls,
            "Infarct",
            "Infarct: adjust thresholds and define the infarct region.",
            shortcuts,
            True,
        )
        _update_display()
        return self._wait_for_result()

    def run_pre_roi(self, base_folder: str) -> Optional[dict]:
        """Interactive pre-drawn ROI creation within the workspace."""
        self._set_status("Pre-draw ROI")
        import json
        sections = []
        if not os.path.isdir(base_folder):
            return None
        for animal in sorted(os.listdir(base_folder)):
            subpath = os.path.join(base_folder, animal)
            if not os.path.isdir(subpath):
                continue
            files = [f for f in os.listdir(subpath) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
            for f in files:
                if "reference" in f.lower():
                    ref_path = os.path.join(subpath, f)
                    parts = f.split("reference")
                    if len(parts) == 2:
                        inf_candidate = parts[0] + "infarct" + parts[1]
                        inf_path = os.path.join(subpath, inf_candidate)
                        if os.path.exists(inf_path):
                            sections.append({"ref": ref_path, "inf": inf_path, "animal": animal, "fname": f})
                        else:
                            sections.append({"ref": ref_path, "inf": ref_path, "animal": animal, "fname": f})
                    else:
                        sections.append({"ref": ref_path, "inf": ref_path, "animal": animal, "fname": f})
        if not sections:
            pyautogui.alert("No sections found in the selected folder.")
            return None
        current_idx = 0
        channel_idx = 0
        flip_h = False
        flip_v = False
        roi_points: list = []

        def get_display_image() -> np.ndarray:
            sec = sections[current_idx]
            path = sec["ref"] if channel_idx == 0 else sec["inf"]
            img = cv2.imread(path)
            if img is None:
                return np.zeros((512, 512, 3), dtype=np.uint8)
            if flip_h:
                img = cv2.flip(img, 1)
            if flip_v:
                img = cv2.flip(img, 0)
            return img

        def redraw():
            frame = get_display_image().copy()
            if len(roi_points) >= 2:
                hull = cv2.convexHull(np.array(roi_points, dtype=np.int32))
                pts = hull.reshape(-1, 2)
                dash_thickness = max(2, DRAW_THICKNESS * 2)
                draw_dashed_polyline(frame, pts, (255, 255, 0), thickness=dash_thickness, dash_length=5, gap_length=10)
            sec_info = f"{sections[current_idx]['animal']} | {os.path.basename(sections[current_idx]['fname'])} | {'ref' if channel_idx==0 else 'inf'}"
            cv2.putText(frame, sec_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            self._viewer.set_images([frame])

        def on_click(x, y):
            roi_points.append((x, y))
            redraw()

        def on_channel():
            nonlocal channel_idx
            channel_idx = 1 - channel_idx
            redraw()

        def on_next():
            nonlocal current_idx
            current_idx = (current_idx + 1) % len(sections)
            roi_points.clear()
            redraw()

        def on_reset():
            roi_points.clear()
            redraw()

        def on_flip_h():
            nonlocal flip_h
            flip_h = not flip_h
            roi_points.clear()
            redraw()

        def on_flip_v():
            nonlocal flip_v
            flip_v = not flip_v
            roi_points.clear()
            redraw()

        def on_exit():
            self._finish_step(None)

        def on_save():
            if len(roi_points) < 3:
                pyautogui.alert("ROI requires at least 3 points to form a polygon.")
                return
            img = get_display_image()
            h, w = img.shape[:2]
            hull = cv2.convexHull(np.array(roi_points, dtype=np.int32)).reshape(-1, 2)
            poly_norm = [(float(x) / w, float(y) / h) for (x, y) in hull]
            cx = np.mean([p[0] for p in hull])
            cy = np.mean([p[1] for p in hull])
            centroid_norm = (cx / w, cy / h)
            roi_name, ok = QInputDialog.getText(self, "ROI Name", "Enter name for this ROI:")
            if not ok or roi_name.strip() == "":
                pyautogui.alert("ROI name cannot be empty. Try again.")
                return
            roi_name = roi_name.strip()
            fname = f"{roi_name}_ROI.json"
            file_path = os.path.join(base_folder, fname)
            data = {
                "name": roi_name,
                "poly_norm": poly_norm,
                "centroid_norm": centroid_norm,
                "flip_h": flip_h,
                "flip_v": flip_v,
                "file_path": file_path,
            }
            try:
                with open(file_path, "w") as fjson:
                    json.dump(data, fjson)
            except Exception as e:
                pyautogui.alert(f"Error saving ROI: {e}")
                data = None
            self._finish_step(data)

        self._viewer.set_click_handler(on_click)
        self._clear_shortcuts()
        self._register_shortcut("C", on_channel)
        self._register_shortcut("N", on_next)
        self._register_shortcut("R", on_reset)
        self._register_shortcut("H", on_flip_h)
        self._register_shortcut("V", on_flip_v)
        self._register_shortcut("S", on_save)
        self._register_shortcut("Esc", on_exit)
        actions = [
            {"label": "Save ROI", "callback": on_save, "tooltip": "Save ROI. Shortcut: S"},
            {"label": "Change channel", "callback": on_channel, "tooltip": "Toggle channel. Shortcut: C"},
            {"label": "Next section", "callback": on_next, "tooltip": "Next section. Shortcut: N"},
            {"label": "Reset points", "callback": on_reset, "tooltip": "Reset ROI points. Shortcut: R"},
            {"label": "Flip H", "callback": on_flip_h, "tooltip": "Flip horizontally. Shortcut: H"},
            {"label": "Flip V", "callback": on_flip_v, "tooltip": "Flip vertically. Shortcut: V"},
            {"label": "Exit", "callback": on_exit, "tooltip": "Exit without saving. Shortcut: Esc"},
        ]
        shortcuts = [
            ("Left Click", "Add ROI point"),
            ("S", "Save ROI"),
            ("C", "Change channel"),
            ("N", "Next section"),
            ("R", "Reset points"),
            ("H", "Flip horizontal"),
            ("V", "Flip vertical"),
            ("Esc", "Exit"),
        ]
        self._set_actions(
            actions,
            [],
            "Pre-draw ROI",
            "Pre-draw ROI: click to place ROI vertices, then save.",
            shortcuts,
            True,
        )
        redraw()
        return self._wait_for_result()


class PreprocessTab(QWidget):
    """Tab for preprocessing slides into sections.

    This tab handles the first stage of the pipeline: taking full slide
    images and dividing them into separate section images based on a
    contour detected from a reference channel. After preprocessing
    finishes it updates the analysis tab with the output folder so
    users don't need to manually browse for the sections.
    """

    def __init__(self, analysis_tab: Optional['AnalysisTab'] = None, parent=None):
        super().__init__(parent)
        # Keep a reference to the analysis tab so we can update its base path
        self.analysis_tab = analysis_tab
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        # Input folder selection
        input_card, input_layout = create_card("1) Input slides")
        description = QLabel("Split whole-slide images into individual section images.")
        description.setWordWrap(True)
        input_layout.addWidget(description)
        # Button for selecting the slides folder. The button text will update to show the path once selected.
        self.select_btn = QPushButton("Select Slides Folder")
        self.select_btn.setObjectName("secondaryAction")
        self.select_btn.clicked.connect(self.select_folder)
        self.folder_path = ""
        input_layout.addWidget(self.select_btn)
        layout.addWidget(input_card)
        # Parameters
        params_card, params_card_layout = create_card("2) Section detection parameters")
        params_layout = QHBoxLayout()
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(6)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10000000)
        self.min_area_spin.setValue(300000)
        self.min_area_label = QLabel("Minimum area:")
        # Help button for minimum area
        min_area_help = QToolButton()
        min_area_help.setText("?")
        min_area_help.setStyleSheet("background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;")
        min_area_help.setToolTip(
            "Minimum pixel area for section detection.\n"
            "Default 300,000 works for forebrain sections. Reduce if your sections are small or slide is under-exposed."
        )
        # Order: help, label, spin box
        params_layout.addWidget(min_area_help)
        params_layout.addWidget(self.min_area_label)
        params_layout.addWidget(self.min_area_spin)
        # Show help on click
        min_area_help.clicked.connect(lambda: QMessageBox.information(self, "Minimum area", min_area_help.toolTip().replace("\\n", "\n")))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 2000)
        self.padding_spin.setValue(20)
        self.padding_label = QLabel("Padding:")
        # Help button for padding
        padding_help = QToolButton()
        padding_help.setText("?")
        padding_help.setStyleSheet("background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;")
        padding_help.setToolTip(
            "Padding in pixels added around each extracted section.\n"
            "Usually should not be changed."
        )
        # Order: help, label, spin box
        params_layout.addWidget(padding_help)
        params_layout.addWidget(self.padding_label)
        params_layout.addWidget(self.padding_spin)
        padding_help.clicked.connect(lambda: QMessageBox.information(self, "Padding", padding_help.toolTip().replace("\\n", "\n")))
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(0, 255)
        self.thresh_spin.setValue(40)
        self.thresh_label = QLabel("Brain threshold:")
        # Help button for intensity threshold
        thresh_help = QToolButton()
        thresh_help.setText("?")
        thresh_help.setStyleSheet("background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;")
        thresh_help.setToolTip(
            "HSV value threshold used to detect section outlines.\n"
            "Increase if too many sections are detected; decrease if sections are missed."
        )
        # Order: help, label, spin box
        params_layout.addWidget(thresh_help)
        params_layout.addWidget(self.thresh_label)
        params_layout.addWidget(self.thresh_spin)
        thresh_help.clicked.connect(lambda: QMessageBox.information(self, "Brain threshold", thresh_help.toolTip().replace("\\n", "\n")))
        # Scale (pixel per micron) input. Optional; leave blank to report areas in pixels.
        self.scale_label = QLabel("Scale (px/um):")
        self.scale_edit = QLineEdit()
        self.scale_edit.setPlaceholderText("optional")
        # Help button for scale
        scale_help = QToolButton()
        scale_help.setText("?")
        scale_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        scale_help.setToolTip(
            "Pixel-to-micron scale used for converting pixel areas to um.\n"
            "When provided, the app will adjust the scale if the image is downsampled during preprocessing."
        )
        # Append scale widgets after threshold widgets
        params_layout.addWidget(scale_help)
        params_layout.addWidget(self.scale_label)
        params_layout.addWidget(self.scale_edit)
        scale_help.clicked.connect(lambda: QMessageBox.information(self, "Scale (px/um)", scale_help.toolTip().replace("\\n", "\n")))
        params_card_layout.addLayout(params_layout)
        layout.addWidget(params_card)
        # Keywords
        keywords_card, keywords_card_layout = create_card("3) Filename keywords")
        kw_layout = QHBoxLayout()
        kw_layout.setContentsMargins(0, 0, 0, 0)
        kw_layout.setSpacing(6)
        self.contour_kw_edit = QLineEdit("merge")
        self.contour_kw_label = QLabel("Reference keyword:")
        # Help button for reference keyword
        ref_help = QToolButton()
        ref_help.setText("?")
        ref_help.setStyleSheet("background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;")
        ref_help.setToolTip(
            "Keyword in filenames identifying the reference channel (whole brain).\n"
            "Typically 'overlay', 'merge', 'NeuN' or similar."
        )
        # Order: help, label, edit
        kw_layout.addWidget(ref_help)
        kw_layout.addWidget(self.contour_kw_label)
        kw_layout.addWidget(self.contour_kw_edit)
        ref_help.clicked.connect(lambda: QMessageBox.information(self, "Reference keyword", ref_help.toolTip().replace("\\n", "\n")))
        self.infarct_kw_edit = QLineEdit("CD68")
        self.infarct_kw_label = QLabel("Infarct keyword:")
        # Help button for infarct keyword
        infarct_help = QToolButton()
        infarct_help.setText("?")
        infarct_help.setStyleSheet("background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;")
        infarct_help.setToolTip(
            "Keyword in filenames identifying the infarct channel.\n"
            "Typically 'CD68'."
        )
        # Order: help, label, edit
        kw_layout.addWidget(infarct_help)
        kw_layout.addWidget(self.infarct_kw_label)
        kw_layout.addWidget(self.infarct_kw_edit)
        infarct_help.clicked.connect(lambda: QMessageBox.information(self, "Infarct keyword", infarct_help.toolTip().replace("\\n", "\n")))
        keywords_card_layout.addLayout(kw_layout)
        layout.addWidget(keywords_card)
        # Process button
        run_card, run_layout = create_card("4) Run preprocessing")
        self.process_btn = QPushButton("Run Preprocessing")
        self.process_btn.clicked.connect(self.run_preprocess)
        self.process_btn.setObjectName("primaryAction")
        run_layout.addWidget(self.process_btn)
        layout.addWidget(run_card)
        layout.addStretch(1)
        self._preprocess_dirty = False
        self._wire_preprocess_dirty_signals()
        self.setLayout(layout)
        self.setMinimumHeight(320)

    def _wire_preprocess_dirty_signals(self) -> None:
        """Mark preprocessing params dirty so the button resets when inputs change."""
        def mark_dirty():
            self._preprocess_dirty = True
            self._reset_preprocess_button()
        self.min_area_spin.valueChanged.connect(mark_dirty)
        self.padding_spin.valueChanged.connect(mark_dirty)
        self.thresh_spin.valueChanged.connect(mark_dirty)
        self.scale_edit.textChanged.connect(mark_dirty)
        self.contour_kw_edit.textChanged.connect(mark_dirty)
        self.infarct_kw_edit.textChanged.connect(mark_dirty)

    def _reset_preprocess_button(self) -> None:
        self.process_btn.setObjectName("primaryAction")
        self.process_btn.setText("Run Preprocessing")
        self.process_btn.style().unpolish(self.process_btn)
        self.process_btn.style().polish(self.process_btn)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Slides Folder", "")
        if folder:
            self.folder_path = folder
            # Update button text with selected path
            self.select_btn.setText(f"Slides Folder: {folder}")
            self._preprocess_dirty = True
            self._reset_preprocess_button()

    def run_preprocess(self):
        if not self.folder_path:
            QMessageBox.warning(self, "No folder", "Please select an input folder first.")
            return
        # Update button state while preprocessing
        self.process_btn.setObjectName("secondaryAction")
        self.process_btn.setText("Running Preprocessing (this may take awhile)")
        self.process_btn.style().unpolish(self.process_btn)
        self.process_btn.style().polish(self.process_btn)
        # Retrieve user parameters
        min_area = self.min_area_spin.value()
        padding = self.padding_spin.value()
        thresh = self.thresh_spin.value()
        contour_kw = self.contour_kw_edit.text()
        infarct_kw = self.infarct_kw_edit.text()
        # Build HSV bounds for brain detection
        hsv_bounds = {
            'lower_H': 0,
            'lower_S': 0,
            'lower_V': thresh,
            'upper_H': 255,
            'upper_S': 255,
            'upper_V': 255,
        }
        # Parse optional pixel scale from text box
        pixel_scale = None
        scale_text = self.scale_edit.text().strip()
        if scale_text:
            try:
                pixel_scale = float(scale_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid scale", "Scale must be a numeric value if provided.")
                return
        # Determine output folder for cropped images
        output_folder = os.path.join(self.folder_path, "preprocessed_sections")
        QApplication.processEvents()
        try:
            # Run preprocessing and collect metadata rows
            count, log_str, metadata_rows = process_folder(
                self.folder_path,
                output_folder,
                hsv_bounds,
                min_area,
                padding,
                contour_kw,
                infarct_kw,
                thresh,
                pixel_scale,
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error during preprocessing: {e}\n{traceback.format_exc()}",
            )
            self._reset_preprocess_button()
            return
        # Write or update metadata CSV inside the output folder. The file name
        # encodes the ACTUAL preprocess inputs -- the reference (contour) keyword
        # and the infarct/detection keyword -- so it's clear how the data was
        # processed, e.g. "preprocessed_ref-merge_detect-CD68.csv".
        import pandas as pd
        import re as _re

        def _slug(value: str, fallback: str) -> str:
            slug = _re.sub(r"[^0-9A-Za-z]+", "", (value or "").strip())
            return slug if slug else fallback

        ref_label = _slug(contour_kw, "ref")
        det_label = _slug(infarct_kw, "detect")
        metadata_filename = f"preprocessed_ref-{ref_label}_detect-{det_label}.csv"
        metadata_path = os.path.join(output_folder, metadata_filename)
        df_new = pd.DataFrame(metadata_rows)
        # Merge with existing metadata if present to avoid duplicate entries
        if os.path.exists(metadata_path):
            try:
                df_old = pd.read_csv(metadata_path)
                key_cols = ["animal_id", "section_id"]
                merge_keys = df_new[key_cols].apply(lambda row: f"{row['animal_id']}|{row['section_id']}", axis=1).tolist()
                mask = df_old[key_cols].apply(lambda row: f"{row['animal_id']}|{row['section_id']}", axis=1).isin(merge_keys)
                df_old = df_old[~mask]
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new
        df_combined.to_csv(metadata_path, index=False)
        # After preprocessing, automatically populate the analysis tab if available
        if self.analysis_tab is not None:
            self.analysis_tab.base_path = output_folder
            # Update analysis tab button text to reflect automatically selected sections folder
            self.analysis_tab.base_btn.setText(f"Sections Folder: {output_folder}")
            # Synchronise the brain threshold spin box in the analysis tab with the intensity threshold used during preprocessing
            try:
                self.analysis_tab.section_thresh_spin.setValue(thresh)
            except Exception:
                pass
        # Inform the user about completion
        self.process_btn.setObjectName("primaryAction")
        self.process_btn.setText(
            f"Preprocessing Complete: {count} sections extracted from {len([f for f in os.listdir(self.folder_path) if f.lower().endswith(('.tif', '.tiff'))])} images"
        )
        self.process_btn.style().unpolish(self.process_btn)
        self.process_btn.style().polish(self.process_btn)
        self._preprocess_dirty = False
        QMessageBox.information(
            self,
            "Preprocessing complete",
            (
                f"Preprocessing finished. Extracted {count} sections.\n"
                f"Output saved in {output_folder}.\n\n"
                "Please review the cropped sections in the output folder against the original slide images.\n"
                "If too many or too few sections were detected, adjust the Intensity threshold and run preprocessing again.\n\n"
                "Note: You may want to delete the 'preprocessed_sections' folder before re-running to reduce clutter."
            ),
        )


class AnalysisTab(QWidget):
    """Tab for running infarct analysis on preprocessed sections."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Set by MainWindow so a finished analysis can pre-fill the Quantify tab.
        self.quantify_tab = None
        self.last_results_csv_path = ""
        params_layout = QVBoxLayout()
        params_layout.setSizeConstraint(QLayout.SetMinimumSize)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(6)
        params_page = QWidget()
        # Description for the analysis step
        input_card, input_layout = create_card("1) Sections folder")
        description = QLabel("Perform infarct quantification on individual section images.")
        description.setWordWrap(True)
        input_layout.addWidget(description)
        # Base directory selection button. Shows selected path once chosen.
        self.base_btn = QPushButton("Select Preprocessed Sections Folder")
        self.base_btn.setObjectName("secondaryAction")
        self.base_btn.clicked.connect(self.select_base)
        self.base_path = ""
        input_layout.addWidget(self.base_btn)
        params_layout.addWidget(input_card)
        # Parameters for analysis
        param_card, param_card_layout = create_card("2) Area calculation")
        param_layout = QHBoxLayout()
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(6)
        self.method_label = QLabel("Method:")
        self.method_edit = QLineEdit("contour")
        # Add help button for method
        method_help = QToolButton()
        method_help.setText("?")
        method_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        method_help.setToolTip(
            "Area calculation method.\n"
            "\u2022 contour: use contour areas of brain and infarct.\n"
            "\u2022 mask: use pixel counts in masks."
        )
        # Order: help, label, edit
        param_layout.addWidget(method_help)
        param_layout.addWidget(self.method_label)
        param_layout.addWidget(self.method_edit)
        self.detection_label = QLabel("Detection:")
        self.detection_edit = QLineEdit("max")
        detection_help = QToolButton()
        detection_help.setText("?")
        detection_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        detection_help.setToolTip(
            "Infarct contour detection strategy.\n"
            "\u2022 max: choose the largest detected contour.\n"
            "\u2022 all: use all detected contours."
        )
        param_layout.addWidget(detection_help)
        param_layout.addWidget(self.detection_label)
        param_layout.addWidget(self.detection_edit)
        param_card_layout.addLayout(param_layout)
        params_layout.addWidget(param_card)
        # Additional thresholds and color selections
        thresh_card, thresh_card_layout = create_card("3) Thresholds and channels")
        thresh_layout = QVBoxLayout()
        thresh_layout.setContentsMargins(0, 0, 0, 0)
        thresh_layout.setSpacing(14)
        thresh_top_row = QHBoxLayout()
        thresh_top_row.setContentsMargins(0, 0, 0, 0)
        thresh_top_row.setSpacing(6)
        thresh_bottom_row = QHBoxLayout()
        thresh_bottom_row.setContentsMargins(0, 0, 0, 0)
        thresh_bottom_row.setSpacing(6)
        self.section_thresh_label = QLabel("Brain threshold:")
        self.section_thresh_spin = QSpinBox()
        self.section_thresh_spin.setRange(0, 255)
        self.section_thresh_spin.setValue(40)
        section_help = QToolButton()
        section_help.setText("?")
        section_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        section_help.setToolTip(
            "Threshold applied to the reference image to outline the brain.\n"
            "Increase if the brain outline is faint; decrease if too many artefacts are detected."
        )
        # Order: help, label, spin
        thresh_top_row.addWidget(section_help)
        thresh_top_row.addWidget(self.section_thresh_label)
        thresh_top_row.addWidget(self.section_thresh_spin)
        # CD68 color selection
        self.cd68_label = QLabel("CD68 channel:")
        self.cd68_combo = QComboBox()
        self.cd68_combo.addItems(["red", "green", "blue", "grayscale"])
        self.cd68_combo.setCurrentText("red")
        cd68_help = QToolButton()
        cd68_help.setText("?")
        cd68_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        cd68_help.setToolTip(
            "Color channel used for the CD68 infarct marker.\n"
            "Choose the color corresponding to the channel where CD68 staining is present.\n"
            "Use 'grayscale' for grayscale images (combines all channels into luminance)."
        )
        thresh_top_row.addWidget(cd68_help)
        thresh_top_row.addWidget(self.cd68_label)
        thresh_top_row.addWidget(self.cd68_combo)
        # Exclude colour selection
        self.exclude_label = QLabel("exclude channel:")
        self.exclude_combo = QComboBox()
        # Allow 'None' option so exclusion can be disabled
        self.exclude_combo.addItems(["red", "green", "blue", "grayscale", "none"])
        self.exclude_combo.setCurrentText("none")
        exclude_help = QToolButton()
        exclude_help.setText("?")
        exclude_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        exclude_help.setToolTip(
            "Color channel used to exclude non-infarct signal (e.g. 'GFAP', 'ChR2').\n"
            "Choose the colour corresponding to the channel you want to exclude from the infarct segmentation.\n"
            "Select 'none' to disable exclusion entirely."
        )
        thresh_bottom_row.addWidget(exclude_help)
        thresh_bottom_row.addWidget(self.exclude_label)
        thresh_bottom_row.addWidget(self.exclude_combo)

        # Starting intensity thresholds for CD68 and exclusion channels
        self.cd68_start_label = QLabel("CD68 intensity:")
        self.cd68_start_spin = QSpinBox()
        self.cd68_start_spin.setRange(0, 255)
        self.cd68_start_spin.setValue(100)
        cd68_start_help = QToolButton()
        cd68_start_help.setText("?")
        cd68_start_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        cd68_start_help.setToolTip(
            "Starting intensity threshold for the CD68 (infarct) channel.\n"
            "This value is used to initialise the slider in the infarct detection step. You can adjust it during analysis."
        )
        self.exclude_start_label = QLabel("Exclude intensity:")
        self.exclude_start_spin = QSpinBox()
        self.exclude_start_spin.setRange(0, 255)
        self.exclude_start_spin.setValue(175)
        exclude_start_help = QToolButton()
        exclude_start_help.setText("?")
        exclude_start_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        exclude_start_help.setToolTip(
            "Starting intensity threshold for the exclusion channel.\n"
            "Only pixels below this value in the exclusion channel will be considered for infarct detection."
        )
        # Add to layout: help, label, spin for CD68 and exclusion starting intensities
        thresh_top_row.addWidget(cd68_start_help)
        thresh_top_row.addWidget(self.cd68_start_label)
        thresh_top_row.addWidget(self.cd68_start_spin)
        thresh_bottom_row.addWidget(exclude_start_help)
        thresh_bottom_row.addWidget(self.exclude_start_label)
        thresh_bottom_row.addWidget(self.exclude_start_spin)
        thresh_top_row.addStretch(1)
        thresh_bottom_row.addStretch(1)
        thresh_layout.addLayout(thresh_top_row)
        thresh_layout.addLayout(thresh_bottom_row)
        thresh_card_layout.addLayout(thresh_layout)
        params_layout.addWidget(thresh_card)

        # --- Background percentile and pre-drawn ROI controls ---
        # Create a horizontal layout for the background percentile and pre-drawn ROI components
        roi_card, roi_card_layout = create_card("4) ROI and background")
        roi_layout = QHBoxLayout()
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(6)
        # Background percentile help button and entry
        self.bg_percent_help = QToolButton()
        self.bg_percent_help.setText("?")
        self.bg_percent_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        self.bg_percent_help.setToolTip(
            "Percentage of the lowest-intensity pixels within the brain contour used to \n"
            "calculate the background intensity. A value of 10 means the mean of the\n"
            "darkest 10% of pixels will be used as the background reference."
        )
        # Label for the background percentile
        self.bg_percent_label = QLabel("Background %:")
        # Text entry allowing the user to override the default percentile; leave blank for 10
        self.bg_percent_edit = QLineEdit()
        self.bg_percent_edit.setPlaceholderText("10")
        # When clicking the help button, show the tooltip text in an information dialog
        self.bg_percent_help.clicked.connect(lambda: QMessageBox.information(self, "Background %", self.bg_percent_help.toolTip().replace("\n", "\n")))
        # Help toolbutton explaining pre-ROI functionality
        self.pre_roi_help = QToolButton()
        self.pre_roi_help.setText("?")
        self.pre_roi_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        self.pre_roi_help.setToolTip(
            "Optionally predefine a fixed ROI before analysis.\n"
            "Click 'Pre-draw ROI' to launch an interactive window where you can\n"
            "draw an ROI on any section. The ROI will be saved and applied to all\n"
            "sections during infarct detection. You can reposition the ROI during\n"
            "analysis, but its shape remains fixed."
        )
        # Label
        self.pre_roi_label = QLabel("Pre-drawn ROI:")
        # Read-only line edit to display the currently selected ROI name (empty if none)
        self.pre_roi_field = QLineEdit()
        # Keep the line edit read-only so the user cannot type arbitrary text. However,
        # we still want to respond to mouse clicks on this field to allow loading
        # an existing ROI specification from a JSON file. We override the
        # mousePressEvent of the QLineEdit to trigger a file selection dialog.
        self.pre_roi_field.setReadOnly(True)
        self.pre_roi_field.setPlaceholderText("None")
        # Button to launch pre-draw ROI workflow
        self.pre_roi_btn = QPushButton("Pre-draw ROI")
        self.pre_roi_btn.setObjectName("accentAction")
        self.pre_roi_btn.clicked.connect(self.pre_draw_roi_action)
        # Add background percentile widgets first
        roi_layout.addWidget(self.bg_percent_help)
        roi_layout.addWidget(self.bg_percent_label)
        roi_layout.addWidget(self.bg_percent_edit)
        # Assemble layout: help, label, field, button for pre-drawn ROI
        roi_layout.addWidget(self.pre_roi_help)
        roi_layout.addWidget(self.pre_roi_label)
        roi_layout.addWidget(self.pre_roi_field)
        roi_layout.addWidget(self.pre_roi_btn)
        roi_card_layout.addLayout(roi_layout)
        params_layout.addWidget(roi_card)
        # Storage for pre-drawn ROI data (loaded from JSON or created via workflow)
        self.pre_draw_roi_data: Optional[dict] = None

        # Bind a custom mouse press handler to the pre-drawn ROI field. When the user
        # clicks on the (read-only) text field, a file dialog will open allowing
        # selection of an existing pre-drawn ROI JSON. If a valid JSON is loaded,
        # the ROI name is displayed in the field and stored in `self.pre_draw_roi_data`.
        def _handle_pre_roi_field_click(event):
            # Only respond to left-clicks; ignore other buttons (e.g. right-click)
            if event.button() == Qt.LeftButton:
                # Ensure a base folder is selected so that the app knows where to
                # look for section images and to provide a sensible starting
                # directory for the file dialog. If no base folder is set,
                # instruct the user to select one first.
                if not getattr(self, 'base_path', ''):
                    QMessageBox.warning(self, "No folder", "Please select a sections folder first.")
                    # Call the original handler to maintain cursor focus, if defined
                    return
                # Compose a starting directory: use the current base folder by
                # default, but fall back to the user's home directory if the
                # base folder is invalid.
                start_dir = self.base_path if os.path.isdir(self.base_path) else os.path.expanduser('~')
                # Open a file dialog filtered for JSON files that might contain
                # pre-drawn ROIs. The user can still choose any file type.
                fname, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Pre-drawn ROI JSON",
                    start_dir,
                    "ROI JSON Files (*.json);;All Files (*)",
                )
                if fname:
                    try:
                        import json
                        with open(fname, 'r') as fj:
                            data = json.load(fj)
                        # Basic validation: ensure the loaded JSON has the required
                        # keys for an ROI specification. At minimum we expect a
                        # 'name' and 'poly_norm'. If these keys are missing,
                        # warn the user and do not update the current ROI.
                        if not isinstance(data, dict) or 'name' not in data or 'poly_norm' not in data:
                            QMessageBox.warning(
                                self,
                                "Invalid ROI file",
                                "The selected file does not appear to be a valid ROI specification."
                            )
                            return
                        # Store the loaded ROI data and display its name. Also
                        # record the file path so that downstream code knows
                        # where the ROI came from.
                        data['file_path'] = fname
                        self.pre_draw_roi_data = data
                        self.pre_roi_field.setText(data.get('name', ''))
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to load ROI: {e}")
                        # Do not modify existing ROI data on failure
                        return
            # Fall back to the default QLineEdit mousePressEvent behaviour to ensure
            # proper focus handling. Without this, the line edit may not update
            # its cursor or selection state correctly.
            return QLineEdit.mousePressEvent(self.pre_roi_field, event)

        # Assign the custom click handler to the QLineEdit instance. This is
        # necessary because QLineEdit does not emit a clicked signal by default.
        self.pre_roi_field.mousePressEvent = _handle_pre_roi_field_click

        # Connect help buttons to show detailed info on click
        method_help.clicked.connect(lambda: QMessageBox.information(self, "Method", method_help.toolTip().replace("\\n", "\n")))
        detection_help.clicked.connect(lambda: QMessageBox.information(self, "Detection", detection_help.toolTip().replace("\\n", "\n")))
        section_help.clicked.connect(lambda: QMessageBox.information(self, "Brain threshold", section_help.toolTip().replace("\\n", "\n")))
        cd68_help.clicked.connect(lambda: QMessageBox.information(self, "CD68 channel", cd68_help.toolTip().replace("\\n", "\n")))
        exclude_help.clicked.connect(lambda: QMessageBox.information(self, "exclude channel", exclude_help.toolTip().replace("\\n", "\n")))
        cd68_start_help.clicked.connect(lambda: QMessageBox.information(self, "CD68 intensity", cd68_start_help.toolTip().replace("\\n", "\n")))
        exclude_start_help.clicked.connect(lambda: QMessageBox.information(self, "Exclude intensity", exclude_start_help.toolTip().replace("\\n", "\n")))
        # Run button
        run_card, run_layout = create_card("5) Run analysis")
        self.run_btn = QPushButton("Run Analysis on Folder")
        self.run_btn.setObjectName("primaryAction")
        self.run_btn.clicked.connect(self.run_analysis)
        run_layout.addWidget(self.run_btn)
        params_layout.addWidget(run_card)
        params_layout.addStretch(1)
        params_page.setLayout(params_layout)
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.params_scroll.setWidget(params_page)
        # Workspace (interactive viewer) and log panel
        self.page_stack = QStackedWidget()
        self.workspace = AnalysisWorkspace()
        self.page_stack.addWidget(self.params_scroll)
        self.page_stack.addWidget(self.workspace)
        self.page_stack.setCurrentIndex(0)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)
        main_layout.addWidget(self.page_stack, 1)
        self.setMinimumHeight(320)

    def show_workspace(self, show: bool) -> None:
        """Toggle between parameter form and interactive workspace."""
        self.page_stack.setCurrentIndex(1 if show else 0)
        if not show:
            try:
                self.workspace.set_progress("")
            except Exception:
                pass

    def select_base(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Sections Folder", "")
        if folder:
            self.base_path = folder
            # Update button text with selected path
            self.base_btn.setText(f"Sections Folder: {folder}")
            # Automatically load a previously saved pre-drawn ROI if present
            try:
                import json
                # Search for any *_ROI.json in the selected folder
                roi_files = [f for f in os.listdir(folder) if f.lower().endswith('_roi.json')]
                if roi_files:
                    roi_path = os.path.join(folder, roi_files[0])
                    with open(roi_path, 'r') as fj:
                        data = json.load(fj)
                        self.pre_draw_roi_data = data
                        self.pre_roi_field.setText(data.get('name', ''))
                else:
                    # Reset ROI field if none found
                    self.pre_draw_roi_data = None
                    self.pre_roi_field.setText("")
            except Exception:
                # Ignore errors silently to avoid blocking folder selection
                self.pre_draw_roi_data = None
                self.pre_roi_field.setText("")

    def run_analysis(self):
        if not self.base_path:
            QMessageBox.warning(self, "No folder", "Please select a sections folder first.")
            return
        method = self.method_edit.text().strip()
        detection = self.detection_edit.text().strip()
        if method not in ("contour", "mask"):
            QMessageBox.warning(self, "Invalid method", "Method must be 'contour' or 'mask'.")
            return
        if detection not in ("max", "all"):
            QMessageBox.warning(self, "Invalid detection", "Detection must be 'max' or 'all'.")
            return
        section_thresh = self.section_thresh_spin.value()
        # Retrieve colors from combo boxes
        cd68_color = self.cd68_combo.currentText().strip().lower()
        exclude_color = self.exclude_combo.currentText().strip().lower()
        # Retrieve starting intensity thresholds from spin boxes
        cd68_start_val = self.cd68_start_spin.value()
        exclude_start_val = self.exclude_start_spin.value()
        # Determine background percentile. Use default of 10 if left blank. Validate range (0, 100].
        bg_percent = 10.0
        bg_text = self.bg_percent_edit.text().strip()
        if bg_text:
            try:
                val = float(bg_text)
                # Require a sensible range for percentiles
                if val <= 0 or val > 100:
                    QMessageBox.warning(self, "Invalid background %", "Background percent must be between 0 and 100.")
                    return
                bg_percent = val
            except ValueError:
                QMessageBox.warning(self, "Invalid background %", "Background percent must be a numeric value.")
                return
        QApplication.processEvents()
        # --- Data-driven channel validation gate --------------------------------
        # Before processing every section, confirm the chosen CD68/exclude colours
        # map to usable planes of the ACTUAL images: hard-stop on non-3-channel
        # images, warn (Proceed/Cancel) if a selected plane has no signal. Runs
        # once on the first section. See analysis.validate_channel_selection.
        import re as _re_chk
        _sample_ref = None
        try:
            for _af in sorted(os.listdir(self.base_path)):
                _sf = os.path.join(self.base_path, _af)
                if not os.path.isdir(_sf):
                    continue
                _refs = sorted(
                    fn for fn in os.listdir(_sf)
                    if fn.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
                    and "reference" in fn.lower()
                )
                if _refs:
                    _sample_ref = os.path.join(_sf, _refs[0])
                    break
        except Exception:
            _sample_ref = None
        if _sample_ref is not None:
            _inf_name = _re_chk.sub(r"reference", "infarct", os.path.basename(_sample_ref), flags=_re_chk.IGNORECASE)
            _inf_path = os.path.join(os.path.dirname(_sample_ref), _inf_name)
            _has_inf = os.path.exists(_inf_path)
            _ref_img = cv2.imread(_sample_ref)
            _inf_img = cv2.imread(_inf_path) if _has_inf else _ref_img
            if _inf_img is None:
                _inf_img = _ref_img
            # Only validate if the sample read OK; the per-section step reports read errors itself.
            if _ref_img is not None:
                _warnings = []
                try:
                    _, _w1 = validate_channel_selection(
                        _inf_img, cd68_color,
                        tiff_path=(_inf_path if _has_inf else _sample_ref),
                        role="CD68 channel",
                    )
                    if _w1:
                        _warnings.append(_w1)
                    _, _w2 = validate_channel_selection(
                        _ref_img, exclude_color, tiff_path=_sample_ref, role="exclude channel",
                    )
                    if _w2:
                        _warnings.append(_w2)
                except ChannelValidationError as _cve:
                    QMessageBox.critical(self, "Cannot use selected channel", str(_cve))
                    return
                if _warnings:
                    _resp = QMessageBox.warning(
                        self, "Channel check",
                        "\n\n".join(_warnings) + "\n\nProceed anyway?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                    )
                    if _resp != QMessageBox.Yes:
                        return
        # ------------------------------------------------------------------------
        # Switch into the interactive workspace for analysis steps
        self.show_workspace(True)
        # Aggregated results
        import datetime
        import pandas as pd
        import re
        CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
        # Compose a descriptive filename for the results CSV: the detection (CD68)
        # channel, the exclusion channel (or "none"), and the analysis method. This
        # makes the processing parameters clear and prevents overwriting between runs,
        # e.g. "results_detect-red_exclude-none_contour.csv".
        final_csv_name = f"results_detect-{cd68_color}_exclude-{exclude_color}_{method}.csv"
        final_csv_path = os.path.join(self.base_path, final_csv_name)
        # Prepare aggregated DataFrame with updated columns (do not include hemisphere diff columns)
        if os.path.exists(final_csv_path):
            aggregated_df = pd.read_csv(final_csv_path)
            # Drop any obsolete hemisphere difference columns
            drop_cols = [col for col in ["hemisphere_diff", "hemisphere_diff_pct"] if col in aggregated_df.columns]
            if drop_cols:
                aggregated_df.drop(columns=drop_cols, inplace=True)
            # Ensure section_id is string for matching
            if "section_id" in aggregated_df.columns:
                aggregated_df["section_id"] = aggregated_df["section_id"].astype(str)
        else:
            aggregated_df = pd.DataFrame()
        # Ensure aggregated_df has the expected columns; add missing ones with empty values
        expected_cols = [
            "animal_id",
            "section_id",
            "filename",
            "whole_area_px",
            "left_area_px",
            "right_area_px",
            "infarct_area_px",
            "whole_area_um^2",
            "left_area_um^2",
            "right_area_um^2",
            "infarct_area_um^2",
            "infarct_area_positive",
            "infarct_area_intensity_avg",
            # mean intensity of threshold-positive pixels
            "infarct_area_positive_intensity_avg",
            # new columns: background intensity and normalized infarct intensity
            "background_intensity",
            "infarct_intensity_avg_normalized",
            # normalized mean intensity of threshold-positive pixels
            "infarct_intensity_positive_avg_normalized",
            "brain_outline_threshold",
            "CD68_threshold",
            "exclude_threshold",
            "pixel_to_um_scale",
            "date_analyzed",
                # Indicates whether a pre-drawn (fixed) ROI was used for this section
                "FIXED_ROI",
        ]
        for col in expected_cols:
            if col not in aggregated_df.columns:
                aggregated_df[col] = ""
        exit_requested = False
        # Load preprocessing metadata if available for pixel-to-micron conversion.
        # Match any preprocess metadata CSV: the current name is
        # "preprocessed_ref-<ref>_detect-<det>.csv", and older runs used
        # "preprocessed_detect_<...>.csv" -- both start with "preprocessed" and
        # contain "detect". Pick the most recently written if several coexist.
        metadata_path = None
        try:
            _candidates = [
                f for f in os.listdir(self.base_path)
                if f.lower().startswith("preprocessed")
                and "detect" in f.lower()
                and f.lower().endswith(".csv")
            ]
            if _candidates:
                metadata_path = os.path.join(
                    self.base_path,
                    max(_candidates, key=lambda f: os.path.getmtime(os.path.join(self.base_path, f))),
                )
        except Exception:
            metadata_path = None
        # Fallback to legacy filename for backwards compatibility
        if metadata_path is None:
            metadata_path = os.path.join(self.base_path, "preprocess_metadata.csv")
        metadata_df = None
        if os.path.exists(metadata_path):
            try:
                metadata_df = pd.read_csv(metadata_path)
                metadata_df["section_id"] = metadata_df["section_id"].astype(str)
            except Exception:
                metadata_df = None

        for animal_folder in sorted(os.listdir(self.base_path)):
            subfolder = os.path.join(self.base_path, animal_folder)
            if not os.path.isdir(subfolder):
                continue
            files = [f for f in os.listdir(subfolder) if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')) and "reference" in f.lower()]
            def _section_sort_key(name: str):
                import re as _re
                m = _re.search(r'_([0-9]+)(?=\.[^.]+$)', name)
                return int(m.group(1)) if m else 999999
            files = sorted(files, key=_section_sort_key)
            for f in files:
                ref_path = os.path.join(subfolder, f)
                # Update progress label with animal and section info
                section_id_display = ""
                try:
                    import re as _re
                    match = _re.search(r'_([0-9]+)(?=\.[^.]+$)', f)
                    section_id_display = match.group(1) if match else f
                except Exception:
                    section_id_display = f
                # Count total sections for this animal folder
                try:
                    files_all = [fn for fn in os.listdir(subfolder) if fn.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')) and "reference" in fn.lower()]
                    total_sections = len(files_all)
                except Exception:
                    total_sections = ""
                try:
                    suffix = f"/{total_sections}" if total_sections != "" else ""
                    self.workspace.set_progress(f"Analyzing animal {animal_folder} - Section {section_id_display}{suffix}")
                except Exception:
                    pass
                QApplication.processEvents()
                try:
                    # Pass the pre-drawn ROI specification if available
                    roi_data = getattr(self, 'pre_draw_roi_data', None)
                    result = qt_process_images(
                        self.workspace,
                        ref_path,
                        method,
                        detection,
                        section_thresh,
                        cd68_color,
                        exclude_color,
                        cd68_start_val,
                        exclude_start_val,
                        fixed_roi_data=roi_data,
                        background_percent=bg_percent,
                    )
                except Exception as e:
                    result = "EXIT"
                    QMessageBox.warning(self, "Error", f"Error processing {f}: {e}\n{traceback.format_exc()}")
                if result == "EXIT":
                    exit_requested = True
                    break
                if result and result != "EXIT":
                    # Compose result row and derive identifiers
                    result_row = result.copy()
                    result_row["animal_id"] = animal_folder
                    match = re.search(r'_([0-9]+)(?=\.[^.]+$)', f)
                    if match:
                        result_row["section_id"] = match.group(1)
                    else:
                        # Use filename as section_id if pattern not matched
                        result_row["section_id"] = f
                    result_row["filename"] = f
                    # Determine pixel_to_um_scale using metadata
                    pixel_to_um = ""
                    if metadata_df is not None:
                        # Attempt to locate metadata row for this animal and section
                        try:
                            md_match = metadata_df[(metadata_df["animal_id"] == animal_folder) & (metadata_df["section_id"] == str(result_row["section_id"]))]
                            if not md_match.empty:
                                row = md_match.iloc[0]
                                # Extract provided scale and downsample factor
                                scale_val = row["scale"]
                                down_factor = row["downsample_factor"] if "downsample_factor" in row else ""
                                if scale_val != "" and not pd.isna(scale_val):
                                    try:
                                        scale_val = float(scale_val)
                                        if down_factor != "" and not pd.isna(down_factor):
                                            down_factor = float(down_factor)
                                            pixel_to_um = scale_val * down_factor
                                        else:
                                            pixel_to_um = scale_val
                                    except Exception:
                                        pixel_to_um = ""
                        except Exception:
                            pixel_to_um = ""
                    result_row["pixel_to_um_scale"] = pixel_to_um
                    # Convert pixel areas to real units (um^2) when a scale is available.
                    # Scale is linear (px/um), so an area divides by the scale squared.
                    if pixel_to_um != "" and pixel_to_um is not None:
                        try:
                            scale_sq = float(pixel_to_um) ** 2
                        except (TypeError, ValueError):
                            scale_sq = 0.0
                        if scale_sq > 0:
                            for _base in ("whole_area", "left_area", "right_area", "infarct_area"):
                                _val = result_row.get(f"{_base}_px", "")
                                if _val != "" and _val is not None:
                                    try:
                                        result_row[f"{_base}_um^2"] = float(_val) / scale_sq
                                    except (TypeError, ValueError):
                                        pass
                    # Set analysis date if not already present
                    if "date_analyzed" not in result_row:
                        result_row["date_analyzed"] = CURRENT_DATE
                    # Remove columns that should not be carried over
                    # When updating aggregated_df, match by animal_id and section_id
                    mask_existing = (aggregated_df['animal_id'] == animal_folder) & (aggregated_df['section_id'] == str(result_row["section_id"]))
                    if not aggregated_df[mask_existing].empty:
                        count_existing = aggregated_df[mask_existing].shape[0]
                        aggregated_df.loc[mask_existing, :] = pd.DataFrame([result_row] * count_existing, index=aggregated_df[mask_existing].index)
                    else:
                        new_row_df = pd.DataFrame([result_row])
                        if aggregated_df.empty:
                            aggregated_df = new_row_df
                        else:
                            aggregated_df = pd.concat([aggregated_df, new_row_df], ignore_index=True)
            if exit_requested:
                break
        if not aggregated_df.empty:
            # Reorder columns so that threshold values come after area metrics and pixel_to_um_scale before date
            # Column order: ids, then each area as pixels (_px) and real units (_um^2),
            # then thresholds and the scale. date_analyzed is forced to the last column.
            desired_order = [
                "animal_id",
                "section_id",
                "filename",
                "whole_area_px",
                "left_area_px",
                "right_area_px",
                "infarct_area_px",
                "whole_area_um^2",
                "left_area_um^2",
                "right_area_um^2",
                "infarct_area_um^2",
                "brain_outline_threshold",
                "CD68_threshold",
                "exclude_threshold",
                "pixel_to_um_scale",
            ]
            # Prune the positive-pixel / intensity columns from the output. Drop them
            # outright so the extra_cols fallback below cannot re-append them.
            pruned_cols = [
                "infarct_area_positive",
                "infarct_area_intensity_avg",
                "infarct_area_positive_intensity_avg",
                "background_intensity",
                "infarct_intensity_avg_normalized",
                "infarct_intensity_positive_avg_normalized",
            ]
            aggregated_df = aggregated_df.drop(columns=pruned_cols, errors="ignore")
            # Keep date_analyzed as the very last column, after any extra columns.
            extra_cols = [
                col for col in aggregated_df.columns
                if col not in desired_order and col != "date_analyzed"
            ]
            final_order = desired_order + extra_cols + ["date_analyzed"]
            aggregated_df = aggregated_df.reindex(columns=final_order)
            # Any unspecified value (e.g. no scale -> no um^2 area, no scale value) is
            # written as "NA" rather than left blank.
            aggregated_df = aggregated_df.where(aggregated_df != "", np.nan)
            if self._save_results_with_retry(aggregated_df, final_csv_path):
                # Remember the results CSV and pre-fill the Quantify tab so the
                # user can compute % infarct without re-selecting the file.
                self.last_results_csv_path = final_csv_path
                if getattr(self, "quantify_tab", None) is not None:
                    self.quantify_tab.autoload_results_csv(final_csv_path)
                QMessageBox.information(
                    self,
                    "Analysis complete",
                    f"Analysis finished.\nView results at: {final_csv_path}",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Results not saved",
                    "Results were not saved because the file stayed open/locked.\n"
                    f"No changes were written to:\n{final_csv_path}",
                )
        else:
            QMessageBox.information(self, "Analysis complete", "Analysis finished. No results were saved.")
        # Return to the parameter page once analysis completes
        self.show_workspace(False)

    def _save_results_with_retry(self, df, path) -> bool:
        """Delegate to the module-level :func:`save_csv_with_retry`.

        Kept as a thin method for the existing call sites and tests; the retry
        logic is shared with the Quantify tab.
        """
        return save_csv_with_retry(self, df, path)

    def pre_draw_roi_action(self) -> None:
        """Launch the interactive pre-ROI drawing workflow.

        This method wraps the call to `patched_draw_pre_roi` and updates the
        AnalysisTab state with the returned ROI specification. If no base
        directory is selected, a warning is shown. On success, the ROI name
        displayed in the GUI is updated and the internal data stored.
        """
        if not getattr(self, 'base_path', ''):
            QMessageBox.warning(self, "No folder", "Please select a sections folder first.")
            return
        try:
            self.show_workspace(True)
            data = self.workspace.run_pre_roi(self.base_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error drawing ROI: {e}\n{traceback.format_exc()}")
            self.show_workspace(False)
            return
        if data:
            self.pre_draw_roi_data = data
            # Display only the ROI name (not full path) in the field
            self.pre_roi_field.setText(data.get('name', ''))
        else:
            # Reset field if user cancelled
            self.pre_draw_roi_data = None
            self.pre_roi_field.setText("")
        self.show_workspace(False)


class QuantifyTab(QWidget):
    """Tab for computing the edema-corrected infarct percentage from a results CSV.

    Consumes the per-section results CSV produced by the Analyze tab (loaded
    manually or auto-filled after an analysis run) and writes a new CSV that
    relabels the hemisphere areas as ipsi_/contra_ and adds the indirect /
    edema-corrected infarct percentage (Swanson's method):

        % infarct = [Contra - (Ipsi - Infarct)] / Contra * 100
    """

    # Columns the results CSV must contain to be quantifiable.
    REQUIRED_COLUMNS = [
        "animal_id",
        "section_id",
        "whole_area_px",
        "left_area_px",
        "right_area_px",
        "infarct_area_px",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_csv_path = ""

        params_layout = QVBoxLayout()
        params_layout.setSizeConstraint(QLayout.SetMinimumSize)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(6)
        params_page = QWidget()

        # 1) Equation header -------------------------------------------------
        eq_card, eq_layout = create_card("1) Infarct equation")
        equation = QLabel()
        equation.setTextFormat(Qt.RichText)
        equation.setWordWrap(True)
        equation.setText(
            "<div style='font-size:12pt;'>"
            "<b>% infarct</b>&nbsp;=&nbsp; "
            "[ Contra hemi area − ( Ipsi hemi area − Infarct area ) ]"
            "&nbsp;÷&nbsp; Contra hemi area&nbsp;×&nbsp; 100"
            "</div>"
        )
        eq_layout.addWidget(equation)
        note = QLabel(
            "The contralesional hemisphere is the healthy reference; this corrects "
            "the infarct measurement for swelling (edema) of the ipsilesional side."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6B7280;")
        eq_layout.addWidget(note)
        params_layout.addWidget(eq_card)

        # 2) Results CSV -----------------------------------------------------
        csv_card, csv_layout = create_card("2) Results CSV")
        csv_desc = QLabel(
            "Use the results from the Analyze step. This is filled in automatically "
            "after an analysis run, or load a results CSV manually."
        )
        csv_desc.setWordWrap(True)
        csv_layout.addWidget(csv_desc)
        self.csv_btn = QPushButton("Select Results CSV...")
        self.csv_btn.setObjectName("secondaryAction")
        self.csv_btn.clicked.connect(self.select_results_csv)
        csv_layout.addWidget(self.csv_btn)
        self.csv_field = QLineEdit()
        self.csv_field.setReadOnly(True)
        self.csv_field.setPlaceholderText("Auto-filled from the last analysis, or load one")
        csv_layout.addWidget(self.csv_field)
        params_layout.addWidget(csv_card)

        # 3) Ipsilesional side ----------------------------------------------
        side_card, side_layout = create_card("3) Ipsilesional side")
        side_row = QHBoxLayout()
        side_row.setContentsMargins(0, 0, 0, 0)
        side_row.setSpacing(6)
        side_help = QToolButton()
        side_help.setText("?")
        side_help.setStyleSheet(
            "background-color:#5FA2E5; color:white; border-radius:8px; min-width:16px; min-height:16px;"
        )
        side_help.setToolTip(
            "Which hemisphere is the lesioned (ipsilesional) side.\n"
            "The opposite hemisphere is used as the healthy contralesional reference.\n"
            "Its area columns are relabeled ipsi_ / contra_ in the output."
        )
        side_help.clicked.connect(
            lambda: QMessageBox.information(self, "Ipsilesional side", side_help.toolTip())
        )
        self.side_label = QLabel("Ipsilesional side:")
        self.ipsi_combo = QComboBox()
        self.ipsi_combo.addItems(["Left", "Right"])
        self.ipsi_combo.setCurrentText("Left")
        side_row.addWidget(side_help)
        side_row.addWidget(self.side_label)
        side_row.addWidget(self.ipsi_combo)
        side_row.addStretch(1)
        side_holder = QWidget()
        side_holder.setLayout(side_row)
        side_layout.addWidget(side_holder)
        params_layout.addWidget(side_card)

        # 4) Run -------------------------------------------------------------
        run_card, run_layout = create_card("4) Run quantification")
        self.run_btn = QPushButton("Run Quantification")
        self.run_btn.setObjectName("primaryAction")
        self.run_btn.clicked.connect(self.run_quantify)
        run_layout.addWidget(self.run_btn)
        params_layout.addWidget(run_card)

        params_layout.addStretch(1)
        params_page.setLayout(params_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(params_page)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)
        main_layout.addWidget(scroll, 1)
        self.setMinimumHeight(320)

    def autoload_results_csv(self, path: str) -> None:
        """Pre-fill the results CSV path (called after a successful analysis)."""
        self.results_csv_path = path or ""
        self.csv_field.setText(self.results_csv_path)

    def select_results_csv(self) -> None:
        """Let the user pick a results CSV manually."""
        start_dir = os.path.dirname(self.results_csv_path) if self.results_csv_path else ""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select Results CSV",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        if fname:
            self.results_csv_path = fname
            self.csv_field.setText(fname)

    def run_quantify(self) -> None:
        """Compute the edema-corrected infarct percentage and write a new CSV."""
        import datetime
        import pandas as pd

        if not self.results_csv_path:
            QMessageBox.warning(self, "No CSV", "Please load a results CSV first.")
            return
        try:
            df = pd.read_csv(self.results_csv_path)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot read CSV", f"Failed to read the results CSV:\n{exc}")
            return

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            QMessageBox.warning(
                self,
                "Unexpected CSV",
                "The selected file does not look like an InfarQuant results CSV.\n"
                "Missing column(s): " + ", ".join(missing),
            )
            return

        side = self.ipsi_combo.currentText().strip()
        # Coerce area columns to numeric so stray text / "NA" become NaN.
        for col in ("whole_area_px", "left_area_px", "right_area_px", "infarct_area_px"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if side.lower() == "left":
            ipsi = df["left_area_px"]
            contra = df["right_area_px"]
        else:
            ipsi = df["right_area_px"]
            contra = df["left_area_px"]
        infarct = df["infarct_area_px"]

        # Edema-corrected (indirect) infarct percentage. Guard divide-by-zero /
        # missing contralesional area -> NaN (written as "NA").
        pct = (contra - (ipsi - infarct)) / contra * 100.0
        pct = pct.where((contra != 0) & contra.notna(), np.nan)

        CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
        out = pd.DataFrame()
        out["animal_id"] = df["animal_id"]
        out["section_id"] = df["section_id"]
        if "filename" in df.columns:
            out["filename"] = df["filename"]
        out["ipsilesional_side"] = side
        out["whole_area_px"] = df["whole_area_px"]
        out["ipsi_area_px"] = ipsi
        out["contra_area_px"] = contra
        out["infarct_area_px"] = infarct
        out["contra_adjusted_infarct_pct"] = pct
        out["date_quantified"] = CURRENT_DATE

        # Any blank value is written as "NA", matching the analysis CSV.
        out = out.where(out != "", np.nan)
        out_dir = os.path.dirname(self.results_csv_path)
        out_name = "quantify_" + os.path.basename(self.results_csv_path)
        out_path = os.path.join(out_dir, out_name)
        if save_csv_with_retry(self, out, out_path):
            QMessageBox.information(
                self,
                "Quantification complete",
                f"Infarct quantification finished.\nView results at: {out_path}",
            )
        else:
            QMessageBox.warning(
                self,
                "Results not saved",
                "Results were not saved because the file stayed open/locked.\n"
                f"No changes were written to:\n{out_path}",
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InfarQuant")
        self.setMinimumSize(980, 600)
        # Create the preprocessing, analysis and quantification pages and use a
        # QStackedWidget to switch between them.
        self.analysis_tab = AnalysisTab()
        self.preprocess_tab = PreprocessTab(self.analysis_tab)
        self.quantify_tab = QuantifyTab()
        # Let the analysis step pre-fill the Quantify tab with its results CSV.
        self.analysis_tab.quantify_tab = self.quantify_tab
        self.stack = QStackedWidget()
        self.stack.addWidget(self.preprocess_tab)
        self.stack.addWidget(self.analysis_tab)
        self.stack.addWidget(self.quantify_tab)
        # Create navigation buttons that behave like tabs
        self.btn_preprocess = QPushButton("Preprocess")
        self.btn_analyze = QPushButton("Analyze")
        self.btn_quantify = QPushButton("Quantify")
        for btn in (self.btn_preprocess, self.btn_analyze, self.btn_quantify):
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            btn.setMinimumHeight(36)
            btn.setCheckable(True)
            btn.setProperty("nav", True)
        self.btn_preprocess.clicked.connect(lambda: self.switch_page(0))
        self.btn_analyze.clicked.connect(lambda: self.switch_page(1))
        self.btn_quantify.clicked.connect(lambda: self.switch_page(2))
        # Build header with title and navigation
        header = QFrame()
        header.setProperty("card", True)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        header_layout.setSpacing(12)
        header.setMinimumHeight(72)
        title = QLabel()
        logo_path = _resolve_logo_path()
        logo_pix = QPixmap(logo_path) if logo_path else QPixmap()
        if not logo_pix.isNull():
            target_h = 46
            title.setPixmap(logo_pix.scaledToHeight(target_h, Qt.SmoothTransformation))
            title.setFixedHeight(target_h)
            title.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        else:
            title.setText("InfarQuant")
            title.setStyleSheet("font-size: 18pt; font-weight: 700;")
        header_layout.addWidget(self.btn_preprocess)
        header_layout.addWidget(self.btn_analyze)
        header_layout.addWidget(self.btn_quantify)
        header_layout.addStretch(1)
        header_layout.addWidget(title, 0, Qt.AlignRight | Qt.AlignVCenter)
        # Central layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        main_layout.addWidget(header)
        main_layout.addWidget(self.stack)
        self.setCentralWidget(main_widget)
        # Status bar for lightweight progress hints
        status = QStatusBar()
        self.setStatusBar(status)
        self.statusBar().showMessage("Ready")
        # Select the first page by default
        self.switch_page(0)

    def switch_page(self, index: int) -> None:
        """
        Switch to the given page index and update button styles.

        This method is used by the navigation buttons to display the
        preprocessing page (index 0), the analysis page (index 1) or the
        quantification page (index 2).  The selected button is highlighted,
        while the others use the unselected style.
        """
        self.stack.setCurrentIndex(index)
        # Update checked state
        self.btn_preprocess.setChecked(index == 0)
        self.btn_analyze.setChecked(index == 1)
        self.btn_quantify.setChecked(index == 2)
        # Update status bar context
        if index == 0:
            self.statusBar().showMessage("Preprocess: split whole-slide images into sections.")
        elif index == 1:
            self.statusBar().showMessage("Analyze: interactively segment and quantify infarcts.")
        else:
            self.statusBar().showMessage("Quantify: compute edema-corrected % infarct from results.")
