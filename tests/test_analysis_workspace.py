import os
import json
import numpy as np
import pandas as pd
import pytest
import cv2
from PyQt5.QtWidgets import QInputDialog

from infarquant import ui as app
from .helpers import WorkspaceHarness, make_test_images, make_brain_contours, get_action, get_action_button


def test_midline_draw_accept_and_flip(qapp, monkeypatch):
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    img[0, 0] = (10, 20, 30)
    img[3, 4] = (40, 50, 60)
    alerts = []
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: alerts.append(msg))

    ws = WorkspaceHarness()

    def auto_action():
        get_action(ws.last_actions, "Flip H")()
        get_action(ws.last_actions, "Flip V")()
        handler = ws._viewer._full_pane._click_handler
        handler(1, 1)
        handler(3, 2)
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action
    points, mod_image, flip_flags = ws.run_midline(img, "test")
    assert len(points) == 2
    assert flip_flags == (True, True)
    assert np.array_equal(mod_image, cv2.flip(img, -1))
    assert alerts == []


def test_midline_invalid_then_valid(qapp, monkeypatch):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    alerts = []
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: alerts.append(msg))

    ws = WorkspaceHarness()

    def auto_action():
        get_action(ws.last_actions, "Accept")()
        handler = ws._viewer._full_pane._click_handler
        handler(2, 2)
        handler(7, 7)
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action
    points, mod_image, flip_flags = ws.run_midline(img, "test")
    assert len(points) == 2
    assert any("Invalid midline" in msg for msg in alerts)


def test_midline_accept_button_state_updates(qapp, monkeypatch):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: None)
    ws = WorkspaceHarness()

    def auto_action():
        accept_btn = get_action_button(ws, "Accept")
        assert not accept_btn.isEnabled()
        assert not bool(accept_btn.property("ready"))

        handler = ws._viewer._full_pane._click_handler
        handler(2, 2)
        assert not accept_btn.isEnabled()
        handler(7, 7)
        assert accept_btn.isEnabled()
        assert bool(accept_btn.property("ready"))

        get_action(ws.last_actions, "Clear")()
        assert not accept_btn.isEnabled()
        assert not bool(accept_btn.property("ready"))

        handler(3, 3)
        handler(8, 8)
        assert accept_btn.isEnabled()
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action
    points, _, _ = ws.run_midline(img, "test")
    assert len(points) == 2


def test_brain_threshold_slider_sync_and_accept(qapp, monkeypatch):
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (70, 70), (255, 255, 255), -1)
    split_points = [(40, 0), (40, 79)]
    alerts = []
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: alerts.append(msg))

    ws = WorkspaceHarness()

    def auto_action():
        # Controls: [label, slider, box]
        slider = ws.last_controls[1]
        box = ws.last_controls[2]
        slider.setValue(30)
        assert box.value() == 30
        box.setValue(40)
        assert slider.value() == 40
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action
    result = ws.run_brain_threshold(img, split_points, 10)
    assert result[0]["whole"] is not None
    assert alerts == []


def test_brain_threshold_accept_disabled_when_no_contour(qapp, monkeypatch):
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    split_points = [(40, 0), (40, 79)]
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: None)
    ws = WorkspaceHarness()

    def auto_action():
        accept_btn = get_action_button(ws, "Accept")
        assert not accept_btn.isEnabled()
        assert not bool(accept_btn.property("ready"))
        get_action(ws.last_actions, "Exit")()

    ws._auto_action = auto_action
    result = ws.run_brain_threshold(img, split_points, 10)
    assert result[0] == "EXIT"


def test_infarct_threshold_multi_contour_add_and_clear(qapp, monkeypatch):
    infarct_img, ref_img = make_test_images()
    split_points = [(50, 0), (50, 99)]
    brain = make_brain_contours()
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: None)

    ws = WorkspaceHarness()

    # Add a contour, then clear, then accept -> should return a single contour (not list)
    def auto_action_clear():
        get_action(ws.last_actions, "Add contour")()
        get_action(ws.last_actions, "Clear contours")()
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action_clear
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=100,
        exclude_start_val=0,
        fixed_roi_data=None,
    )
    assert result["no_infarct"] is False
    assert not isinstance(result["selected_infarct_contour"], list)

    # Add contours and accept -> should return list
    ws = WorkspaceHarness()

    def auto_action_multi():
        get_action(ws.last_actions, "Add contour")()
        get_action(ws.last_actions, "Cycle contour")()
        get_action(ws.last_actions, "Add contour")()
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action_multi
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=100,
        exclude_start_val=0,
        fixed_roi_data=None,
    )
    assert isinstance(result["selected_infarct_contour"], list)


def test_infarct_threshold_accept_button_tracks_contour_validity(qapp, monkeypatch):
    infarct_img, ref_img = make_test_images()
    split_points = [(50, 0), (50, 99)]
    brain = make_brain_contours()
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: None)

    ws = WorkspaceHarness()

    def auto_action():
        accept_btn = get_action_button(ws, "Accept")
        cd68_slider = ws.last_controls[1]

        assert not accept_btn.isEnabled()
        assert not bool(accept_btn.property("ready"))

        cd68_slider.setValue(100)
        assert accept_btn.isEnabled()
        assert bool(accept_btn.property("ready"))

        cd68_slider.setValue(255)
        assert not accept_btn.isEnabled()
        assert not bool(accept_btn.property("ready"))

        get_action(ws.last_actions, "No infarct")()

    ws._auto_action = auto_action
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=255,
        exclude_start_val=0,
        fixed_roi_data=None,
    )
    assert result["no_infarct"] is True


def test_infarct_threshold_multi_contour_distinct(qapp, monkeypatch):
    infarct_img, ref_img = make_test_images()
    split_points = [(50, 0), (50, 99)]
    brain = make_brain_contours()
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: None)

    ws = WorkspaceHarness()

    def auto_action_multi():
        get_action(ws.last_actions, "Add contour")()
        get_action(ws.last_actions, "Cycle contour")()
        get_action(ws.last_actions, "Add contour")()
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action_multi
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=100,
        exclude_start_val=0,
        fixed_roi_data=None,
    )
    contours = result["selected_infarct_contour"]
    assert isinstance(contours, list)
    centers = []
    for cnt in contours:
        m = cv2.moments(cnt)
        assert m["m00"] > 0
        centers.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    assert len(set(centers)) >= 2


def test_infarct_threshold_roi_set_reset_and_popup(qapp, monkeypatch):
    infarct_img, ref_img = make_test_images()
    split_points = [(50, 0), (50, 99)]
    brain = make_brain_contours()
    alerts = []
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: alerts.append(msg))

    ws = WorkspaceHarness()

    def auto_action_set_roi():
        handler = ws._viewer._full_pane._click_handler
        handler(10, 10)
        handler(20, 10)
        handler(20, 20)
        get_action(ws.last_actions, "Set ROI")()
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action_set_roi
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=100,
        exclude_start_val=0,
        fixed_roi_data=None,
    )
    assert result["roi_mask"] is not None

    ws = WorkspaceHarness()

    def auto_action_reset_roi():
        handler = ws._viewer._full_pane._click_handler
        handler(10, 10)
        handler(20, 10)
        handler(20, 20)
        get_action(ws.last_actions, "Set ROI")()
        get_action(ws.last_actions, "Reset ROI")()
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action_reset_roi
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=100,
        exclude_start_val=0,
        fixed_roi_data=None,
    )
    assert result["roi_mask"] is None

    # Popup when trying to set ROI with insufficient points
    ws = WorkspaceHarness()

    def auto_action_bad_roi():
        get_action(ws.last_actions, "Set ROI")()
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action_bad_roi
    _ = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=100,
        exclude_start_val=0,
        fixed_roi_data=None,
    )
    assert any("Need at least 3 points" in msg for msg in alerts)


def test_infarct_threshold_fixed_roi_and_no_infarct(qapp, monkeypatch):
    infarct_img, ref_img = make_test_images()
    split_points = [(50, 0), (50, 99)]
    brain = make_brain_contours()
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: None)

    fixed_roi = {
        "poly_norm": [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)],
        "centroid_norm": (0.15, 0.15),
        "flip_h": False,
        "flip_v": False,
    }

    ws = WorkspaceHarness()

    def auto_action_accept():
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action_accept
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=100,
        exclude_start_val=0,
        fixed_roi_data=fixed_roi,
    )
    assert result["fixed_roi_used"] is True
    assert result["roi_mask"] is not None

    ws = WorkspaceHarness()

    def auto_action_no_infarct():
        get_action(ws.last_actions, "No infarct")()

    ws._auto_action = auto_action_no_infarct
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="none",
        cd68_start_val=100,
        exclude_start_val=0,
        fixed_roi_data=None,
    )
    assert result["no_infarct"] is True
    assert result["selected_infarct_contour"] is None


def test_infarct_threshold_slider_sync_for_exclude(qapp, monkeypatch):
    infarct_img, ref_img = make_test_images()
    split_points = [(50, 0), (50, 99)]
    brain = make_brain_contours()
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: None)

    ws = WorkspaceHarness()

    def auto_action():
        # Controls include CD68 and Exclude sliders/boxes
        cd68_slider = ws.last_controls[1]
        cd68_box = ws.last_controls[2]
        exclude_slider = ws.last_controls[4]
        exclude_box = ws.last_controls[5]
        cd68_slider.setValue(120)
        assert cd68_box.value() == 120
        exclude_slider.setValue(80)
        assert exclude_box.value() == 80
        get_action(ws.last_actions, "Accept")()

    ws._auto_action = auto_action
    result = ws.run_infarct_threshold(
        infarct_img,
        ref_img,
        brain,
        split_points,
        infarct_contour_detection="max",
        display_info="",
        cd68_color="red",
        exclude_color="green",
        cd68_start_val=100,
        exclude_start_val=50,
        fixed_roi_data=None,
    )
    assert result["no_infarct"] is False


def test_pre_draw_roi_save_and_popup(qapp, monkeypatch, tmp_path):
    # Build folder structure expected by run_pre_roi
    base = tmp_path / "preprocessed_sections"
    animal = base / "A1"
    animal.mkdir(parents=True)
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (70, 70), (255, 255, 255), -1)
    ref_path = animal / "A1_reference_1.png"
    inf_path = animal / "A1_infarct_1.png"
    cv2.imwrite(str(ref_path), img)
    cv2.imwrite(str(inf_path), img)

    alerts = []
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: alerts.append(msg))

    ws = WorkspaceHarness()

    # Save ROI successfully
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("TestROI", True))

    def auto_action_save():
        handler = ws._viewer._full_pane._click_handler
        handler(10, 10)
        handler(20, 10)
        handler(20, 20)
        get_action(ws.last_actions, "Save ROI")()

    ws._auto_action = auto_action_save
    data = ws.run_pre_roi(str(base))
    assert data is not None
    assert (base / "TestROI_ROI.json").exists()

    # Popup for empty ROI name
    alerts.clear()
    ws = WorkspaceHarness()
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("", True))

    def auto_action_empty_name():
        handler = ws._viewer._full_pane._click_handler
        handler(10, 10)
        handler(20, 10)
        handler(20, 20)
        get_action(ws.last_actions, "Save ROI")()

    ws._auto_action = auto_action_empty_name
    data = ws.run_pre_roi(str(base))
    assert data is None
    assert any("ROI name cannot be empty" in msg for msg in alerts)


def test_pre_draw_roi_no_sections_popup(qapp, monkeypatch, tmp_path):
    base = tmp_path / "empty"
    base.mkdir()
    alerts = []
    monkeypatch.setattr(app.pyautogui, "alert", lambda msg: alerts.append(msg))
    ws = WorkspaceHarness()
    data = ws.run_pre_roi(str(base))
    assert data is None
    assert any("No sections found" in msg for msg in alerts)


def test_pixel_to_um_scale_from_metadata(qapp, monkeypatch, tmp_path):
    base = tmp_path / "preprocessed_sections"
    animal = base / "A1"
    animal.mkdir(parents=True)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(animal / "A1_reference_1.png"), img)
    cv2.imwrite(str(animal / "A1_reference_2.png"), img)

    metadata = pd.DataFrame([
        {"animal_id": "A1", "section_id": "1", "scale": 2.0, "downsample_factor": 0.5},
        {"animal_id": "A1", "section_id": "2", "scale": 3.0, "downsample_factor": ""},
    ])
    metadata.to_csv(base / "preprocessed_detect_red.csv", index=False)

    def fake_qt_process_images(*args, **kwargs):
        return {
            "whole_area": 100,
            "left_area": 50,
            "right_area": 50,
            "infarct_area": 10,
            "infarct_area_positive": 10,
            "infarct_area_intensity_avg": 1.0,
            "infarct_area_positive_intensity_avg": 1.0,
            "background_intensity": 0.0,
            "infarct_intensity_avg_normalized": 0.0,
            "infarct_intensity_positive_avg_normalized": 0.0,
            "brain_outline_threshold": 20,
            "CD68_threshold": 100,
            "exclude_threshold": 0,
        }

    monkeypatch.setattr(app, "qt_process_images", fake_qt_process_images)
    monkeypatch.setattr(app.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args, **kwargs: None)

    tab = app.AnalysisTab()
    tab.base_path = str(base)
    tab.method_edit.setText("contour")
    tab.detection_edit.setText("max")
    tab.exclude_combo.setCurrentText("none")
    tab.run_analysis()

    results_path = base / "results_detect_red_contour.csv"
    assert results_path.exists()
    df = pd.read_csv(results_path)
    df["section_id"] = df["section_id"].astype(str)
    row1 = df[df["section_id"] == "1"].iloc[0]
    row2 = df[df["section_id"] == "2"].iloc[0]
    assert float(row1["pixel_to_um_scale"]) == pytest.approx(1.0)
    assert float(row2["pixel_to_um_scale"]) == pytest.approx(3.0)
