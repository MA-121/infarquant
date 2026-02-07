import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cv2

from infarquant import ui as app


def _write_reference_section(base: Path, animal: str = "A1", section: int = 1) -> Path:
    animal_dir = base / animal
    animal_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((12, 12, 3), dtype=np.uint8)
    ref_path = animal_dir / f"{animal}_reference_{section}.png"
    cv2.imwrite(str(ref_path), img)
    return ref_path


def _result_stub():
    return {
        "whole_area": 100,
        "left_area": 50,
        "right_area": 50,
        "infarct_area": 10,
        "infarct_area_positive": 7,
        "infarct_area_intensity_avg": 1.0,
        "infarct_area_positive_intensity_avg": 1.0,
        "background_intensity": 0.0,
        "infarct_intensity_avg_normalized": 0.0,
        "infarct_intensity_positive_avg_normalized": 0.0,
        "brain_outline_threshold": 20,
        "CD68_threshold": 100,
        "exclude_threshold": 0,
        "FIXED_ROI": False,
    }


def test_select_base_autoloads_roi_json(qapp, monkeypatch, tmp_path):
    roi = {"name": "SavedROI", "poly_norm": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}
    with open(tmp_path / "SavedROI_ROI.json", "w", encoding="utf-8") as f:
        json.dump(roi, f)

    monkeypatch.setattr(app.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path))
    tab = app.AnalysisTab()
    tab.select_base()

    assert tab.base_path == str(tmp_path)
    assert tab.pre_draw_roi_data is not None
    assert tab.pre_draw_roi_data["name"] == "SavedROI"
    assert tab.pre_roi_field.text() == "SavedROI"


def test_run_analysis_validation_errors(qapp, monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[2]))

    tab = app.AnalysisTab()
    tab.run_analysis()
    assert any("Please select a sections folder first." in w for w in warnings)

    warnings.clear()
    tab.base_path = str(tmp_path)
    tab.method_edit.setText("invalid")
    tab.run_analysis()
    assert any("Method must be 'contour' or 'mask'." in w for w in warnings)

    warnings.clear()
    tab.method_edit.setText("contour")
    tab.detection_edit.setText("invalid")
    tab.run_analysis()
    assert any("Detection must be 'max' or 'all'." in w for w in warnings)

    warnings.clear()
    tab.detection_edit.setText("max")
    tab.bg_percent_edit.setText("abc")
    tab.run_analysis()
    assert any("Background percent must be a numeric value." in w for w in warnings)

    warnings.clear()
    tab.bg_percent_edit.setText("0")
    tab.run_analysis()
    assert any("Background percent must be between 0 and 100." in w for w in warnings)


def test_run_analysis_writes_expected_filename_for_exclude_channel(qapp, monkeypatch, tmp_path):
    _write_reference_section(tmp_path, animal="A1", section=1)

    monkeypatch.setattr(app, "qt_process_images", lambda *args, **kwargs: _result_stub())
    monkeypatch.setattr(app.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args, **kwargs: None)

    tab = app.AnalysisTab()
    tab.base_path = str(tmp_path)
    tab.method_edit.setText("contour")
    tab.detection_edit.setText("max")
    tab.cd68_combo.setCurrentText("red")
    tab.exclude_combo.setCurrentText("green")
    tab.run_analysis()

    out_csv = tmp_path / "results_detect_red_exclude_green_contour.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert str(df.iloc[0]["animal_id"]) == "A1"


def test_run_analysis_output_csv_column_contract(qapp, monkeypatch, tmp_path):
    _write_reference_section(tmp_path, animal="A1", section=1)

    monkeypatch.setattr(app, "qt_process_images", lambda *args, **kwargs: _result_stub())
    monkeypatch.setattr(app.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args, **kwargs: None)

    tab = app.AnalysisTab()
    tab.base_path = str(tmp_path)
    tab.method_edit.setText("contour")
    tab.detection_edit.setText("max")
    tab.exclude_combo.setCurrentText("none")
    tab.run_analysis()

    out_csv = tmp_path / "results_detect_red_contour.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    expected_prefix = [
        "animal_id",
        "section_id",
        "filename",
        "whole_area",
        "left_area",
        "right_area",
        "infarct_area",
        "infarct_area_positive",
        "infarct_area_intensity_avg",
        "infarct_area_positive_intensity_avg",
        "background_intensity",
        "infarct_intensity_avg_normalized",
        "infarct_intensity_positive_avg_normalized",
        "brain_outline_threshold",
        "CD68_threshold",
        "exclude_threshold",
        "pixel_to_um_scale",
        "date_analyzed",
    ]
    assert list(df.columns[: len(expected_prefix)]) == expected_prefix
    assert "FIXED_ROI" in df.columns


def test_run_analysis_exit_path_saves_no_csv(qapp, monkeypatch, tmp_path):
    _write_reference_section(tmp_path, animal="A1", section=1)

    messages = []
    monkeypatch.setattr(app, "qt_process_images", lambda *args, **kwargs: "EXIT")
    monkeypatch.setattr(app.QMessageBox, "information", lambda *args, **kwargs: messages.append(args[2]))
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args, **kwargs: None)

    tab = app.AnalysisTab()
    tab.base_path = str(tmp_path)
    tab.method_edit.setText("contour")
    tab.detection_edit.setText("max")
    tab.exclude_combo.setCurrentText("none")
    tab.run_analysis()

    out_csv = tmp_path / "results_detect_red_contour.csv"
    assert not out_csv.exists()
    assert any("No results were saved." in m for m in messages)
