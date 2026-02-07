import os

import pandas as pd
import pytest

import cv2

from infarquant import ui as app


def test_run_preprocess_requires_folder(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[2]))

    tab = app.PreprocessTab()
    tab.folder_path = ""
    tab.run_preprocess()

    assert any("Please select an input folder first." in w for w in warnings)


def test_run_preprocess_invalid_scale_does_not_call_process_folder(qapp, monkeypatch, tmp_path):
    called = {"process_folder": False}
    warnings = []

    def fake_process_folder(*args, **kwargs):
        called["process_folder"] = True
        return 0, "", []

    monkeypatch.setattr(app, "process_folder", fake_process_folder)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[2]))

    tab = app.PreprocessTab()
    tab.folder_path = str(tmp_path)
    tab.scale_edit.setText("not-a-number")
    tab.run_preprocess()

    assert called["process_folder"] is False
    assert any("Scale must be a numeric value if provided." in w for w in warnings)


def test_run_preprocess_writes_metadata_and_syncs_analysis_tab(qapp, monkeypatch, tmp_path):
    analysis_tab = app.AnalysisTab()
    analysis_tab.cd68_combo.setCurrentText("blue")
    preprocess_tab = app.PreprocessTab(analysis_tab=analysis_tab)

    slides = tmp_path / "slides"
    slides.mkdir()
    # Counted in completion button text.
    (slides / "dummy_overlay.tif").write_bytes(b"")

    def fake_process_folder(
        folder_path,
        output_folder,
        hsv_bounds,
        min_area,
        padding,
        contour_kw,
        infarct_kw,
        thresh,
        pixel_scale,
    ):
        os.makedirs(output_folder, exist_ok=True)
        rows = [
            {
                "animal_id": "A1",
                "section_id": "1",
                "scale": 2.0,
                "downsample_factor": 1.0,
            }
        ]
        return 1, "ok", rows

    monkeypatch.setattr(app, "process_folder", fake_process_folder)
    monkeypatch.setattr(app.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args, **kwargs: None)

    preprocess_tab.folder_path = str(slides)
    preprocess_tab.thresh_spin.setValue(42)
    preprocess_tab.run_preprocess()

    out_folder = slides / "preprocessed_sections"
    metadata_path = out_folder / "preprocessed_detect_blue.csv"
    assert metadata_path.exists()
    df = pd.read_csv(metadata_path)
    assert len(df) == 1
    assert str(df.iloc[0]["animal_id"]) == "A1"

    assert analysis_tab.base_path == str(out_folder)
    assert analysis_tab.section_thresh_spin.value() == 42
    assert preprocess_tab.process_btn.text().startswith("Preprocessing Complete")
