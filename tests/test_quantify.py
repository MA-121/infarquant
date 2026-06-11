import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cv2

from infarquant import ui as app


def _write_results_csv(path: Path, rows) -> Path:
    """Write a minimal InfarQuant results CSV (one row per dict in ``rows``)."""
    base = {
        "animal_id": "A1",
        "section_id": 1,
        "filename": "A1_reference_1.png",
        "whole_area_px": 110,
        "left_area_px": 60,
        "right_area_px": 50,
        "infarct_area_px": 20,
        "pixel_to_um_scale": "NA",
        "date_analyzed": "2026-06-11",
    }
    records = []
    for r in rows:
        rec = dict(base)
        rec.update(r)
        records.append(rec)
    pd.DataFrame(records).to_csv(path, index=False, na_rep="NA")
    return path


def _write_reference_section(base: Path, animal: str = "A1", section: int = 1) -> Path:
    animal_dir = base / animal
    animal_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((12, 12, 3), dtype=np.uint8)
    ref_path = animal_dir / f"{animal}_reference_{section}.png"
    cv2.imwrite(str(ref_path), img)
    return ref_path


def _result_stub():
    return {
        "whole_area_px": 100,
        "left_area_px": 60,
        "right_area_px": 40,
        "infarct_area_px": 10,
        "brain_outline_threshold": 20,
        "CD68_threshold": 100,
        "exclude_threshold": 0,
        "FIXED_ROI": False,
    }


def test_quantify_left_side_math(qapp, monkeypatch, tmp_path):
    csv_path = _write_results_csv(tmp_path / "results.csv", [{}])
    monkeypatch.setattr(app.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *a, **k: None)

    tab = app.QuantifyTab()
    tab.results_csv_path = str(csv_path)
    tab.ipsi_combo.setCurrentText("Left")
    tab.run_quantify()

    out = pd.read_csv(tmp_path / "quantify_results.csv")
    assert len(out) == 1
    row = out.iloc[0]
    # Left ipsi -> ipsi = left = 60, contra = right = 50.
    assert row["ipsi_area_px"] == 60
    assert row["contra_area_px"] == 50
    assert row["ipsilesional_side"] == "Left"
    # (50 - (60 - 20)) / 50 * 100 = 20.0
    assert row["contra_adjusted_infarct_pct"] == pytest.approx(20.0)


def test_quantify_right_side_swaps_ipsi_contra(qapp, monkeypatch, tmp_path):
    csv_path = _write_results_csv(tmp_path / "results.csv", [{}])
    monkeypatch.setattr(app.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *a, **k: None)

    tab = app.QuantifyTab()
    tab.results_csv_path = str(csv_path)
    tab.ipsi_combo.setCurrentText("Right")
    tab.run_quantify()

    out = pd.read_csv(tmp_path / "quantify_results.csv")
    row = out.iloc[0]
    # Right ipsi -> ipsi = right = 50, contra = left = 60.
    assert row["ipsi_area_px"] == 50
    assert row["contra_area_px"] == 60
    assert row["ipsilesional_side"] == "Right"
    # (60 - (50 - 20)) / 60 * 100 = 50.0
    assert row["contra_adjusted_infarct_pct"] == pytest.approx(50.0)


def test_quantify_output_column_contract(qapp, monkeypatch, tmp_path):
    csv_path = _write_results_csv(tmp_path / "results.csv", [{}, {"section_id": 2}])
    monkeypatch.setattr(app.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *a, **k: None)

    tab = app.QuantifyTab()
    tab.results_csv_path = str(csv_path)
    tab.run_quantify()

    out_csv = tmp_path / "quantify_results.csv"
    assert out_csv.exists()  # named quantify_<input> next to the input
    out = pd.read_csv(out_csv)
    expected = [
        "animal_id",
        "section_id",
        "filename",
        "ipsilesional_side",
        "whole_area_px",
        "ipsi_area_px",
        "contra_area_px",
        "infarct_area_px",
        "contra_adjusted_infarct_pct",
        "date_quantified",
    ]
    assert list(out.columns) == expected
    assert out.columns[-1] == "date_quantified"
    assert len(out) == 2


def test_quantify_missing_column_warns_and_writes_nothing(qapp, monkeypatch, tmp_path):
    # A CSV that is not an InfarQuant results file (no hemisphere columns).
    bad = tmp_path / "not_results.csv"
    pd.DataFrame({"animal_id": ["A1"], "section_id": [1]}).to_csv(bad, index=False)

    warnings = []
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(app.QMessageBox, "information", lambda *a, **k: None)

    tab = app.QuantifyTab()
    tab.results_csv_path = str(bad)
    tab.run_quantify()

    assert any("does not look like" in w for w in warnings)
    assert not (tmp_path / "quantify_not_results.csv").exists()


def test_quantify_no_csv_warns(qapp, monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))

    tab = app.QuantifyTab()
    tab.results_csv_path = ""
    tab.run_quantify()
    assert any("Please load a results CSV first." in w for w in warnings)


def test_quantify_divide_by_zero_writes_na(qapp, monkeypatch, tmp_path):
    # Contralesional (right, since ipsi=Left) area is 0 -> percentage is undefined.
    csv_path = _write_results_csv(tmp_path / "results.csv", [{"right_area_px": 0}])
    monkeypatch.setattr(app.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *a, **k: None)

    tab = app.QuantifyTab()
    tab.results_csv_path = str(csv_path)
    tab.ipsi_combo.setCurrentText("Left")
    tab.run_quantify()

    raw = pd.read_csv(tmp_path / "quantify_results.csv", keep_default_na=False)
    assert raw.iloc[0]["contra_adjusted_infarct_pct"] == "NA"
    # The contralesional area itself is still recorded as the real value 0.
    assert str(raw.iloc[0]["contra_area_px"]) == "0"


def test_quantify_autofills_from_analysis(qapp, monkeypatch, tmp_path):
    _write_reference_section(tmp_path, animal="A1", section=1)
    monkeypatch.setattr(app, "qt_process_images", lambda *a, **k: _result_stub())
    monkeypatch.setattr(app.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *a, **k: None)

    analysis = app.AnalysisTab()
    quantify = app.QuantifyTab()
    analysis.quantify_tab = quantify  # wired by MainWindow in the real app
    analysis.base_path = str(tmp_path)
    analysis.method_edit.setText("contour")
    analysis.detection_edit.setText("max")
    analysis.exclude_combo.setCurrentText("none")
    analysis.run_analysis()

    out_csv = tmp_path / "results_detect-red_exclude-none_contour.csv"
    assert out_csv.exists()
    assert os.path.normpath(quantify.results_csv_path) == os.path.normpath(str(out_csv))
    assert quantify.csv_field.text() == quantify.results_csv_path
