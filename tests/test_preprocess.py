import os
import numpy as np
import pytest
import cv2
import tifffile as tiff

from infarquant import analysis as analysis_mod
from infarquant import preprocess as preprocess_mod


def test_preprocess_mask_detects_contours():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (70, 70), (255, 255, 255), -1)
    hsv_bounds = {
        "lower_H": 0,
        "lower_S": 0,
        "lower_V": 10,
        "upper_H": 255,
        "upper_S": 255,
        "upper_V": 255,
    }
    mask, contours = preprocess_mod.preprocess_mask(img, hsv_bounds)
    assert mask is not None
    assert len(contours) >= 1


def test_filter_contours_min_area():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (15, 15), 255, -1)   # small
    cv2.rectangle(mask, (30, 30), (80, 80), 255, -1)  # large
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = preprocess_mod.filter_contours(contours, min_area=500)
    assert len(filtered) == 1


def test_sort_contours_dynamic_row_major():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Two rows of rectangles
    cv2.rectangle(mask, (10, 10), (40, 40), 255, -1)
    cv2.rectangle(mask, (60, 10), (90, 40), 255, -1)
    cv2.rectangle(mask, (10, 80), (40, 110), 255, -1)
    cv2.rectangle(mask, (60, 80), (90, 110), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ordered = preprocess_mod.sort_contours_dynamic(contours)
    # Expect four contours in row-major order by bounding rect
    rects = [cv2.boundingRect(c) for c in ordered]
    ys = [r[1] for r in rects]
    xs = [r[0] for r in rects]
    assert ys[0] <= ys[1] <= ys[2] <= ys[3]
    assert xs[0] < xs[1]


def test_calculate_areas_mask_and_contour():
    left = np.zeros((50, 100), dtype=np.uint8)
    right = np.zeros((50, 100), dtype=np.uint8)
    left[:, :50] = 255
    right[:, 50:] = 255
    infarct = np.zeros((50, 100), dtype=np.uint8)
    infarct[:, :25] = 255
    contour = np.array([[[0, 0]], [[99, 0]], [[99, 49]], [[0, 49]]], dtype=np.int32)
    brain = {"left": contour, "right": contour, "whole": contour}

    mask_areas = analysis_mod.calculate_areas(left, right, infarct, contour, [], "mask", "max", brain)
    assert mask_areas["whole_area"] == 50 * 100
    assert mask_areas["infarct_area"] == 50 * 25

    contour_areas = analysis_mod.calculate_areas(left, right, infarct, contour, [contour], "contour", "max", brain)
    assert contour_areas["whole_area"] > 0
    assert contour_areas["infarct_area"] > 0


def test_process_folder_outputs_metadata(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    # Create synthetic reference and infarct images
    ref = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(ref, (10, 10), (70, 70), (255, 255, 255), -1)
    inf = ref.copy()
    tiff.imwrite(str(input_dir / "A1_overlay.tif"), ref)
    tiff.imwrite(str(input_dir / "A1_CD68.tif"), inf)

    hsv_bounds = {
        "lower_H": 0,
        "lower_S": 0,
        "lower_V": 10,
        "upper_H": 255,
        "upper_S": 255,
        "upper_V": 255,
    }
    count, log, rows = preprocess_mod.process_folder(
        str(input_dir),
        str(output_dir),
        hsv_bounds,
        min_area=200,
        padding=2,
        contour_keyword="overlay",
        infarct_keyword="CD68",
        thresh=10,
        pixel_scale=None,
    )
    assert count >= 1
    assert isinstance(count, int)
    assert isinstance(log, str)
    assert isinstance(rows, list)
    assert rows
    required_keys = {
        "animal_id",
        "section_id",
        "sections_found",
        "threshold",
        "scale",
        "downsampled",
        "downsample_factor",
        "reference_keyword",
        "infarct_keyword",
        "padding",
        "min_area",
        "date_preprocessed",
    }
    assert required_keys.issubset(set(rows[0].keys()))
    # Output folder should contain a subdir for A1
    assert (output_dir / "A1").exists()


def test_process_folder_keeps_reference_infarct_pairs_consistent_for_multi_file_key(tmp_path):
    input_dir = tmp_path / "input_multi"
    output_dir = tmp_path / "output_multi"
    input_dir.mkdir()

    def make_reference(side: str) -> np.ndarray:
        # Red slab provides the contour; blue marker encodes side.
        img = np.zeros((60, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (5, 5), (95, 55), (0, 0, 220), -1)
        if side == "left":
            cv2.rectangle(img, (12, 20), (26, 34), (255, 0, 0), -1)
        else:
            cv2.rectangle(img, (74, 20), (88, 34), (255, 0, 0), -1)
        return img

    def make_infarct(side: str) -> np.ndarray:
        # White marker encodes side on infarct image.
        img = np.zeros((60, 100, 3), dtype=np.uint8)
        if side == "left":
            cv2.rectangle(img, (12, 20), (26, 34), (255, 255, 255), -1)
        else:
            cv2.rectangle(img, (74, 20), (88, 34), (255, 255, 255), -1)
        return img

    # Filenames intentionally collapse to the same extract_key ("A1") under old logic.
    tiff.imwrite(str(input_dir / "A1 slideA merge.tif"), make_reference("left"))
    tiff.imwrite(str(input_dir / "A1 slideA CD68.tif"), make_infarct("left"))
    tiff.imwrite(str(input_dir / "A1 slideB merge.tif"), make_reference("right"))
    tiff.imwrite(str(input_dir / "A1 slideB CD68.tif"), make_infarct("right"))

    hsv_bounds = {
        "lower_H": 0,
        "lower_S": 0,
        "lower_V": 10,
        "upper_H": 255,
        "upper_S": 255,
        "upper_V": 255,
    }
    count, _, rows = preprocess_mod.process_folder(
        str(input_dir),
        str(output_dir),
        hsv_bounds,
        min_area=200,
        padding=2,
        contour_keyword="merge",
        infarct_keyword="CD68",
        thresh=10,
        pixel_scale=None,
    )
    assert count == 2
    assert len(rows) == 2

    slide_dir = output_dir / "A1"
    ref_files = sorted(slide_dir.glob("A1_reference_*.tif"))
    inf_files = sorted(slide_dir.glob("A1_infarct_*.tif"))
    assert len(ref_files) == 2
    assert len(inf_files) == 2

    def detect_side_from_reference(img: np.ndarray) -> str:
        blue = img[:, :, 0].astype(np.float32)
        mid = blue.shape[1] // 2
        return "left" if float(blue[:, :mid].sum()) > float(blue[:, mid:].sum()) else "right"

    def detect_side_from_infarct(img: np.ndarray) -> str:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mid = gray.shape[1] // 2
        return "left" if float(gray[:, :mid].sum()) > float(gray[:, mid:].sum()) else "right"

    for ref_path in ref_files:
        sec = ref_path.stem.split("_")[-1]
        inf_path = slide_dir / f"A1_infarct_{sec}.tif"
        assert inf_path.exists()
        ref_img = tiff.imread(str(ref_path))
        inf_img = tiff.imread(str(inf_path))
        assert detect_side_from_reference(ref_img) == detect_side_from_infarct(inf_img)
