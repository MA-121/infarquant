from __future__ import annotations

import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import tifffile as tiff
except ImportError:
    tiff = None


def process_folder(
    input_folder: str,
    output_folder: str,
    hsv_bounds: dict,
    min_area: int,
    padding: int,
    contour_keyword: str,
    infarct_keyword: str,
    thresh: int,
    pixel_scale: Optional[float],
) -> Tuple[int, str, List[dict]]:
    """
    Simplified version of process_folder for GUI with metadata collection.

    This function processes all TIFF images in `input_folder`, extracting
    individual section images based on contours detected from a reference channel.
    Cropped sections (reference and infarct channels) are optionally
    downsampled to improve performance and saved into the `output_folder`.
    A metadata list describing each section is returned along with the
    total section count and log messages.

    Args:
        input_folder: Directory containing the raw slide images.
        output_folder: Destination directory for cropped section images.
        hsv_bounds: HSV threshold dictionary used to segment the brain outlines.
        min_area: Minimum area in pixels for detected contours to be kept.
        padding: Padding in pixels added around each cropped section.
        contour_keyword: Keyword used to identify reference channel files.
        infarct_keyword: Keyword used to identify infarct channel files.
        thresh: Intensity threshold used for the HSV bounds (for logging).
        pixel_scale: Optional pixel-per-micron scale provided by the user.

    Returns:
        A tuple `(count, log_str, metadata_rows)` where `count` is the total
        number of extracted sections, `log_str` is a newline-separated log of
        operations, and `metadata_rows` is a list of dictionaries with
        preprocessing details for each section.
    """
    log_lines: List[str] = []
    os.makedirs(output_folder, exist_ok=True)
    # Collect TIFF filenames
    tif_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))])
    # Pair images by animal key extracted from filename
    def extract_key(fname: str) -> str:
        m = re.match(r"^([A-Za-z]+\d+(?:_\d+)?).*", fname)
        return m.group(1) if m else fname

    def normalize_for_match(fname: str) -> str:
        stem, _ = os.path.splitext(fname.lower())
        # Remove channel identifiers so same-section files can be matched.
        for kw in (contour_keyword, infarct_keyword):
            kw = (kw or "").strip().lower()
            if kw:
                stem = stem.replace(kw, " ")
        stem = re.sub(r"[_\-\s]+", " ", stem).strip()
        return stem

    def pair_score(ref_name: str, inf_name: str) -> tuple:
        ref_norm = normalize_for_match(ref_name)
        inf_norm = normalize_for_match(inf_name)
        ref_nums = set(re.findall(r"\d+", ref_norm))
        inf_nums = set(re.findall(r"\d+", inf_norm))
        number_overlap = len(ref_nums & inf_nums)
        prefix_len = len(os.path.commonprefix([ref_norm, inf_norm]))
        ratio = SequenceMatcher(None, ref_norm, inf_norm).ratio()
        return number_overlap, prefix_len, ratio

    paired: Dict[str, Dict[str, List[str]]] = {}
    for f in tif_files:
        key = extract_key(f)
        paired.setdefault(key, {"reference": [], "infarct": []})
        fname_lower = f.lower()
        # If both infarct and reference keywords appear, treat as reference to prioritize
        if infarct_keyword.lower() in fname_lower and contour_keyword.lower() in fname_lower:
            paired[key]["reference"].append(f)
        elif infarct_keyword.lower() in fname_lower:
            paired[key]["infarct"].append(f)
        elif contour_keyword.lower() in fname_lower:
            paired[key]["reference"].append(f)

    # Build explicit reference/infarct file pairs per key.
    file_pairs: List[Tuple[str, str, str]] = []
    for key in sorted(paired.keys()):
        refs = sorted(set(paired[key]["reference"]))
        infs = sorted(set(paired[key]["infarct"]))
        if not refs:
            continue
        if not infs:
            # Fallback to reference-only mode when no infarct channel exists.
            for ref_file in refs:
                file_pairs.append((key, ref_file, ref_file))
            continue
        available = infs.copy()
        for ref_file in refs:
            if not available:
                # More references than infarct files; duplicate reference as infarct for the remainder.
                file_pairs.append((key, ref_file, ref_file))
                continue
            best_idx = max(range(len(available)), key=lambda idx: pair_score(ref_file, available[idx]))
            infarct_file = available.pop(best_idx)
            file_pairs.append((key, ref_file, infarct_file))

    if not file_pairs:
        log_lines.append("No valid reference/infarct pairs found.")
        return 0, "\n".join(log_lines), []

    section_count = 0
    metadata_rows: List[dict] = []
    section_index_by_key = defaultdict(int)
    # Import datetime locally to avoid requiring a global import
    import datetime
    CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
    for key, reference_file, infarct_file in file_pairs:
        contour_path = os.path.join(input_folder, reference_file)
        infarct_path = os.path.join(input_folder, infarct_file)
        ref_img = tiff.imread(contour_path)
        inf_img = tiff.imread(infarct_path)
        # Generate mask and contours from reference image
        mask, contours = preprocess_mask(ref_img, hsv_bounds)
        valid_contours = filter_contours(contours, min_area)
        sorted_contours = sort_contours_dynamic(valid_contours)
        if not sorted_contours:
            log_lines.append(f"No contours found in {reference_file}")
            continue
        num_sections = len(sorted_contours)
        slide_dir = os.path.join(output_folder, key.replace(" ", "_"))
        os.makedirs(slide_dir, exist_ok=True)
        for cnt in sorted_contours:
            section_index_by_key[key] += 1
            section_id = section_index_by_key[key]
            # Crop reference and infarct channels using same transform
            ref_crop, M, dims = crop_rotate_section(cnt, ref_img, padding)
            inf_rotated = cv2.warpPerspective(inf_img, M, dims)
            # Align orientation of infarct channel
            if inf_rotated.shape[0] > inf_rotated.shape[1]:
                inf_rotated = cv2.rotate(inf_rotated, cv2.ROTATE_90_CLOCKWISE)
            inf_crop = cv2.copyMakeBorder(inf_rotated, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
            # Downsample both channels and determine downsample factor (min of the two)
            ref_crop_ds, ref_factor = downsample_section_image(ref_crop, max_width=5000)
            inf_crop_ds, inf_factor = downsample_section_image(inf_crop, max_width=5000)
            down_factor = min(ref_factor, inf_factor)
            downsampled = down_factor < 1.0
            # Save downsampled images
            tiff.imwrite(os.path.join(slide_dir, f"{key}_reference_{section_id}.tif"), ref_crop_ds, compression=None)
            tiff.imwrite(os.path.join(slide_dir, f"{key}_infarct_{section_id}.tif"), inf_crop_ds, compression=None)
            section_count += 1
            # Build metadata row
            meta_row = {
                "animal_id": key,
                "section_id": section_id,
                "sections_found": num_sections,
                "threshold": thresh,
                "scale": pixel_scale if pixel_scale is not None else "",
                "downsampled": downsampled,
                "downsample_factor": down_factor if downsampled else "",
                "reference_keyword": contour_keyword,
                "infarct_keyword": infarct_keyword,
                "source_reference_file": reference_file,
                "source_infarct_file": infarct_file,
                "padding": padding,
                "min_area": min_area,
                "date_preprocessed": CURRENT_DATE,
            }
            metadata_rows.append(meta_row)
        log_lines.append(f"Processed pair: {key} | ref={reference_file} | infarct={infarct_file}")
    log_lines.append(f"All slides processed. Total sections: {section_count}")
    return section_count, "\n".join(log_lines), metadata_rows

# -----------------------------------------------------------------------------
# Contour detection and section extraction helpers
# -----------------------------------------------------------------------------

def downsample_section_image(image: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    """
    Downsamples cropped section if width is greater than a specified maximum width.

    Returns a tuple of the downsampled image and the downsample factor.  If the
    input image width does not exceed `max_width`, the original image is
    returned along with a downsample factor of 1.0.  The downsample factor can
    be used downstream to adjust pixel-to-micron conversions.

    Args:
        image: The input image to potentially downsample.
        max_width: The maximum allowable width in pixels.

    Returns:
        (resized_image, factor) where `factor` is `new_width / original_width`.
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image, 1.0
    scale = max_width / w
    new_size = (max_width, int(h * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale

def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders four points to standardise perspective transformation.

    Args:
        pts: An array of four points representing a box.

    Returns:
        Reordered points in (top-left, top-right, bottom-right, bottom-left) order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


def crop_rotate_section(cnt: np.ndarray, image: np.ndarray, padding: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Extracts and rotates a section of an image based on a contour.

    Performs a perspective transform to extract the region defined by the
    contour, rotates the result to ensure the longest axis is horizontal,
    and adds padding.

    Args:
        cnt: The contour to extract.
        image: The source image.
        padding: Number of pixels for padding around the extracted section.

    Returns:
        A tuple containing the cropped, rotated, and padded image, the
        perspective transform matrix, and the dimensions (width, height) of
        the cropped section before padding.
    """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)
    rect_pts = order_points(box)
    width = int(np.linalg.norm(rect_pts[1] - rect_pts[0]))
    height = int(np.linalg.norm(rect_pts[3] - rect_pts[0]))
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    rotated = cv2.warpPerspective(image, M, (width, height))
    if rotated.shape[0] > rotated.shape[1]:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
    rotated_padded = cv2.copyMakeBorder(rotated, padding, padding, padding, padding,
                                        cv2.BORDER_CONSTANT, value=0)
    return rotated_padded, M, (width, height)


def preprocess_mask(image: np.ndarray, hsv_bounds: dict) -> Tuple[np.ndarray, list]:
    """Converts an image to HSV and applies thresholding and morphological operations.

    Args:
        image: Input image.
        hsv_bounds: HSV range for thresholding.

    Returns:
        A tuple containing the binary mask of detected regions and the list of contours.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_bounds['lower_H'], hsv_bounds['lower_S'], hsv_bounds['lower_V']])
    upper = np.array([hsv_bounds['upper_H'], hsv_bounds['upper_S'], hsv_bounds['upper_V']])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours


def filter_contours(contours: list, min_area: int) -> list:
    """Filters contours based on area and solidity."""
    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        hull = cv2.convexHull(cnt)
        if hull is None or cv2.contourArea(hull) == 0:
            continue
        solidity = cv2.contourArea(cnt) / cv2.contourArea(hull)
        if solidity < 0.8:
            continue
        valid_contours.append(cnt)
    return valid_contours


def sort_contours_dynamic(contours: list) -> list:
    """Sorts contours into rows (left-to-right) and orders rows top-to-bottom."""
    if not contours:
        return []
    contour_data = [(cnt, cv2.boundingRect(cnt)) for cnt in contours]
    median_height = np.median([h for _, (_, _, _, h) in contour_data])
    threshold = 0.5 * median_height
    contour_data.sort(key=lambda item: item[1][1] + item[1][3] / 2)
    rows = []
    current_row = [contour_data[0]]
    for item in contour_data[1:]:
        _, (x, y, w, h) = item
        if abs(y + h / 2 - (current_row[0][1][1] + current_row[0][1][3] / 2)) <= threshold:
            current_row.append(item)
        else:
            current_row.sort(key=lambda item: item[1][0])
            rows.append([c for c, _ in current_row])
            current_row = [item]
    if current_row:
        current_row.sort(key=lambda item: item[1][0])
        rows.append([c for c, _ in current_row])
    rows.sort(key=lambda row: np.mean([cv2.boundingRect(cnt)[1] for cnt in row]))
    return [cnt for row in rows for cnt in row]
