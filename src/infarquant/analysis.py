from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui
from PyQt5.QtWidgets import QInputDialog

# Global constant to control line thickness for drawings.
DRAW_THICKNESS = 8

# Instruction sink support for optional in-app instructions.
INSTRUCTION_SINK = None

def set_instruction_sink(sink) -> None:
    global INSTRUCTION_SINK
    INSTRUCTION_SINK = sink

def instructions_use_cv() -> bool:
    return INSTRUCTION_SINK is None

def update_instruction_list(title: str, lines: list) -> None:
    if INSTRUCTION_SINK is not None:
        INSTRUCTION_SINK.show_list(title, lines)

def update_instruction_table(title: str, pairs: list) -> None:
    if INSTRUCTION_SINK is not None:
        INSTRUCTION_SINK.show_table(title, pairs)


def calculate_areas(left_mask: np.ndarray,
                    right_mask: np.ndarray,
                    infarct_mask: np.ndarray,
                    brain_contour: np.ndarray,
                    infarct_contours: list,
                    method: str,
                    infarct_contour_detection: str,
                    brain_contours_dict: dict) -> dict:
    """
    Compute whole/hemisphere areas and infarct area. For method='contour',
    infarct area is computed from a filled contour mask (pixel count) to match
    the discrete definition used for 'infarct_area_positive'.
    """
    # Whole/hemisphere areas
    if method == "mask":
        left_area  = float(np.sum(left_mask  > 0))
        right_area = float(np.sum(right_mask > 0))
    else:
        left_area  = float(cv2.contourArea(brain_contours_dict.get("left"))  if brain_contours_dict.get("left")  is not None else 0.0)
        right_area = float(cv2.contourArea(brain_contours_dict.get("right")) if brain_contours_dict.get("right") is not None else 0.0)
    whole_area = left_area + right_area

    # Infarct area (discrete pixels for both modes)
    if method == "mask":
        infarct_area = float(np.sum(infarct_mask > 0))
    else:
        if infarct_contours:
            cmask = np.zeros_like(infarct_mask, dtype=np.uint8)
            if isinstance(infarct_contours, list):
                cv2.drawContours(cmask, infarct_contours, -1, 255, thickness=-1)
            else:
                cv2.drawContours(cmask, [infarct_contours], -1, 255, thickness=-1)
            infarct_area = float(np.sum(cmask > 0))
        else:
            infarct_area = 0.0

    return {
        "whole_area": whole_area,
        "left_area": left_area,
        "right_area": right_area,
        "infarct_area": infarct_area,
    }

# -----------------------------------------------------------------------------
# Helper functions for interactive segmentation
# -----------------------------------------------------------------------------

def create_resizable_window(name: str, image: Optional[np.ndarray] = None, scale_factor: float = 0.20) -> None:
    """Creates a resizable OpenCV window and optionally resizes it based on the provided image.

    Args:
        name: Name of the window.
        image: Optional image used to compute initial size.
        scale_factor: Fraction of screen width to use for the window width.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if image is not None:
        screen_width, screen_height = pyautogui.size()
        new_w = int(screen_width * scale_factor)
        h, w = image.shape[:2]
        new_h = int(h * (new_w / w))
        cv2.resizeWindow(name, new_w, new_h)


def create_trackbar_window(window_name: str, defaults: dict) -> None:
    """Creates a window with trackbars based on default threshold values."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for key, val in defaults.items():
        cv2.createTrackbar(key, window_name, val, 255, lambda x: None)
    # Adjust window size for better visibility of sliders
    height = 80 * len(defaults) + 40
    width = 600
    cv2.resizeWindow(window_name, width, height)


def get_trackbar_values(window_name: str, keys: list) -> dict:
    """Retrieves current values from the trackbars."""
    return {key: cv2.getTrackbarPos(key, window_name) for key in keys}


def segment_image_by_hsv(image: np.ndarray, lower_keys: list, upper_keys: list, positions: dict) -> np.ndarray:
    """Segments an image using HSV thresholds."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([positions[k] for k in lower_keys])
    upper = np.array([positions[k] for k in upper_keys])
    return cv2.inRange(hsv, lower, upper)


def get_largest_contour(mask: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Finds and returns the largest contour in a mask and its drawn mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest], -1, 255, thickness=6)
    return largest, contour_mask


def split_image(image: np.ndarray, split_points: list, brain_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Splits the image into left and right regions based on the provided split line."""
    h, w = image.shape[:2]
    if not split_points or len(split_points) < 2:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    split_points_sorted = sorted(split_points, key=lambda p: p[1])
    extended = [(split_points_sorted[0][0], 0)] + split_points_sorted + [(split_points_sorted[-1][0], h - 1)]
    x_vals = np.interp(np.arange(h), [p[1] for p in extended], [p[0] for p in extended])
    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        split_x = int(x_vals[y])
        left_mask[y, :split_x] = 255
        right_mask[y, split_x:] = 255
    left_mask = cv2.bitwise_and(left_mask, brain_mask)
    right_mask = cv2.bitwise_and(right_mask, brain_mask)
    return left_mask, right_mask


def draw_dashed_polyline(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], thickness: int = DRAW_THICKNESS,
                         dash_length: int = 5, gap_length: int = 10) -> None:
    """Draws a dashed polyline through the given points."""
    if pts is None or len(pts) < 2:
        return
    # Draw dashed segments between consecutive points
    for i in range(len(pts) - 1):
        pt1 = pts[i]
        pt2 = pts[i + 1]
        dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
        if dist == 0:
            continue
        dash_count = max(int(dist / (dash_length + gap_length)), 1)
        for j in range(dash_count):
            start_ratio = (j * (dash_length + gap_length)) / dist
            end_ratio = min((j * (dash_length + gap_length) + dash_length) / dist, 1.0)
            start_point = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio), int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
            end_point = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio), int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
            cv2.line(img, start_point, end_point, color, thickness)
    # Close the polyline by connecting the last point to the first
    if len(pts) >= 3:
        pt1 = pts[-1]
        pt2 = pts[0]
        dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
        if dist > 0:
            dash_count = max(int(dist / (dash_length + gap_length)), 1)
            for j in range(dash_count):
                start_ratio = (j * (dash_length + gap_length)) / dist
                end_ratio = min((j * (dash_length + gap_length) + dash_length) / dist, 1.0)
                start_point = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio), int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
                end_point = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio), int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
                cv2.line(img, start_point, end_point, color, thickness)


def patched_draw_pre_roi(base_folder: str) -> Optional[dict]:
    """
    Launch an interactive window allowing users to draw a pre-defined ROI on any
    section from the provided base directory. The user can flip the view,
    cycle between sections and channels, and save the resulting ROI.

    Args:
        base_folder: The path to the folder containing preprocessed sections
            (subdirectories per animal). Each section should have both a
            reference and infarct image file containing "reference" and
            "infarct" in the name.

    Returns:
        A dictionary containing the ROI specification if saved, otherwise
        None. The returned dictionary has keys:
            - 'name': user-provided name for the ROI
            - 'poly_norm': list of (x,y) coordinates normalised to the image
              width/height
            - 'centroid_norm': (x,y) of ROI centroid normalised to width/height
            - 'flip_h': whether a horizontal flip was applied during drawing
            - 'flip_v': whether a vertical flip was applied during drawing
            - 'file_path': full path to the saved JSON file
    """
    import json
    # Collect all pairs of reference/infarct images
    sections = []
    if not os.path.isdir(base_folder):
        return None
    for animal in sorted(os.listdir(base_folder)):
        subpath = os.path.join(base_folder, animal)
        if not os.path.isdir(subpath):
            continue
        # Gather reference files
        files = [f for f in os.listdir(subpath) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
        for f in files:
            if 'reference' in f.lower():
                ref_path = os.path.join(subpath, f)
                # Build matching infarct path by replacing 'reference' with 'infarct'
                parts = f.split('reference')
                if len(parts) == 2:
                    inf_candidate = parts[0] + 'infarct' + parts[1]
                    inf_path = os.path.join(subpath, inf_candidate)
                    if os.path.exists(inf_path):
                        sections.append({'ref': ref_path, 'inf': inf_path, 'animal': animal, 'fname': f})
                    else:
                        sections.append({'ref': ref_path, 'inf': ref_path, 'animal': animal, 'fname': f})
                else:
                    # fallback: use same image for both channels
                    sections.append({'ref': ref_path, 'inf': ref_path, 'animal': animal, 'fname': f})
    if not sections:
        pyautogui.alert("No sections found in the selected folder.")
        return None
    # Instruction window content
    instruction_list = [
        "Left-click: add vertex",
        "R: reset points",
        "S: set/save ROI",
        "C: change channel (reference/infarct)",
        "N: next section",
        "H: flip horizontally",
        "V: flip vertically",
        "Esc: exit without saving",
    ]
    instr_title = "Pre-draw ROI Instructions"
    # Create instruction window separate from drawing window
    draw_instruction_window(instr_title, instruction_list)
    # Random start index
    current_idx = 0
    # Channel toggle: 0 for reference, 1 for infarct
    channel_idx = 0
    # Flip flags
    flip_h = False
    flip_v = False
    # ROI points
    roi_points: list = []
    # Setup window
    window_name = "Pre-draw ROI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    # Mouse callback to record vertices
    def roi_mouse_cb(event, x, y, flags, param):
        nonlocal roi_points
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
    cv2.setMouseCallback(window_name, roi_mouse_cb)
    # Function to get current display image based on state
    def get_display_image() -> np.ndarray:
        sec = sections[current_idx]
        # Load appropriate image
        path = sec['ref'] if channel_idx == 0 else sec['inf']
        img = cv2.imread(path)
        if img is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        # Apply flips
        if flip_h:
            img = cv2.flip(img, 1)
        if flip_v:
            img = cv2.flip(img, 0)
        return img
    # Main loop
    saved_data: Optional[dict] = None
    while True:
        frame = get_display_image().copy()
        # Draw ROI if at least two points
        if len(roi_points) >= 2:
            # Compute convex hull to get closed polygon; flatten to (n,2)
            hull = cv2.convexHull(np.array(roi_points, dtype=np.int32))
            pts = hull.reshape(-1, 2)
            # Increase line thickness for better visibility
            dash_thickness = max(2, DRAW_THICKNESS * 2)
            draw_dashed_polyline(frame, pts, (255, 255, 0), thickness=dash_thickness, dash_length=5, gap_length=10)
        # Display current section and channel name as overlay
        sec_info = f"{sections[current_idx]['animal']} | {os.path.basename(sections[current_idx]['fname'])} | {'ref' if channel_idx==0 else 'inf'}"
        cv2.putText(frame, sec_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        # If instruction window closed, reopen to ensure user guidance
        if instructions_use_cv():
            try:
                if cv2.getWindowProperty(instr_title, cv2.WND_PROP_VISIBLE) < 1:
                    draw_instruction_window(instr_title, instruction_list)
            except Exception:
                pass
        if key == 27:  # Esc
            saved_data = None
            break
        if key in (ord('c'), ord('C')):
            channel_idx = 1 - channel_idx
            continue
        if key in (ord('n'), ord('N')):
            # Cycle to next section and reset ROI points
            current_idx = (current_idx + 1) % len(sections)
            roi_points.clear()
            continue
        if key in (ord('r'), ord('R')):
            roi_points.clear()
            continue
        if key in (ord('h'), ord('H')):
            flip_h = not flip_h
            # Reset ROI points when flipping for consistency
            # Clear ROI to avoid mismatched coordinates across flips
            roi_points.clear()
            continue
        if key in (ord('v'), ord('V')):
            flip_v = not flip_v
            roi_points.clear()
            continue
        if key in (ord('s'), ord('S')):
            # Save ROI only if at least three points
            if len(roi_points) < 3:
                pyautogui.alert("ROI requires at least 3 points to form a polygon.")
                continue
            # Build normalised coordinates relative to image size
            img = get_display_image()
            h, w = img.shape[:2]
            hull = cv2.convexHull(np.array(roi_points, dtype=np.int32)).reshape(-1, 2)
            poly_norm = [ (float(x)/w, float(y)/h) for (x, y) in hull ]
            # Compute centroid of ROI for later positioning
            cx = np.mean([p[0] for p in hull])
            cy = np.mean([p[1] for p in hull])
            centroid_norm = (cx / w, cy / h)
            # Prompt for name
            roi_name = pyautogui.prompt("Enter name for this ROI:")
            if roi_name is None or roi_name.strip() == "":
                pyautogui.alert("ROI name cannot be empty. Try again.")
                continue
            roi_name = roi_name.strip()
            # Build output path: save in base_folder as name+ROI.json
            fname = f"{roi_name}_ROI.json"
            file_path = os.path.join(base_folder, fname)
            data = {
                'name': roi_name,
                'poly_norm': poly_norm,
                'centroid_norm': centroid_norm,
                'flip_h': flip_h,
                'flip_v': flip_v,
                'file_path': file_path,
            }
            try:
                with open(file_path, 'w') as fjson:
                    json.dump(data, fjson)
                pyautogui.alert(f"ROI saved as {file_path}")
            except Exception as e:
                pyautogui.alert(f"Error saving ROI: {e}")
                data = None
            saved_data = data
            break
    cv2.destroyWindow(window_name)
    try:
        cv2.destroyWindow(instr_title)
    except Exception:
        pass
    return saved_data


def create_output_directory(directory: str, base_name: str = "results") -> str:
    """Creates an output directory to store results."""
    out_dir = os.path.join(directory, base_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_image(image: np.ndarray, path: str) -> None:
    """Saves an image to the specified path."""
    cv2.imwrite(path, image)


# =============================================================================
# Modified analysis functions for manual segmentation
# =============================================================================

# These functions adapt the original OpenCV-based interactive segmentation
# routines to better suit the GUI. They remove overlay text from the
# displayed images and present instructions via a docked Qt panel when
# available (falling back to OpenCV instruction windows if needed). They
# also use pyautogui.alert to warn the user when necessary (e.g. if
# attempting to proceed without drawing a midline).

points: list = []  # global list for storing midline segmentation points

def draw_instruction_window(window_name: str, lines: list, width: int = 400, height: int = 200) -> None:
    """Create a simple instruction window with multiple lines of text.

    Args:
        window_name: Name for the cv2 window.
        lines: List of strings to display.
        width: Width of the instruction window.
        height: Height of the instruction window.
    """
    if not instructions_use_cv():
        update_instruction_list(window_name, lines)
        return
    # Create a blank white image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # dark background
    y0 = 20
    for line in lines:
        cv2.putText(img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 25
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.resizeWindow(window_name, width, height)


def draw_instruction_table(window_name: str, pairs: list, width: int = 450, height_per_row: int = 30) -> None:
    """Create a more polished instruction window with Key/Action columns.

    Args:
        window_name: Name for the cv2 window.
        pairs: List of (key, action) tuples.
        width: Total width of the window.
        height_per_row: Height of each row in pixels.
    """
    if not instructions_use_cv():
        update_instruction_table(window_name, pairs)
        return
    # Determine dynamic sizes based on screen resolution
    try:
        screen_w, screen_h = pyautogui.size()
    except Exception:
        # Fallback values if screen size cannot be determined
        screen_w, screen_h = 1920, 1080
    # Use 30% of screen width for window width but not less than given width and not more than 550
    width = max(width, int(screen_w * 0.30))
    width = min(width, 550)
    # Row height scales with screen height; ensure at least 28px
    height_per_row = max(height_per_row, int(screen_h * 0.035))
    num_rows = len(pairs) + 1  # include header row
    height = num_rows * height_per_row + 20
    # Create blank canvas with light grey background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    # Define column widths: 30% for key, remainder for action
    col1_w = int(width * 0.28)
    col2_w = width - col1_w
    # Draw header background
    cv2.rectangle(img, (0, 0), (width, height_per_row), (220, 220, 220), thickness=-1)
    # Larger font for header
    header_font_scale = 0.8
    body_font_scale = 0.7
    cv2.putText(img, "Key", (10, int(height_per_row * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, header_font_scale, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, "Action", (col1_w + 10, int(height_per_row * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, header_font_scale, (30, 30, 30), 2, cv2.LINE_AA)
    # Draw rows
    for idx, (key, action) in enumerate(pairs):
        y_top = (idx + 1) * height_per_row
        # alternate row shading
        if idx % 2 == 0:
            cv2.rectangle(img, (0, y_top), (width, y_top + height_per_row), (255, 255, 255), thickness=-1)
        cv2.putText(img, str(key), (10, int(y_top + height_per_row * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, body_font_scale, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(img, str(action), (col1_w + 10, int(y_top + height_per_row * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, body_font_scale, (10, 10, 10), 2, cv2.LINE_AA)
        # draw vertical line separating columns
        cv2.line(img, (col1_w, y_top), (col1_w, y_top + height_per_row), (200, 200, 200), 1)
    # Draw outer border
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.resizeWindow(window_name, width, height)


def patched_get_segmentation_line(image: np.ndarray, display_info: str):
    """Allows the user to draw a midline segmentation on the provided image.

    Returns a tuple of (points, modified_image, (flip_h, flip_v)) or ("EXIT", None, None) if the user exits.
    """
    global points
    points = []
    mod_image = image.copy()
    flip_h = False
    flip_v = False
    window_name = f"Draw Split Line - {display_info}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    create_resizable_window(window_name, image=image, scale_factor=0.25)
    cv2.moveWindow(window_name, 50 + random.randint(0, 50), 50 + random.randint(0, 50))
    cv2.setMouseCallback(window_name, lambda event, x, y, flags, param: points.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)
    # Create instruction window as a table of key actions
    draw_instruction_table(
        "Midline Instructions",
        [
            ("Left Click", "Draw midline"),
            ("C", "Clear"),
            ("H", "Flip horizontal"),
            ("V", "Flip vertical"),
            ("A", "Accept"),
            ("Esc", "Exit"),
        ],
    )
    while True:
        # Ensure the instruction window remains visible; reopen if closed
        if instructions_use_cv():
            try:
                if cv2.getWindowProperty("Midline Instructions", cv2.WND_PROP_VISIBLE) < 1:
                    draw_instruction_window(
                        "Midline Instructions",
                        [
                            "Draw midline with left mouse clicks",
                            "C: Clear",
                            "H: Flip horizontal",
                            "V: Flip vertical",
                            "A: Accept",
                            "Esc: Exit",
                        ],
                    )
            except Exception:
                pass
        temp = mod_image.copy()
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(temp, points[i], points[i + 1], (255, 255, 255), DRAW_THICKNESS)
        for pt in points:
            cv2.circle(temp, pt, DRAW_THICKNESS, (255, 255, 255), -1)
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):
            points = []
        elif key == ord('h'):
            flip_h = not flip_h
            if flip_h and flip_v:
                mod_image = cv2.flip(image, -1)
            elif flip_h:
                mod_image = cv2.flip(image, 1)
            elif flip_v:
                mod_image = cv2.flip(image, 0)
            else:
                mod_image = image.copy()
        elif key == ord('v'):
            flip_v = not flip_v
            if flip_h and flip_v:
                mod_image = cv2.flip(image, -1)
            elif flip_h:
                mod_image = cv2.flip(image, 1)
            elif flip_v:
                mod_image = cv2.flip(image, 0)
            else:
                mod_image = image.copy()
        elif key == 27:
            cv2.destroyWindow(window_name)
            if instructions_use_cv():
                cv2.destroyWindow("Midline Instructions")
            return "EXIT", None, None
        elif key == ord('a') or key == 13:
            if len(points) < 2:
                pyautogui.alert("Invalid midline: Please draw a midline with at least two points.")
                points = []
                continue
            cv2.destroyWindow(window_name)
            if instructions_use_cv():
                cv2.destroyWindow("Midline Instructions")
            return points, mod_image, (flip_h, flip_v)


def patched_adjust_brain_contour_multi(ref_image: np.ndarray, split_points: list, section_contour_threshold: int, display_info: str = ""):
    """Adjusts the brain contour using HSV thresholding with user input.

    Displays the segmented whole, left and right images alongside a trackbar window and the in-app shortcuts panel.
    Returns (contours_dict, thresh_mask, final_positions) or ("EXIT", None, None) if exited.
    """
    trackbar_window = "Brain Threshold"
    # Use a shorter key name so it fits within the OpenCV trackbar label without being truncated
    user_defaults = {"Brain threshold": section_contour_threshold}
    create_trackbar_window(trackbar_window, user_defaults)
    cv2.moveWindow(trackbar_window, 50 + random.randint(0, 50), 10 + random.randint(0, 50))
    window_whole = f"Brain Whole - {display_info}"
    window_left = f"Brain Left - {display_info}"
    window_right = f"Brain Right - {display_info}"
    cv2.namedWindow(window_whole, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_left, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_right, cv2.WINDOW_NORMAL)
    create_resizable_window(window_whole, image=ref_image)
    create_resizable_window(window_left, image=ref_image)
    create_resizable_window(window_right, image=ref_image)
    cv2.moveWindow(window_whole, 50 + random.randint(0, 50), 200 + random.randint(0, 50))
    cv2.moveWindow(window_left, 700 + random.randint(0, 50), 200 + random.randint(0, 50))
    cv2.moveWindow(window_right, 50 + random.randint(0, 50), 700 + random.randint(0, 50))
    # Instructions window (table format)
    draw_instruction_table(
        "Brain Instructions",
        [
            ("Slider", "Adjust to outline brain"),
            ("A", "Accept"),
            ("Esc", "Exit"),
        ],
    )
    while True:
        # Ensure the instruction window remains visible; reopen if closed
        if instructions_use_cv():
            try:
                if cv2.getWindowProperty("Brain Instructions", cv2.WND_PROP_VISIBLE) < 1:
                    draw_instruction_window(
                        "Brain Instructions",
                        ["Adjust slider to outline brain", "A: Accept", "Esc: Exit"],
                    )
            except Exception:
                pass
        slider_vals = get_trackbar_values(trackbar_window, list(user_defaults.keys()))
        positions = {
            "Brain Lower H": 0,
            "Brain Lower S": 0,
            # Retrieve the value from the shorter label defined above
            "Brain Lower V": slider_vals["Brain threshold"],
            "Brain Upper H": 255,
            "Brain Upper S": 255,
            "Brain Upper V": 255,
        }
        lower_keys = ["Brain Lower H", "Brain Lower S", "Brain Lower V"]
        upper_keys = ["Brain Upper H", "Brain Upper S", "Brain Upper V"]
        thresh_mask = segment_image_by_hsv(ref_image, lower_keys, upper_keys, positions)
        whole_contour, _ = get_largest_contour(thresh_mask)
        if whole_contour is None:
            # no contour, skip draw
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                cv2.destroyWindow(trackbar_window)
                cv2.destroyWindow(window_whole)
                cv2.destroyWindow(window_left)
                cv2.destroyWindow(window_right)
                if instructions_use_cv():
                    cv2.destroyWindow("Brain Instructions")
                return "EXIT", None, None
            continue
        whole_mask_filled = np.zeros_like(thresh_mask)
        cv2.drawContours(whole_mask_filled, [whole_contour], -1, 255, thickness=cv2.FILLED)
        left_region_mask, right_region_mask = split_image(ref_image, split_points, np.ones(ref_image.shape[:2], dtype=np.uint8))
        left_brain_mask = cv2.bitwise_and(whole_mask_filled, left_region_mask)
        right_brain_mask = cv2.bitwise_and(whole_mask_filled, right_region_mask)
        left_contour, _ = get_largest_contour(left_brain_mask)
        right_contour, _ = get_largest_contour(right_brain_mask)
        brain_contours = {"whole": whole_contour, "left": left_contour, "right": right_contour}
        # Previews
        whole = ref_image.copy()
        cv2.drawContours(whole, [whole_contour], -1, (255, 255, 255), DRAW_THICKNESS)
        left_preview = cv2.bitwise_and(ref_image, ref_image, mask=left_region_mask)
        if left_contour is not None:
            cv2.drawContours(left_preview, [left_contour], -1, (255, 255, 255), DRAW_THICKNESS)
        right_preview = cv2.bitwise_and(ref_image, ref_image, mask=right_region_mask)
        if right_contour is not None:
            cv2.drawContours(right_preview, [right_contour], -1, (255, 255, 255), DRAW_THICKNESS)
        cv2.imshow(window_whole, whole)
        cv2.imshow(window_left, left_preview)
        cv2.imshow(window_right, right_preview)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(trackbar_window)
            cv2.destroyWindow(window_whole)
            cv2.destroyWindow(window_left)
            cv2.destroyWindow(window_right)
            if instructions_use_cv():
                cv2.destroyWindow("Brain Instructions")
            return "EXIT", None, None
        if key == ord('a') or key == 13:
            final_positions = positions.copy()
            cv2.destroyWindow(trackbar_window)
            cv2.destroyWindow(window_whole)
            cv2.destroyWindow(window_left)
            cv2.destroyWindow(window_right)
            if instructions_use_cv():
                cv2.destroyWindow("Brain Instructions")
            return brain_contours, thresh_mask, final_positions


def patched_adjust_infarct_threshold(
    infarct_img: np.ndarray,
    reference_img: np.ndarray,
    brain_contours: dict,
    split_points: list,
    infarct_contour_detection: str,
    display_info: str = "",
    cd68_color: str = "green",
    exclude_color: str = "red",
    cd68_start_val: int = 100,
    exclude_start_val: int = 175,
    fixed_roi_data: Optional[dict] = None,
) -> Optional[dict]:
    """Interactive adjustment of infarct segmentation thresholds.

    A single window is displayed showing either the infarct channel or the reference channel. Users
    can cycle through available channels using the 'V' hotkey. Only one channel is visible at a
    time to reduce visual clutter. A separate instruction window lists all hotkeys.

    Args:
        infarct_img: The image to use for infarct detection (usually the CD68 channel or merged).
        reference_img: The reference/overlay image used for channel comparison and annotation.
        brain_contours: Dictionary of contours for the whole, left and right brain regions.
        split_points: Points defining the midline for splitting left/right hemispheres.
        infarct_contour_detection: Strategy for selecting contours ('max' or 'all').
        display_info: A label describing the current section (for window titles).
        cd68_color: Colour channel for CD68 (infarct) segmentation.
        exclude_color: Colour channel to exclude (or 'none' to disable exclusion).
        cd68_start_val: Default threshold for the CD68 channel slider.
        exclude_start_val: Default threshold for the exclusion channel slider.

    Returns:
        A dictionary containing segmented views, final threshold positions and the selected
        infarct contours, or "EXIT"/None if the user cancelled. If `fixed_roi_data` is
        provided, the ROI shape is preloaded and cannot be edited; left-click will
        reposition the ROI across images.
    """
    # Map colours to channel indices
    COLOR_MAP = {"blue": 0, "green": 1, "red": 2}
    cd68_idx = COLOR_MAP.get(cd68_color, 1)
    exclude_idx = COLOR_MAP.get(exclude_color, 2)
    # Prepare trackbars; include exclude threshold only if an exclusion channel is specified
    trackbar_window = "Infarct Thresholds"
    user_defaults: dict = {"CD68 threshold": cd68_start_val}
    trackbar_keys = ["CD68 threshold"]
    if exclude_color != "none":
        user_defaults["exclude threshold"] = exclude_start_val
        trackbar_keys.append("exclude threshold")
    create_trackbar_window(trackbar_window, user_defaults)
    cv2.moveWindow(trackbar_window, 50 + random.randint(0, 50), 10 + random.randint(0, 50))
    # Prepare viewing window
    window_name = f"Infarct View - {display_info}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Precompute a merged image for saving (overlaid infarct on reference) but not necessarily used for display
    # We'll update this inside the loop
    # Make the infarct detection window larger by default (approx. 20% larger)
    create_resizable_window(window_name, image=infarct_img, scale_factor=0.24)
    cv2.moveWindow(window_name, 50 + random.randint(0, 50), 200 + random.randint(0, 50))
    # Determine if a fixed ROI is in use
    roi_fixed = fixed_roi_data is not None
    # Precompute fixed ROI hull and centroid positions (normalised) if provided
    fixed_poly_norm = None
    fixed_centroid_norm = None
    if roi_fixed and fixed_roi_data:
        # Retrieve and optionally flip the normalised polygon and centroid
        poly_norm_orig = fixed_roi_data.get('poly_norm')
        centroid_norm_orig = fixed_roi_data.get('centroid_norm')
        flip_h_flag = bool(fixed_roi_data.get('flip_h', False))
        flip_v_flag = bool(fixed_roi_data.get('flip_v', False))
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
    # Instruction table
    # Build instruction entries; omit ROI editing controls if a fixed ROI is in use
    instruction_entries = [
        ("Sliders", "Adjust CD68/exclude"),
        ("C", "Cycle contour"),
        ("M", "Add contour"),
        ("D", "Clear stored contours"),
    ]
    if not roi_fixed:
        instruction_entries.extend([
            ("R", "Reset ROI"),
            ("S", "Set ROI"),
        ])
    else:
        instruction_entries.append(("Click", "Reposition ROI"))
    instruction_entries.extend([
        ("N", "No infarct"),
        ("Z", "Change channel"),
        ("A", "Accept"),
        ("Esc", "Exit"),
    ])
    draw_instruction_table("Infarct Instructions", instruction_entries)
    # Mouse callback for ROI selection or repositioning
    roi_points: list = []  # Only used when drawing a new ROI
    roi_mask: Optional[np.ndarray] = None
    roi_hull_points: Optional[np.ndarray] = None
    # For fixed ROI we track offsets relative to the centroid
    roi_offset_dx: float = 0.0
    roi_offset_dy: float = 0.0
    def roi_mouse_callback(event, x, y, flags, param):
        nonlocal roi_points, roi_hull_points, roi_mask, roi_offset_dx, roi_offset_dy
        if roi_fixed:
            # In fixed mode, clicking repositions the ROI based on centroid
            if event == cv2.EVENT_LBUTTONDOWN and fixed_poly_norm is not None and fixed_centroid_norm is not None:
                img_h, img_w = infarct_img.shape[:2]
                # Original centroid (in pixels) based on normalised centroid
                orig_cx = fixed_centroid_norm[0] * img_w
                orig_cy = fixed_centroid_norm[1] * img_h
                # Proposed offsets relative to click
                dx = x - orig_cx
                dy = y - orig_cy
                # Compute new hull points based on proposed offset
                pts = []
                for xn, yn in fixed_poly_norm:
                    px = xn * img_w + dx
                    py = yn * img_h + dy
                    pts.append((px, py))
                pts_arr = np.array(pts, dtype=np.int32)
                # Compute bounding box and adjust if outside image bounds
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
                # Apply shift
                pts_arr[:, 0] = np.clip(pts_arr[:, 0] + shift_x, 0, img_w - 1)
                pts_arr[:, 1] = np.clip(pts_arr[:, 1] + shift_y, 0, img_h - 1)
                roi_hull_points = pts_arr.reshape(-1, 1, 2)
                # Generate mask
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask, [roi_hull_points], 255)
                roi_mask = mask
                # Update offsets for record (optional)
                roi_offset_dx = dx + shift_x
                roi_offset_dy = dy + shift_y
        else:
            # Interactive drawing mode: add vertices on left-click
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points.append((x, y))
    cv2.setMouseCallback(window_name, roi_mouse_callback)
    # Variables to handle contours and multi-contours
    current_index = 0
    selected_infarct_contour = None
    multi_contours: list = []
    # Channel view toggling: start with infarct image
    views = [
        ("infarct", infarct_img),
        ("reference", reference_img),
    ]
    view_index = 0
    final_positions = None
    save_whole = None
    save_left = None
    save_right = None

    # If a fixed ROI is provided, compute its initial position and mask now
    if roi_fixed and fixed_poly_norm is not None and fixed_centroid_norm is not None:
        img_h, img_w = infarct_img.shape[:2]
        # Points at nominal position (no offset)
        pts = []
        for xn, yn in fixed_poly_norm:
            pts.append((xn * img_w, yn * img_h))
        pts_arr = np.array(pts, dtype=np.int32)
        # Compute bounding and clamp within image
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
    while True:
        # Ensure the instruction window remains visible; reopen if closed
        if instructions_use_cv():
            try:
                if cv2.getWindowProperty("Infarct Instructions", cv2.WND_PROP_VISIBLE) < 1:
                    draw_instruction_window(
                        "Infarct Instructions",
                        [
                            "Adjust CD68/exclude sliders",
                            "C: Cycle contour",
                            "M: Add contour",
                            "D: Clear stored contours",
                            "R: Reset ROI",
                            "S: Set ROI",
                            "N: No infarct",
                            "Z: Change channel",
                            "A: Accept",
                            "Esc: Exit",
                        ],
                    )
            except Exception:
                pass
        # Read current trackbar values
        slider_vals = get_trackbar_values(trackbar_window, trackbar_keys)
        # Build positions dict with full keys used elsewhere
        positions = {
            'Infarct Lower H': 0,
            'Infarct Lower S': 0,
            'Infarct Lower V': slider_vals.get("CD68 threshold", cd68_start_val),
            'Infarct Upper H': 255,
            'Infarct Upper S': 255,
            'Infarct Upper V': 255,
            'exclude threshold': slider_vals.get("exclude threshold", exclude_start_val),
        }
        # Create masks for current thresholds
        # Threshold the CD68 channel on the infarct image. This channel corresponds to the infarct marker
        cd68_channel = infarct_img[:, :, cd68_idx]
        _, mask_cd68 = cv2.threshold(cd68_channel, positions['Infarct Lower V'], 255, cv2.THRESH_BINARY)
        # For the exclusion channel, use the reference image rather than the infarct image.  In many
        # datasets the channel to exclude (e.g. GFAP, DAPI) is present in the reference image and not
        # in the infarct marker image.  Using the reference image here ensures the exclusion slider
        # operates on the correct channel.  If the user selected 'none', produce a full mask of ones.
        if exclude_color != "none":
            exclude_channel = reference_img[:, :, exclude_idx]
            _, mask_exclude = cv2.threshold(
                exclude_channel,
                positions['exclude threshold'],
                255,
                cv2.THRESH_BINARY_INV,
            )
        else:
            mask_exclude = np.ones_like(mask_cd68, dtype=np.uint8) * 255
        merged_mask = cv2.bitwise_and(mask_cd68, mask_exclude)
        # Apply ROI mask if defined
        if roi_mask is not None:
            merged_mask = cv2.bitwise_and(merged_mask, roi_mask)
        # Find contours
        infarct_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_contours: list = []
        if infarct_contour_detection == "max" and infarct_contours:
            sorted_contours = sorted(infarct_contours, key=cv2.contourArea, reverse=True)
            top_contours = sorted_contours[:5]
            if current_index >= len(top_contours):
                current_index = 0
            if top_contours:
                selected_infarct_contour = top_contours[current_index]
        else:
            selected_infarct_contour = None
        # Build display frame based on current channel view
        view_label, base_img = views[view_index]
        display_frame = base_img.copy()
        # Draw brain outline on display frame
        if brain_contours is not None and brain_contours.get("whole") is not None:
            cv2.drawContours(display_frame, [brain_contours["whole"]], -1, (255, 255, 255), DRAW_THICKNESS)
        # Draw infarct contour(s) on display frame
        if infarct_contour_detection == "max" and selected_infarct_contour is not None:
            cv2.drawContours(display_frame, [selected_infarct_contour], -1, (0, 255, 0), DRAW_THICKNESS)
        elif infarct_contours:
            cv2.drawContours(display_frame, infarct_contours, -1, (0, 255, 0), max(1, DRAW_THICKNESS - 1))
        if multi_contours:
            for cnt in multi_contours:
                cv2.drawContours(display_frame, [cnt], -1, (0, 0, 255), DRAW_THICKNESS)
        # Draw ROI overlays
        if roi_mask is not None and roi_hull_points is not None:
            # Increase ROI outline thickness: use 2x the global DRAW_THICKNESS
            dash_thickness = max(2, DRAW_THICKNESS * 2)
            draw_dashed_polyline(display_frame, roi_hull_points.reshape(-1, 2), (255, 255, 0), thickness=dash_thickness, dash_length=5, gap_length=10)
        elif roi_points:
            dash_thickness = max(2, DRAW_THICKNESS * 2)
            draw_dashed_polyline(display_frame, np.array(roi_points, np.int32).reshape(-1, 2), (255, 255, 0), thickness=dash_thickness, dash_length=5, gap_length=10)
        # Save clean copies for output BEFORE drawing ROI overlays on individual channel views
        # We need to compute the segmented whole/left/right images using the infarct image for saving
        # Prepare these images at each iteration; they will be updated upon acceptance
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
        # Save copies for later on acceptance
        save_whole = whole_out.copy()
        save_left = left_out.copy()
        save_right = right_out.copy()
        # Show display frame
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF
        # Handle key events
        if key == 27:
            # Escape key: abort
            cv2.destroyWindow(trackbar_window)
            cv2.destroyWindow(window_name)
            if instructions_use_cv():
                cv2.destroyWindow("Infarct Instructions")
            return "EXIT"
        if key == ord('n'):
            # No infarct: return immediately with empty contours
            result = {
                "segmented": {"whole": infarct_img.copy(), "left": infarct_img.copy(), "right": infarct_img.copy()},
                "final_positions": positions,
                "selected_infarct_contour": None,
                "no_infarct": True,
            }
            cv2.destroyWindow(trackbar_window)
            cv2.destroyWindow(window_name)
            if instructions_use_cv():
                cv2.destroyWindow("Infarct Instructions")
            return result
        if key == ord('r'):
            # Reset ROI: only available in interactive drawing mode
            if not roi_fixed:
                roi_points.clear()
                roi_mask = None
                roi_hull_points = None
            continue
        if key == ord('s'):
            # Set ROI when at least 3 points defined (interactive mode only)
            if not roi_fixed:
                if len(roi_points) >= 3:
                    pts = np.array(roi_points, np.int32)
                    hull = cv2.convexHull(pts)
                    roi_hull_points = hull
                    roi_mask = np.zeros(infarct_img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(roi_mask, [hull], 255)
                    # Clear points after locking ROI
                    roi_points.clear()
                else:
                    pyautogui.alert("Need at least 3 points to set ROI.")
            continue
        if key == ord('c'):
            # Cycle through top contours if detection mode is 'max'
            if infarct_contour_detection == "max" and infarct_contours:
                current_index = (current_index + 1) % len(top_contours)
            continue
        if key == ord('m'):
            # Add currently selected contour to multi-contour list
            if infarct_contour_detection == "max" and infarct_contours and selected_infarct_contour is not None:
                multi_contours.append(selected_infarct_contour)
                current_index = (current_index + 1) % len(top_contours)
            continue
        if key == ord('d'):
            # Clear multi-contours
            multi_contours = []
            continue
        if key == ord('z') or key == ord('Z'):
            # Cycle channel view
            view_index = (view_index + 1) % len(views)
            continue
        if key == ord('a') or key == 13:
            # Accept segmentation and exit loop
            final_positions = positions.copy()
            cv2.destroyWindow(trackbar_window)
            cv2.destroyWindow(window_name)
            if instructions_use_cv():
                cv2.destroyWindow("Infarct Instructions")
            break
    # Build final contours list if multi-contours present
    if multi_contours:
        all_contours = multi_contours.copy()
        if selected_infarct_contour is not None:
            all_contours.append(selected_infarct_contour)
        selected_infarct_contour = all_contours
    # Return segmented images and positions; include flag whether a fixed ROI was used
    return {
        "segmented": {"whole": save_whole, "left": save_left, "right": save_right},
        "final_positions": final_positions if final_positions is not None else positions,
        "selected_infarct_contour": selected_infarct_contour,
        "fixed_roi_used": roi_fixed,
        "FIXED_ROI": roi_fixed,
        # Include the final ROI mask when using a fixed ROI so that downstream
        # processing can compute ROI-based metrics. None when no ROI is used.
        "roi_mask": roi_mask,
        # When the user accepts (A), no_infarct is always False
        "no_infarct": False,
    }


def patched_process_images(
    ref_image_path: str,
    method: str,
    infarct_contour_detection: str,
    section_contour_threshold: int,
    cd68_color: str,
    exclude_color: str,
    cd68_start_val: int,
    exclude_start_val: int,
    fixed_roi_data: Optional[dict] = None,
    background_percent: float = 10.0,
) -> Optional[dict]:
    """Process a single reference image using the patched interactive functions.

    This wrapper calls the interactive segmentation functions to obtain the midline,
    brain contour and infarct contour. It then calculates areas and additional
    intensity metrics and saves segmented views.
    """
    import re

    def _apply_flips(img: np.ndarray, flip_flags) -> np.ndarray:
        """Apply flips to `img` based on flip_flags from midline step.

        Accepts:
          - tuple(bool, bool): (flip_h, flip_v)
          - legacy ints: 1-> vertical (x-axis) flip, 2-> horizontal (y-axis) flip, 3-> both
        """
        # Normalize to booleans
        if isinstance(flip_flags, tuple):
            flip_h, flip_v = bool(flip_flags[0]), bool(flip_flags[1])
        elif isinstance(flip_flags, int):
            # backward compatibility with the previous mapping
            if flip_flags == 1:      # vertical
                flip_h, flip_v = False, True
            elif flip_flags == 2:    # horizontal
                flip_h, flip_v = True, False
            elif flip_flags == 3:    # both
                flip_h, flip_v = True, True
            else:
                flip_h, flip_v = False, False
        else:
            flip_h, flip_v = False, False

        # OpenCV: 0 = vertical (x-axis), 1 = horizontal (y-axis), -1 = both
        if flip_h and flip_v:
            return cv2.flip(img, -1)
        if flip_h:
            return cv2.flip(img, 1)
        if flip_v:
            return cv2.flip(img, 0)
        return img

    # Load reference image
    ref_img = cv2.imread(ref_image_path)
    if ref_img is None:
        print("Error loading image.")
        return None
    display_info = os.path.basename(ref_image_path)

    # Step 1: Obtain midline and aligned reference image
    result = patched_get_segmentation_line(ref_img, display_info)
    if result[0] == "EXIT" or result[0] is None:
        return "EXIT"
    split_points, mod_ref_img, flip_flags = result  # flip_flags is (flip_h, flip_v)

    # Step 1b: Load infarct image and apply the SAME flips so it matches the reference
    folder = os.path.dirname(ref_image_path)
    base = os.path.basename(ref_image_path)
    m = re.search(r"(.*)_reference_(\d+)(\.(?:tif|tiff|png|jpg|jpeg))$", base, re.IGNORECASE)
    infarct_path = None
    if m:
        animal_prefix, sec_num, ext = m.groups()
        candidate = f"{animal_prefix}_infarct_{sec_num}{ext}"
        candidate_path = os.path.join(folder, candidate)
        if os.path.exists(candidate_path):
            infarct_path = candidate_path

    if infarct_path is None:
        infarct_img = mod_ref_img.copy()  # use flipped reference if no separate infarct
    else:
        inf_raw = cv2.imread(infarct_path)
        if inf_raw is None:
            infarct_img = mod_ref_img.copy()
        else:
            infarct_img = _apply_flips(inf_raw, flip_flags)

    # Step 2: brain segmentation on the (already flipped) reference image
    brain_contours_result = patched_adjust_brain_contour_multi(
        mod_ref_img, split_points, section_contour_threshold, display_info
    )
    if brain_contours_result[0] == "EXIT" or brain_contours_result[0] is None:
        return "EXIT"
    brain_contours, brain_thresh_mask, final_brain_positions = brain_contours_result
    left_mask, right_mask = split_image(mod_ref_img, split_points, brain_thresh_mask)

    # Step 3: infarct segmentation on the (now correctly flipped) infarct image
    seg_result = patched_adjust_infarct_threshold(
        infarct_img,
        mod_ref_img,
        brain_contours,
        split_points,
        infarct_contour_detection,
        display_info,
        cd68_color,
        exclude_color,
        cd68_start_val,
        exclude_start_val,
        fixed_roi_data=fixed_roi_data,
    )
    if seg_result == "EXIT" or seg_result is None:
        return "EXIT"

    final_infarct_positions = seg_result["final_positions"]
    selected_infarct_contour = seg_result.get("selected_infarct_contour", None)
    if isinstance(selected_infarct_contour, np.ndarray):
        selected_infarct_contour = [selected_infarct_contour]

    # Build infarct mask from thresholds
    COLOR_MAP = {"blue": 0, "green": 1, "red": 2}
    if seg_result.get("no_infarct", False):
        infarct_mask = np.zeros(infarct_img.shape[:2], dtype=np.uint8)
        infarct_contours: list = []
    else:
        cd68_idx = COLOR_MAP.get(cd68_color, 1)
        cd68_channel = infarct_img[:, :, cd68_idx]
        # Threshold the CD68 (infarct) channel
        _, mask_cd68 = cv2.threshold(cd68_channel, final_infarct_positions['Infarct Lower V'], 255, cv2.THRESH_BINARY)
        # For exclusion, use the flipped reference image rather than the infarct image.  This mirrors the
        # interactive adjustment and allows exclusion of channels stored in the reference file.  If the
        # user disabled exclusion ("none"), create a full mask of ones.
        if exclude_color == "none":
            mask_exclude = np.ones_like(mask_cd68, dtype=np.uint8) * 255
        else:
            exclude_idx = COLOR_MAP[exclude_color] if exclude_color != "none" else None
            # Use mod_ref_img (flipped reference image) for the exclusion channel
            exclude_channel = mod_ref_img[:, :, exclude_idx]
            _, mask_exclude = cv2.threshold(
                exclude_channel,
                final_infarct_positions['exclude threshold'],
                255,
                cv2.THRESH_BINARY_INV,
            )
        infarct_mask = cv2.bitwise_and(mask_cd68, mask_exclude)
        infarct_contours, _ = cv2.findContours(infarct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if infarct_contour_detection == "max" and selected_infarct_contour is not None:
            infarct_contours = selected_infarct_contour if isinstance(selected_infarct_contour, list) else [selected_infarct_contour]

    # Compute metrics
    areas = calculate_areas(
        left_mask=left_mask,
        right_mask=right_mask,
        infarct_mask=infarct_mask,
        brain_contour=brain_contours.get("whole"),
        infarct_contours=infarct_contours,
        method=method,
        infarct_contour_detection=infarct_contour_detection,
        brain_contours_dict=brain_contours,
    )

    # Compute additional metrics: positive area and intensity averages. These values are
    # computed differently when a fixed ROI is used versus a manually drawn infarct
    # contour. Initialise metrics here and override as appropriate.
    infarct_area_positive: float = 0.0
    infarct_intensity_avg: float = 0.0
    infarct_area_positive_intensity_avg: float = 0.0
    # Determine if a fixed ROI was used and retrieve its mask from seg_result (provided by patched_adjust_infarct_threshold)
    fixed_roi_used = bool(seg_result.get("FIXED_ROI", False))
    roi_mask = seg_result.get("roi_mask", None)
    if fixed_roi_used and roi_mask is not None:
        # When a fixed ROI is used, treat the entire ROI area as the infarct. Compute
        # area and intensity metrics over this region. The positive area is the
        # fraction of ROI pixels above threshold (and not excluded by the exclude channel).
        roi_bool = (roi_mask > 0)
        # Override the infarct area in the areas dict with the ROI area
        areas["infarct_area"] = float(np.sum(roi_bool))
        # Compute positive mask: threshold-positive pixels within the ROI
        positive_mask_bool = np.logical_and((infarct_mask > 0), roi_bool)
        infarct_area_positive = float(np.sum(positive_mask_bool))
        # Intensity averages: all ROI pixels
        all_vals = infarct_img[:, :, COLOR_MAP.get(cd68_color, 1)][roi_bool]
        if all_vals.size > 0:
            infarct_intensity_avg = float(np.mean(all_vals))
        # Intensity average of threshold-positive pixels within ROI
        pos_vals = infarct_img[:, :, COLOR_MAP.get(cd68_color, 1)][positive_mask_bool]
        if pos_vals.size > 0:
            infarct_area_positive_intensity_avg = float(np.mean(pos_vals))
    else:
        # Standard case: use infarct contours derived from thresholding. Compute
        # positive area and intensity averages within the contour mask.
        if infarct_contours:
            contour_mask = np.zeros(infarct_img.shape[:2], dtype=np.uint8)
            for cnt in infarct_contours:
                cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1)
            # positive_mask marks pixels above threshold inside the contour
            positive_mask = cv2.bitwise_and(infarct_mask, contour_mask)
            infarct_area_positive = float(np.sum(positive_mask > 0))
            # Mean intensity of all pixels inside the contour (irrespective of threshold)
            vals_all = infarct_img[:, :, COLOR_MAP.get(cd68_color, 1)][contour_mask > 0]
            if vals_all.size > 0:
                infarct_intensity_avg = float(np.mean(vals_all))
            # Mean intensity of threshold-positive pixels inside the contour
            vals_pos = infarct_img[:, :, COLOR_MAP.get(cd68_color, 1)][positive_mask > 0]
            if vals_pos.size > 0:
                infarct_area_positive_intensity_avg = float(np.mean(vals_pos))

    areas.pop("hemisphere_diff", None)
    areas.pop("hemisphere_diff_pct", None)
    areas["infarct_area_positive"] = infarct_area_positive
    areas["infarct_area_intensity_avg"] = infarct_intensity_avg
    # Store new metric: mean intensity of threshold-positive pixels (empty or float)
    areas["infarct_area_positive_intensity_avg"] = infarct_area_positive_intensity_avg
    areas["brain_outline_threshold"] = final_brain_positions["Brain Lower V"]
    areas["CD68_threshold"] = final_infarct_positions['Infarct Lower V']
    areas["exclude_threshold"] = final_infarct_positions['exclude threshold'] if exclude_color != "none" else ""

    # --------------------------------------------------------------------------
    # Compute background intensity and normalised average intensity.
    # The background is defined as the mean of the lowest-intensity pixels within
    # the brain contour on the infarct channel.  The fraction of pixels used is
    # specified by `background_percent` (default 10), for example a value of
    # 10 means the darkest 10Â % of pixels will be averaged.  This normalisation
    # accounts for section-to-section variability in exposure.  If the brain
    # mask or channel values are unavailable, leave blank.
    try:
        # Determine channel index for cd68_color; fallback to green (1)
        cd68_idx_norm = {"blue": 0, "green": 1, "red": 2}.get(cd68_color, 1)
        normalization_value = ""
        normalized_intensity = ""
        # Use the brain_thresh_mask defined earlier to extract brain pixels
        # brain_thresh_mask is defined earlier in this function and corresponds to the mask of the entire brain
        mask_arr = brain_thresh_mask
        if mask_arr is not None and infarct_img is not None:
            # Extract intensity values from the infarct image within the brain mask
            channel_vals = infarct_img[:, :, cd68_idx_norm][mask_arr > 0]
            if channel_vals.size > 0:
                # Sort values and take the lowest-intensity fraction defined by background_percent
                sorted_vals = np.sort(channel_vals.flatten())
                n_vals = max(int(len(sorted_vals) * (background_percent / 100.0)), 1)
                bottom_vals = sorted_vals[:n_vals]
                normalization_value_float = float(np.mean(bottom_vals))
                normalization_value = normalization_value_float
                if normalization_value_float > 0:
                    # Normalize the infarct intensity average by the background value
                    normalized_intensity = float(infarct_intensity_avg) / normalization_value_float if infarct_intensity_avg else 0.0
        # Compute normalised intensity for threshold-positive pixels
        infarct_intensity_positive_avg_normalized = ""
        try:
            if normalization_value_float > 0:
                infarct_intensity_positive_avg_normalized = (
                    float(infarct_area_positive_intensity_avg) / normalization_value_float
                ) if infarct_area_positive_intensity_avg else 0.0
        except Exception:
            infarct_intensity_positive_avg_normalized = ""
        # Assign to areas dictionary
        areas["background_intensity"] = normalization_value
        areas["infarct_intensity_avg_normalized"] = normalized_intensity
        areas["infarct_intensity_positive_avg_normalized"] = infarct_intensity_positive_avg_normalized
    except Exception:
        areas["background_intensity"] = ""
        areas["infarct_intensity_avg_normalized"] = ""
        areas["infarct_intensity_positive_avg_normalized"] = ""

    # Include fixed ROI flag in areas for downstream storage. Use False if not provided.
    try:
        areas["FIXED_ROI"] = bool(seg_result.get("FIXED_ROI", False))
    except Exception:
        areas["FIXED_ROI"] = False

    # Save segmented images for later review. Generate outputs for each available channel (infarct and reference)
    # Derive a base name for saving results by stripping the channel identifier from the
    # reference filename.  For example, 'RK822_reference_10.tif' becomes 'RK822_10'.  This avoids
    # repeatedly saving files prefixed with 'reference' when saving other channels.
    import re as _re
    raw_base = os.path.splitext(os.path.basename(ref_image_path))[0]
    # Remove '_reference_' or '_infarct_' identifiers if present
    base_name_infarct = _re.sub(r"_(?:reference|infarct)_", "_", raw_base)
    output_dir = create_output_directory(os.path.dirname(ref_image_path), base_name="results")
    # Determine final contours list for drawing
    draw_contours = []
    if seg_result.get("no_infarct", False):
        draw_contours = []
    elif isinstance(infarct_contours, list):
        draw_contours = infarct_contours
    else:
        draw_contours = [infarct_contours]
    # Define channel bases
    channel_views = {
        "infarct": infarct_img,
        "reference": mod_ref_img,
    }
    for channel_name, base_img in channel_views.items():
        # Create images per region
        whole_out = base_img.copy()
        left_mask_split = split_image(base_img, split_points, np.ones(base_img.shape[:2], dtype=np.uint8))[0]
        right_mask_split = split_image(base_img, split_points, np.ones(base_img.shape[:2], dtype=np.uint8))[1]
        left_out = cv2.bitwise_and(base_img, base_img, mask=left_mask_split)
        right_out = cv2.bitwise_and(base_img, base_img, mask=right_mask_split)
        # Draw brain outlines
        if brain_contours is not None:
            if brain_contours.get("whole") is not None:
                cv2.drawContours(whole_out, [brain_contours["whole"]], -1, (255, 255, 255), DRAW_THICKNESS)
            if brain_contours.get("left") is not None:
                cv2.drawContours(left_out, [brain_contours["left"]], -1, (255, 255, 255), DRAW_THICKNESS)
            if brain_contours.get("right") is not None:
                cv2.drawContours(right_out, [brain_contours["right"]], -1, (255, 255, 255), DRAW_THICKNESS)
        # Draw infarct contours
        for cnt in draw_contours:
            cv2.drawContours(whole_out, [cnt], -1, (0, 255, 0), DRAW_THICKNESS)
            cv2.drawContours(left_out, [cnt], -1, (0, 255, 0), DRAW_THICKNESS)
            cv2.drawContours(right_out, [cnt], -1, (0, 255, 0), DRAW_THICKNESS)
        # Save images with channel identifier
        save_image(whole_out, os.path.join(output_dir, f"{base_name_infarct}_{channel_name}_{method}_whole.tif"))
        save_image(left_out, os.path.join(output_dir, f"{base_name_infarct}_{channel_name}_{method}_left.tif"))
        save_image(right_out, os.path.join(output_dir, f"{base_name_infarct}_{channel_name}_{method}_right.tif"))

    return areas


def qt_process_images(
    workspace: AnalysisWorkspace,
    ref_image_path: str,
    method: str,
    infarct_contour_detection: str,
    section_contour_threshold: int,
    cd68_color: str,
    exclude_color: str,
    cd68_start_val: int,
    exclude_start_val: int,
    fixed_roi_data: Optional[dict] = None,
    background_percent: float = 10.0,
) -> Optional[dict]:
    """Process a single reference image using the Qt workspace for interaction."""
    import re

    def _apply_flips(img: np.ndarray, flip_flags) -> np.ndarray:
        if isinstance(flip_flags, tuple):
            flip_h, flip_v = bool(flip_flags[0]), bool(flip_flags[1])
        elif isinstance(flip_flags, int):
            if flip_flags == 1:
                flip_h, flip_v = False, True
            elif flip_flags == 2:
                flip_h, flip_v = True, False
            elif flip_flags == 3:
                flip_h, flip_v = True, True
            else:
                flip_h, flip_v = False, False
        else:
            flip_h, flip_v = False, False
        if flip_h and flip_v:
            return cv2.flip(img, -1)
        if flip_h:
            return cv2.flip(img, 1)
        if flip_v:
            return cv2.flip(img, 0)
        return img

    ref_img = cv2.imread(ref_image_path)
    if ref_img is None:
        print("Error loading image.")
        return None
    display_info = os.path.basename(ref_image_path)

    result = workspace.run_midline(ref_img, display_info)
    if result[0] == "EXIT" or result[0] is None:
        return "EXIT"
    split_points, mod_ref_img, flip_flags = result

    folder = os.path.dirname(ref_image_path)
    base = os.path.basename(ref_image_path)
    m = re.search(r"(.*)_reference_(\d+)(\.(?:tif|tiff|png|jpg|jpeg))$", base, re.IGNORECASE)
    infarct_path = None
    if m:
        animal_prefix, sec_num, ext = m.groups()
        candidate = f"{animal_prefix}_infarct_{sec_num}{ext}"
        candidate_path = os.path.join(folder, candidate)
        if os.path.exists(candidate_path):
            infarct_path = candidate_path

    if infarct_path is None:
        infarct_img = mod_ref_img.copy()
    else:
        inf_raw = cv2.imread(infarct_path)
        if inf_raw is None:
            infarct_img = mod_ref_img.copy()
        else:
            infarct_img = _apply_flips(inf_raw, flip_flags)

    brain_contours_result = workspace.run_brain_threshold(
        mod_ref_img, split_points, section_contour_threshold, display_info
    )
    if brain_contours_result[0] == "EXIT" or brain_contours_result[0] is None:
        return "EXIT"
    brain_contours, brain_thresh_mask, final_brain_positions = brain_contours_result
    left_mask, right_mask = split_image(mod_ref_img, split_points, brain_thresh_mask)

    seg_result = workspace.run_infarct_threshold(
        infarct_img,
        mod_ref_img,
        brain_contours,
        split_points,
        infarct_contour_detection,
        display_info,
        cd68_color,
        exclude_color,
        cd68_start_val,
        exclude_start_val,
        fixed_roi_data=fixed_roi_data,
    )
    if seg_result == "EXIT" or seg_result is None:
        return "EXIT"

    final_infarct_positions = seg_result["final_positions"]
    selected_infarct_contour = seg_result.get("selected_infarct_contour", None)
    if isinstance(selected_infarct_contour, np.ndarray):
        selected_infarct_contour = [selected_infarct_contour]

    COLOR_MAP = {"blue": 0, "green": 1, "red": 2}
    if seg_result.get("no_infarct", False):
        infarct_mask = np.zeros(infarct_img.shape[:2], dtype=np.uint8)
        infarct_contours: list = []
    else:
        cd68_idx = COLOR_MAP.get(cd68_color, 1)
        cd68_channel = infarct_img[:, :, cd68_idx]
        _, mask_cd68 = cv2.threshold(cd68_channel, final_infarct_positions["Infarct Lower V"], 255, cv2.THRESH_BINARY)
        if exclude_color == "none":
            mask_exclude = np.ones_like(mask_cd68, dtype=np.uint8) * 255
        else:
            exclude_idx = COLOR_MAP[exclude_color] if exclude_color != "none" else None
            exclude_channel = mod_ref_img[:, :, exclude_idx]
            _, mask_exclude = cv2.threshold(
                exclude_channel,
                final_infarct_positions["exclude threshold"],
                255,
                cv2.THRESH_BINARY_INV,
            )
        infarct_mask = cv2.bitwise_and(mask_cd68, mask_exclude)
        infarct_contours, _ = cv2.findContours(infarct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if infarct_contour_detection == "max" and selected_infarct_contour is not None:
            infarct_contours = selected_infarct_contour if isinstance(selected_infarct_contour, list) else [selected_infarct_contour]

    areas = calculate_areas(
        left_mask=left_mask,
        right_mask=right_mask,
        infarct_mask=infarct_mask,
        brain_contour=brain_contours.get("whole"),
        infarct_contours=infarct_contours,
        method=method,
        infarct_contour_detection=infarct_contour_detection,
        brain_contours_dict=brain_contours,
    )

    infarct_area_positive: float = 0.0
    infarct_intensity_avg: float = 0.0
    infarct_area_positive_intensity_avg: float = 0.0
    fixed_roi_used = bool(seg_result.get("FIXED_ROI", False))
    roi_mask = seg_result.get("roi_mask", None)
    if fixed_roi_used and roi_mask is not None:
        roi_bool = (roi_mask > 0)
        areas["infarct_area"] = float(np.sum(roi_bool))
        positive_mask_bool = np.logical_and((infarct_mask > 0), roi_bool)
        infarct_area_positive = float(np.sum(positive_mask_bool))
        all_vals = infarct_img[:, :, COLOR_MAP.get(cd68_color, 1)][roi_bool]
        infarct_intensity_avg = float(np.mean(all_vals)) if all_vals.size else 0.0
        if np.any(positive_mask_bool):
            infarct_area_positive_intensity_avg = float(np.mean(infarct_img[:, :, COLOR_MAP.get(cd68_color, 1)][positive_mask_bool]))
        else:
            infarct_area_positive_intensity_avg = 0.0
    else:
        infarct_area_positive = float(np.sum(infarct_mask > 0))
        infarct_intensity_avg = float(np.mean(infarct_img[:, :, COLOR_MAP.get(cd68_color, 1)][infarct_mask > 0])) if np.any(infarct_mask > 0) else 0.0
        infarct_area_positive_intensity_avg = infarct_intensity_avg

    # Background intensity and normalized metrics
    background_intensity = ""
    infarct_intensity_avg_normalized = ""
    infarct_intensity_positive_avg_normalized = ""
    try:
        # Determine background region: lowest percentile within brain mask
        brain_mask = brain_thresh_mask > 0
        if np.any(brain_mask):
            vals = mod_ref_img[:, :, COLOR_MAP.get(cd68_color, 1)][brain_mask]
            cutoff = np.percentile(vals, background_percent)
            bg_vals = vals[vals <= cutoff]
            background_intensity = float(np.mean(bg_vals)) if bg_vals.size else 0.0
            if background_intensity != 0:
                infarct_intensity_avg_normalized = infarct_intensity_avg / background_intensity
                infarct_intensity_positive_avg_normalized = infarct_area_positive_intensity_avg / background_intensity
    except Exception:
        background_intensity = ""
        infarct_intensity_avg_normalized = ""
        infarct_intensity_positive_avg_normalized = ""

    areas.update({
        "infarct_area_positive": infarct_area_positive,
        "infarct_area_intensity_avg": infarct_intensity_avg,
        "infarct_area_positive_intensity_avg": infarct_area_positive_intensity_avg,
        "background_intensity": background_intensity,
        "infarct_intensity_avg_normalized": infarct_intensity_avg_normalized,
        "infarct_intensity_positive_avg_normalized": infarct_intensity_positive_avg_normalized,
        "brain_outline_threshold": final_brain_positions.get("Brain Lower V") if final_brain_positions else "",
        "CD68_threshold": final_infarct_positions.get("Infarct Lower V") if final_infarct_positions else "",
        "exclude_threshold": final_infarct_positions.get("exclude threshold") if final_infarct_positions else "",
        "FIXED_ROI": fixed_roi_used,
    })

    # Save segmented views (same as patched_process_images)
    output_dir = create_output_directory(os.path.dirname(ref_image_path), base_name="results")
    raw_base = os.path.splitext(os.path.basename(ref_image_path))[0]
    base_name_infarct = re.sub(r"_(?:reference|infarct)_", "_", raw_base)
    channel_views = {
        "infarct": infarct_img,
        "reference": mod_ref_img,
    }
    for channel_name, base_img in channel_views.items():
        whole_out = base_img.copy()
        left_mask_split = split_image(base_img, split_points, np.ones(base_img.shape[:2], dtype=np.uint8))[0]
        right_mask_split = split_image(base_img, split_points, np.ones(base_img.shape[:2], dtype=np.uint8))[1]
        left_out = cv2.bitwise_and(base_img, base_img, mask=left_mask_split)
        right_out = cv2.bitwise_and(base_img, base_img, mask=right_mask_split)
        if brain_contours is not None:
            if brain_contours.get("whole") is not None:
                cv2.drawContours(whole_out, [brain_contours["whole"]], -1, (255, 255, 255), DRAW_THICKNESS)
            if brain_contours.get("left") is not None:
                cv2.drawContours(left_out, [brain_contours["left"]], -1, (255, 255, 255), DRAW_THICKNESS)
            if brain_contours.get("right") is not None:
                cv2.drawContours(right_out, [brain_contours["right"]], -1, (255, 255, 255), DRAW_THICKNESS)
        if seg_result.get("no_infarct", False):
            draw_contours = []
        elif isinstance(infarct_contours, list):
            draw_contours = infarct_contours
        else:
            draw_contours = [infarct_contours]
        for cnt in draw_contours:
            cv2.drawContours(whole_out, [cnt], -1, (0, 255, 0), DRAW_THICKNESS)
            cv2.drawContours(left_out, [cnt], -1, (0, 255, 0), DRAW_THICKNESS)
            cv2.drawContours(right_out, [cnt], -1, (0, 255, 0), DRAW_THICKNESS)
        save_image(whole_out, os.path.join(output_dir, f"{base_name_infarct}_{channel_name}_{method}_whole.tif"))
        save_image(left_out, os.path.join(output_dir, f"{base_name_infarct}_{channel_name}_{method}_left.tif"))
        save_image(right_out, os.path.join(output_dir, f"{base_name_infarct}_{channel_name}_{method}_right.tif"))

    return areas