"""Tests for data-driven channel validation (analysis.validate_channel_selection).

These cover the safeguards added so the Analyze tab no longer blindly indexes a
colour plane: a non-3-channel image is a hard error, an empty selected plane is a
(non-fatal) warning, and 'none' is a clean bypass.
"""

import numpy as np
import pytest
import tifffile as tiff

from infarquant.analysis import (
    COLOR_TO_BGR_INDEX,
    ChannelValidationError,
    channel_plane,
    validate_channel_selection,
)


def _bgr(b=0, g=0, r=0, shape=(16, 16)):
    """Build a cv2-style BGR image with constant per-channel values."""
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[..., 0] = b
    img[..., 1] = g
    img[..., 2] = r
    return img


def test_red_plane_with_signal_returns_index_no_warning():
    img = _bgr(r=200)  # signal only in the red (index 2) plane
    idx, warning = validate_channel_selection(img, "red", role="CD68 channel")
    assert idx == COLOR_TO_BGR_INDEX["red"] == 2
    assert warning is None


def test_picking_empty_plane_warns_but_returns_index():
    img = _bgr(b=200)  # signal only in blue; user asks for red
    idx, warning = validate_channel_selection(img, "red", role="CD68 channel")
    assert idx == 2  # still returns the requested index (non-destructive)
    assert warning is not None
    assert "blue" in warning.lower()  # tells the user where the signal actually is


def test_non_three_channel_image_is_hard_error():
    gray = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ChannelValidationError):
        validate_channel_selection(gray, "red")


def test_four_channel_image_is_hard_error():
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    with pytest.raises(ChannelValidationError):
        validate_channel_selection(rgba, "green")


def test_none_bypasses_validation():
    idx, warning = validate_channel_selection(None, "none")
    assert idx is None and warning is None


def test_unknown_colour_is_hard_error():
    with pytest.raises(ChannelValidationError):
        validate_channel_selection(_bgr(r=200), "magenta")


def test_channel_plane_rgb_selects_single_plane():
    img = _bgr(b=10, g=20, r=30)
    assert np.all(channel_plane(img, "red") == 30)
    assert np.all(channel_plane(img, "green") == 20)
    assert np.all(channel_plane(img, "blue") == 10)


def test_channel_plane_gray_on_true_grayscale_equals_any_channel():
    # A genuine grayscale image loaded as BGR has R==G==B; 'gray' equals that value
    # and matches picking any single channel.
    img = _bgr(b=128, g=128, r=128)
    plane = channel_plane(img, "grayscale")
    assert plane.ndim == 2
    assert np.all(plane == 128)
    assert np.array_equal(plane, channel_plane(img, "red"))


def test_channel_plane_gray_on_colour_is_luminance():
    img = _bgr(b=10, g=20, r=30)
    plane = channel_plane(img, "grayscale")
    assert plane.ndim == 2
    # Luminance combines all channels, so it lies within the per-channel range.
    assert 10 <= int(plane[0, 0]) <= 30


def test_channel_plane_passes_through_2d_image():
    gray = np.full((16, 16), 77, dtype=np.uint8)
    assert np.array_equal(channel_plane(gray, "grayscale"), gray)


def test_gray_validation_accepts_three_channel_image():
    img = _bgr(b=100, g=100, r=100)
    _, warning = validate_channel_selection(img, "grayscale", role="CD68 channel")
    assert warning is None  # 'gray' never warns about an empty single plane


def test_gray_validation_accepts_two_channel_grayscale_image():
    gray = np.full((16, 16), 50, dtype=np.uint8)
    # 'gray' is valid even for a single-plane (2D) image, unlike r/g/b.
    _, warning = validate_channel_selection(gray, "grayscale")
    assert warning is None


def test_lzw_compressed_roundtrip_validates(tmp_path):
    """An LZW-compressed TIFF (needs imagecodecs) still validates cleanly."""
    img = _bgr(g=180)
    path = tmp_path / "compressed.tif"
    tiff.imwrite(str(path), img, compression="lzw")
    import cv2

    loaded = cv2.imread(str(path))
    idx, warning = validate_channel_selection(loaded, "green", tiff_path=str(path), role="CD68 channel")
    assert idx == 1
    assert warning is None
