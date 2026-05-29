"""PyInstaller entry point for the InfarQuant GUI.

Normal run launches the GUI. Running with ``--selftest [result_path]`` instead
performs a headless integrity check of the (frozen) bundle -- it imports every
runtime dependency and decodes an LZW-compressed TIFF (the exact path that needs
``imagecodecs``), then exits 0 on success / 1 on failure. This lets the SHIPPING
exe verify its own bundle without building a second probe artifact.
"""

import sys


def _selftest(result_path: str | None) -> int:
    """Import all runtime deps + decode a compressed TIFF; report pass/fail."""
    import os
    import tempfile
    import traceback

    try:
        import numpy as np
        import pandas  # noqa: F401
        from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: F401
        import tifffile
        import imagecodecs  # noqa: F401  (lazy backend for compressed TIFFs)

        # Importing the app package transitively pulls cv2, pyautogui, PyQt5, etc.
        from infarquant import analysis, preprocess, ui  # noqa: F401

        arr = (np.arange(40 * 40 * 3, dtype=np.uint8) % 255).reshape(40, 40, 3)
        tif = os.path.join(tempfile.gettempdir(), "infarquant_selftest_img.tif")
        tifffile.imwrite(tif, arr, compression="lzw")
        back = tifffile.imread(tif)  # raises if imagecodecs is not bundled
        os.remove(tif)
        assert back.shape == arr.shape and bool((back == arr).all())
        msg, ok = "SELFTEST OK: all deps import; LZW TIFF decode works", True
    except Exception:
        msg, ok = "SELFTEST FAIL:\n" + traceback.format_exc(), False

    # Windowed builds have no console, so write the verdict somewhere readable.
    if result_path:
        try:
            with open(result_path, "w", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass
    return 0 if ok else 1


def main() -> None:
    if "--selftest" in sys.argv:
        idx = sys.argv.index("--selftest")
        out = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        raise SystemExit(_selftest(out))

    from infarquant.app import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
