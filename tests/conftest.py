import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest
from PyQt5.QtWidgets import QApplication

# Force Qt to use offscreen rendering for headless test runs
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure local src package imports resolve without requiring editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# OpenCV is a required test dependency for this project.
try:
    import cv2  # noqa: F401
except Exception as exc:
    raise ImportError(
        "OpenCV (cv2) is required to run this test suite. "
        f"Install dependencies with `pip install -e .`. "
        f"Current interpreter: {sys.executable}. "
        f"Original import error: {exc}. "
        "If multiple Python installs exist, run tests as `python -m pytest -q tests`."
    ) from exc

# Provide a custom tmp_path fixture to avoid permission issues with system temp.
@pytest.fixture
def tmp_path():
    path = Path(tempfile.mkdtemp(prefix="pytest-", dir=os.getcwd()))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
