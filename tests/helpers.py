import numpy as np
from PyQt5.QtWidgets import QSlider, QSpinBox

from infarquant import ui as app


class WorkspaceHarness(app.AnalysisWorkspace):
    """AnalysisWorkspace with hooks to capture actions/controls and auto-drive steps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_actions = []
        self.last_controls = []
        self._auto_action = None

    def _set_actions(self, actions, controls, *args, **kwargs):
        self.last_actions = actions
        self.last_controls = controls
        super()._set_actions(actions, controls, *args, **kwargs)

    def _wait_for_result(self):
        if self._auto_action:
            self._auto_action()
            self._auto_action = None
        return self._pending_result


def make_test_images():
    """Create a simple infarct/reference pair with two red blobs."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Two red squares in BGR
    img[10:30, 10:30, 2] = 200
    img[60:80, 60:80, 2] = 200
    ref = np.zeros_like(img)
    return img, ref


def make_brain_contours():
    """Return a full-frame contour dict for tests."""
    whole = np.array([[[0, 0]], [[99, 0]], [[99, 99]], [[0, 99]]], dtype=np.int32)
    left = np.array([[[0, 0]], [[49, 0]], [[49, 99]], [[0, 99]]], dtype=np.int32)
    right = np.array([[[50, 0]], [[99, 0]], [[99, 99]], [[50, 99]]], dtype=np.int32)
    return {"whole": whole, "left": left, "right": right}


def get_action(actions, label):
    for action in actions:
        if action["label"] == label:
            return action["callback"]
    raise AssertionError(f"Action '{label}' not found: {[a['label'] for a in actions]}")


def get_action_button(workspace, label):
    button = workspace._action_buttons.get(label)
    if button is None:
        raise AssertionError(f"Action button '{label}' not found: {list(workspace._action_buttons.keys())}")
    return button


def get_control(controls, cls, text=None):
    for ctrl in controls:
        if isinstance(ctrl, cls):
            if text is None:
                return ctrl
            if getattr(ctrl, "text", lambda: "")() == text:
                return ctrl
    raise AssertionError(f"Control {cls} with text={text} not found")
