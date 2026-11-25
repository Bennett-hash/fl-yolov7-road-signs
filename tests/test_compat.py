"""fedyolo.compat puts back the NumPy names YOLOv7 uses, as what they referred to."""

import warnings

import numpy as np
import pytest

from fedyolo.compat import restore_numpy_names

NAMES = ["trapz", "int", "float", "bool", "object"]


@pytest.fixture
def numpy_without_the_old_names(monkeypatch):
    """NumPy as a release that has removed every one of the names; put back as it was afterwards."""
    if not hasattr(np, "trapezoid"):
        pytest.skip("this NumPy has no np.trapezoid")
    for name in NAMES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # NumPy warns on looking up np.object
            monkeypatch.setattr(np, name, None, raising=False)
        monkeypatch.delattr(np, name)


def test_restore_numpy_names(numpy_without_the_old_names):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert restore_numpy_names() == NAMES
        assert np.trapz is np.trapezoid
        assert (np.int, np.float, np.bool, np.object) == (int, float, bool, object)
        assert restore_numpy_names() == []  # nothing is replaced once it is there


def test_names_this_numpy_has_are_left_alone(numpy_without_the_old_names, monkeypatch):
    own = object()
    monkeypatch.setattr(np, "bool", own, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert "bool" not in restore_numpy_names()
    assert np.bool is own
