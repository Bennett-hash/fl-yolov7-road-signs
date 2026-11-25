"""Run one of YOLOv7's scripts under a current NumPy.

    python -m fedyolo.compat train.py --img 64 ...

YOLOv7 (August 2023) calls ``np.trapz`` when it computes mAP, and in a few
places the aliases ``np.int`` and ``np.float``. NumPy has since removed these
names. This puts them back, as the functions they referred to, together with
the aliases ``np.bool`` and ``np.object`` (Python's ``bool`` and ``object``)
where this NumPy lacks them, and then runs the script unchanged, so that the
pinned YOLOv7 commit can be used as it is. The only other addition is that
confusion matrices are also saved as numbers (see
:func:`keep_confusion_matrices`); nothing YOLOv7 computes is altered.
"""

from __future__ import annotations

import os
import runpy
import sys

import numpy as np

_ALIASES = {"int": int, "float": float, "bool": bool, "object": object}


def restore_numpy_names() -> list[str]:
    """Add back the NumPy names YOLOv7 uses that this NumPy lacks. Return the ones added."""
    added = []
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
        added.append("trapz")
    for name, value in _ALIASES.items():
        try:
            getattr(np, name)
        except AttributeError:
            setattr(np, name, value)
            added.append(name)
    return added


def keep_confusion_matrices() -> None:
    """Make YOLOv7 save each confusion matrix as numbers too, beside the picture it draws.

    ``test.py`` only draws the matrix as a PNG. The numbers, in
    ``confusion_matrix.csv``, let the figures be redrawn as vector graphics.
    The file holds YOLOv7's matrix as YOLOv7 computed it (``ConfusionMatrix``
    in ``utils/metrics.py``), with ``nc`` classes and one row and one column
    more. A sign of true class t matched to a detection of class p is counted
    at ``[t, p]``; a sign of class t that no detection matched, at ``[nc, t]``
    (the last row); a detection of class p that matched no sign, at
    ``[p, nc]`` (the last column). So the class-by-class block is transposed
    with respect to the axis labels of YOLOv7's own plot (x "True",
    y "Predicted"), which fit only the last row and column. The results
    collector, ``tools/collect_results.py``, turns the class block round so
    that every cell is [predicted, true].
    """
    # YOLOv7's modules, importable once its folder is on sys.path. They are
    # imported in the order listed, and general has to come first, because
    # metrics and general import each other.
    from utils import general, metrics  # noqa: F401

    draw = metrics.ConfusionMatrix.plot

    def plot(self, save_dir="", names=()):
        np.savetxt(os.path.join(save_dir, "confusion_matrix.csv"), self.matrix, delimiter=",", fmt="%d")
        return draw(self, save_dir=save_dir, names=names)

    metrics.ConfusionMatrix.plot = plot


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m fedyolo.compat SCRIPT [ARGS...]")
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.getcwd())  # YOLOv7 imports its own modules from its folder
    restore_numpy_names()
    keep_confusion_matrices()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
