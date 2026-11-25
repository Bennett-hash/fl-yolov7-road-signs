"""Running YOLOv7's own train.py and test.py, as the notebooks did.

Every training run is a separate ``train.py`` process with the arguments the
notebooks passed, plus where to write its output, the device and, if given,
``--workers``, so each training run is the one the notebooks made, apart from
what this package changes on purpose: the averaging and the client split.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .experiments import BATCH_SIZE, IMG_SIZE

# The summary row YOLOv7 prints after validating:
#   "                 all        3921        3921       0.495       0.779       0.646        0.43"
_ALL_ROW = re.compile(
    r"^\s*all\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Scores:
    images: int
    labels: int
    precision: float
    recall: float
    map50: float
    map50_95: float


def parse_scores(text: str) -> Scores | None:
    """Return the last "all" row YOLOv7 printed, or None if there is none."""
    rows = _ALL_ROW.findall(text)
    if not rows:
        return None
    images, labels, p, r, m50, m = rows[-1]
    return Scores(int(images), int(labels), float(p), float(r), float(m50), float(m))


#: How YOLOv7's scripts are started: through fedyolo.compat, which restores
#: the few NumPy names they use that current NumPy has removed.
PYTHON = [sys.executable, "-m", "fedyolo.compat"]


def yolo_env() -> dict[str, str]:
    env = dict(os.environ)
    # YOLOv7 pickles the model into its checkpoints; PyTorch 2.6 and later
    # refuse to load those unless told to.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["WANDB_MODE"] = "disabled"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_logged(cmd: list[str], cwd: Path, log: Path) -> tuple[str, float]:
    """Run a command, writing everything it prints to ``log``. Return the output and the seconds taken."""
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=yolo_env(), stdout=f, stderr=subprocess.STDOUT, check=False)
    seconds = time.monotonic() - start
    text = log.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-30:])
        raise RuntimeError(f"{cmd[len(PYTHON)]} exited with {proc.returncode}; last lines of {log}:\n{tail}")
    return text, seconds


def train(
    yolov7: Path,
    data_yaml: Path,
    weights: Path,
    epochs: int,
    project: Path,
    name: str,
    device: str,
    log: Path,
    workers: int | None = None,
) -> tuple[Scores | None, float]:
    """``python train.py --img 64 --batch 16 --epochs E --data D --weights W --cache``, as in the notebooks.

    ``workers`` sets the number of data-loading processes, which only matters
    when several runs share a machine.
    """
    cmd = [
        *PYTHON, "train.py",
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH_SIZE),
        "--epochs", str(epochs),
        "--data", str(data_yaml),
        "--weights", str(weights),
        "--cache",
        "--project", str(project),
        "--name", name,
        "--exist-ok",
        "--device", device,
    ]  # fmt: skip
    if workers is not None:
        cmd += ["--workers", str(workers)]
    text, seconds = run_logged(cmd, yolov7, log)
    return parse_scores(text), seconds


_PRIME = (
    "import sys; sys.path.insert(0, '.');"
    "from fedyolo.compat import restore_numpy_names; restore_numpy_names();"
    "from utils.datasets import LoadImagesAndLabels;"
    f"[LoadImagesAndLabels(p, img_size={IMG_SIZE}, batch_size={BATCH_SIZE}) for p in sys.argv[1:]]"
)


def prime_label_caches(yolov7: Path, image_dirs: list[Path]) -> None:
    """Have YOLOv7 read these folders once, which writes their label caches."""
    subprocess.run([sys.executable, "-c", _PRIME, *map(str, image_dirs)], cwd=yolov7, env=yolo_env(), check=True,
                   stdout=subprocess.DEVNULL)  # fmt: skip


def resume(yolov7: Path, last: Path, log: Path) -> tuple[Scores | None, float]:
    """Continue an interrupted ``train.py`` run from its ``last.pt``."""
    text, seconds = run_logged([*PYTHON, "train.py", "--resume", str(last)], yolov7, log)
    return parse_scores(text), seconds


def test(
    yolov7: Path,
    weights: Path,
    data_yaml: Path,
    task: str,
    project: Path,
    name: str,
    device: str,
    log: Path,
) -> Scores:
    """``python test.py --img 64 --weights W --data D --task T``, as in the notebooks."""
    cmd = [
        *PYTHON, "test.py",
        "--img", str(IMG_SIZE),
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--task", task,
        "--project", str(project),
        "--name", name,
        "--exist-ok",
        "--device", device,
    ]  # fmt: skip
    text, _ = run_logged(cmd, yolov7, log)
    scores = parse_scores(text)
    if scores is None:
        raise RuntimeError(f"test.py printed no summary row; see {log}")
    return scores
