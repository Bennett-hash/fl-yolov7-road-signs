"""Prepare a Colab VM for the experiments. Safe to run more than once.

Run it on the VM after uploading ``fedyolo.tar.gz`` (``git archive`` of this
repository) to ``/content``:

    colab exec -s fl --env DATASETS="gtsrb 2023" -f colab/setup_runtime.py

It clones YOLOv7 at the commit the 2023 experiments used, downloads the
yolov7-tiny weights they started from and checks them, installs the packages
YOLOv7 needs that the VM lacks, and installs this package. For each name in
``DATASETS`` it then builds a dataset with ``fedyolo.data``, which downloads
the official GTSRB training archive to ``/content`` if it is not there and
checks it: ``gtsrb`` as the paper describes it, into
``/content/GTSRB_YOLO_gtsrb``, and ``2023`` as the 2023 runs had it, into
``/content/GTSRB_YOLO_2023``. A dataset already built is left as it is.
(Without ``DATASETS`` it unpacks an uploaded ``/content/GTSRB_YOLO.zip``
instead.) Last, it prints the image counts and the versions in use.
"""

import hashlib
import importlib.util
import os
import subprocess
import sys
import zipfile

CONTENT = "/content"
YOLOV7 = f"{CONTENT}/yolov7"
YOLOV7_COMMIT = "84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca"
TINY_URL = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
TINY_SHA256 = "0fa1423606133c06362ec716029dade1169d5631e83195b75d8efa66d2ee0b77"
DATA_ZIP = f"{CONTENT}/GTSRB_YOLO.zip"
CODE = f"{CONTENT}/fedyolo.tar.gz"


def sh(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if not os.path.isdir(YOLOV7):
    sh("git", "clone", "-q", "https://github.com/WongKinYiu/yolov7.git", YOLOV7)
sh("git", "-C", YOLOV7, "checkout", "-q", YOLOV7_COMMIT)

tiny = f"{YOLOV7}/yolov7-tiny.pt"
if not os.path.exists(tiny) or sha256(tiny) != TINY_SHA256:
    sh("curl", "-sSL", "-o", tiny, TINY_URL)
assert sha256(tiny) == TINY_SHA256, "yolov7-tiny.pt does not match the 2023 download"

# Packages YOLOv7 imports. Colab has most of them; its requirements.txt pins
# NumPy below 1.24, which no longer installs on Colab's Python, so it is not used.
missing = [m for m in ("cv2", "yaml", "scipy", "pandas", "seaborn", "tqdm", "matplotlib", "tensorboard")
           if importlib.util.find_spec(m) is None]  # fmt: skip
names = {"cv2": "opencv-python-headless", "yaml": "pyyaml"}
if missing:
    sh(sys.executable, "-m", "pip", "install", "-q", *[names.get(m, m) for m in missing])

if os.path.exists(CODE):
    sh(sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", CODE)

# DATASETS names the datasets to build from the official GTSRB download, each
# into /content/GTSRB_YOLO_<name>: "gtsrb" as the paper describes it, "2023" as
# the 2023 runs had it. Without it, the uploaded GTSRB_YOLO.zip is unpacked.
datasets = os.environ.get("DATASETS", "").split()
for name in datasets:
    target = f"{CONTENT}/GTSRB_YOLO_{name}"
    if not os.path.isdir(target):
        as_2023 = ["--as-2023"] if name == "2023" else []
        sh(sys.executable, "-m", "fedyolo.data", "--out", target,
           "--archive", f"{CONTENT}/GTSRB_Final_Training_Images.zip", *as_2023)  # fmt: skip
if not datasets and not os.path.isdir(f"{CONTENT}/GTSRB_YOLO"):
    with zipfile.ZipFile(DATA_ZIP) as z:
        z.extractall(CONTENT)
for folder in sorted(d for d in os.listdir(CONTENT) if d.startswith("GTSRB_YOLO") and not d.endswith(".zip")):
    if os.path.exists(f"{CONTENT}/{folder}/DATASET.txt"):
        print(folder, open(f"{CONTENT}/{folder}/DATASET.txt").read().strip())
    counts = [len(os.listdir(f"{CONTENT}/{folder}/images/{split}")) for split in ("train", "val", "test")]
    print(f"{folder}: {counts[0]} / {counts[1]} / {counts[2]} images", flush=True)

import numpy  # noqa: E402
import torch  # noqa: E402

import fedyolo  # noqa: E402

print(f"python {sys.version.split()[0]}, torch {torch.__version__}, numpy {numpy.__version__}")
print(f"fedyolo {fedyolo.__version__}")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("CPUs:", os.cpu_count())
sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv")
