"""Build the GTSRB dataset in YOLO layout from the official GTSRB download.

    python -m fedyolo.data --out data/gtsrb-yolo            # as the paper describes it
    python -m fedyolo.data --out data/gtsrb-yolo-2023 --as-2023   # as the 2023 runs had it

The paper (Section 2.1) converts each GTSRB training image to PNG and labels
it with its own region of interest, normalised by the image's size
(Equation 1). That is what this does by default.

The dataset the 2023 runs actually used was made differently. Each image was
the sign cut out of the GTSRB image along its region of interest, and each
label was the region of interest of a different image of the same class,
normalised by that other image's size. So every box covers roughly the middle
80% of a picture that is all sign. ``--as-2023`` rebuilds that dataset
exactly, from the official images and the 2023 labels recorded in
``gtsrb_2023.csv`` beside this file, so that the 2023 runs can be repeated.

Both use the same split as 2023: 31,367 training, 3,921 validation and 3,921
test images, 80/10/10 within every class, under the same file names. Within a
class every tenth image is a validation image and another tenth a test image,
so every GTSRB track (the 30 images of one physical sign; one track has 29)
has images in all three sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

GTSRB_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
GTSRB_SHA256 = "d32ac4b5fa9a1cbd1994768413902e8193599d9434cf0a8eb9cfd00a6d3a290c"
#: The 2023 split: one row per image, with its official source file and its 2023 label.
SPLITS = Path(__file__).with_name("gtsrb_2023.csv")
#: Content digests of the two datasets the results in this repository were run on.
#: The 2023 one equals the dataset the notebooks unpacked, image for image and label for label.
EXPECTED_DIGESTS = {
    False: "8664fde4f19038b83dfd27ce0583f2190cc3abe76aacccb52a4f2b487cdb74a2",
    True: "2556dd44960f443e618ce07170b85b36045237c510a67b30ec5d160eb8476378",
}


def yolo_box(width: int, height: int, x1: int, y1: int, x2: int, y2: int) -> tuple[float, float, float, float]:
    """Equation 1 of the paper: the box's centre, width and height as fractions of the image."""
    return (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height


def read_ppm(data: bytes) -> np.ndarray:
    """Decode a binary PPM (P6, 8-bit) into an RGB array."""
    buf = io.BytesIO(data)
    fields: list[bytes] = []
    while len(fields) < 4:
        line = buf.readline()
        if not line:
            raise ValueError("truncated PPM header")
        fields += line.split(b"#")[0].split()
    if fields[0] != b"P6" or int(fields[3]) != 255:
        raise ValueError("not an 8-bit binary PPM")
    w, h = int(fields[1]), int(fields[2])
    pixels = np.frombuffer(buf.read(w * h * 3), dtype=np.uint8)
    return pixels.reshape(h, w, 3)


def fetch(path: Path) -> Path:
    if not path.exists():
        print(f"downloading {GTSRB_URL}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(GTSRB_URL, tmp)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != GTSRB_SHA256:
        raise SystemExit(
            f"{path} is not the GTSRB training archive this was built against (sha256 {digest}); "
            f"delete {path} and run again"
        )
    return path


def build(archive: Path, out: Path, as_2023: bool, splits: Path = SPLITS) -> str:
    """Write ``out/images/<split>/<name>.png`` and ``out/labels/<split>/<name>.txt``. Return a digest of the result."""
    import cv2  # only needed here

    with open(splits, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as z:
        annotations: dict[str, list[str]] = {}
        for name in z.namelist():
            if name.endswith(".csv"):
                folder = name.rsplit("/", 1)[0]
                for r in csv.reader(io.TextIOWrapper(z.open(name), encoding="utf-8"), delimiter=";"):
                    if r and r[0] != "Filename":
                        annotations[f"{folder}/{r[0]}"] = r
        for split in ("train", "val", "test"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        # YOLOv7 keeps the labels it has read in a cache beside each folder of
        # labels (labels/train.cache and so on, also written by
        # fedyolo.yolo.prime_label_caches), and at the pinned commit it never
        # checks a cache against the labels. Caches left from an earlier build
        # into this folder would make it train and test on the old labels.
        for cache in (out / "labels").rglob("*.cache"):
            cache.unlink()
        for row in rows:
            source = f"GTSRB/Final_Training/Images/{row['source']}"
            ann = annotations[source]
            width, height, x1, y1, x2, y2, cls = map(int, ann[1:8])
            image = read_ppm(z.read(source))
            if as_2023:
                image = image[y1:y2, x1:x2]
                label = row["label_2023"]
            else:
                label = " ".join(str(v) for v in (cls, *yolo_box(width, height, x1, y1, x2, y2)))
            if not cv2.imwrite(str(out / "images" / row["split"] / f"{row['name']}.png"), image[..., ::-1]):
                raise OSError(f"could not write {row['name']}.png")
            (out / "labels" / row["split"] / f"{row['name']}.txt").write_text(label + "\n")
            digest.update(row["name"].encode() + image.tobytes() + label.encode())
    kind = "as used in 2023 (--as-2023)" if as_2023 else "as described in the paper"
    (out / "DATASET.txt").write_text(
        f"GTSRB training set in YOLO layout, {kind}; content digest {digest.hexdigest()}\n"
    )
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m fedyolo.data", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="folder to write images/ and labels/ into")
    ap.add_argument("--archive", type=Path, default=Path("data/GTSRB_Final_Training_Images.zip"),
                    help="the official GTSRB training archive; downloaded here if missing")  # fmt: skip
    ap.add_argument("--as-2023", action="store_true", help="rebuild the dataset the 2023 runs used")
    args = ap.parse_args(argv)
    digest = build(fetch(args.archive), args.out, args.as_2023)
    n = sum(1 for _ in (args.out / "labels").rglob("*.txt"))
    print(f"wrote {n} images to {args.out}; content digest {digest}")
    if digest == EXPECTED_DIGESTS[args.as_2023]:
        print("this is the dataset the results in this repository were run on")
    else:
        raise SystemExit("the content differs from the dataset the results were run on")


if __name__ == "__main__":
    main()
