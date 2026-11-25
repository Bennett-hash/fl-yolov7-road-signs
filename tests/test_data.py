"""The dataset builder's pieces, and the recorded 2023 split. No download needed."""

import collections
import csv
import zipfile

import numpy as np
import pytest

from fedyolo.data import SPLITS, build, fetch, read_ppm, yolo_box


def test_equation_1_of_the_paper():
    # GTSRB 00000/00000_00000.ppm: 29 x 30 pixels, region of interest (5, 6) to (24, 25).
    assert yolo_box(29, 30, 5, 6, 24, 25) == pytest.approx((0.5, 31 / 60, 19 / 29, 19 / 30))


def test_read_ppm():
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    data = b"P6\n# a comment\n3 2\n255\n" + pixels.tobytes()
    assert np.array_equal(read_ppm(data), pixels)
    with pytest.raises(ValueError):
        read_ppm(b"P3\n3 2\n255\n")


def test_recorded_split_is_the_2023_one():
    rows = list(csv.DictReader(open(SPLITS, encoding="utf-8")))
    assert len(rows) == 39209
    assert collections.Counter(r["split"] for r in rows) == {"train": 31367, "val": 3921, "test": 3921}
    assert len({r["name"] for r in rows}) == len({r["source"] for r in rows}) == 39209
    per_class = collections.defaultdict(collections.Counter)
    for r in rows:
        cls = r["name"].split("_")[0]
        assert r["source"].startswith(cls + "/")  # an image keeps its class
        assert r["label_2023"].split()[0] == str(int(cls))  # and so does its 2023 label
        per_class[cls][r["split"]] += 1
    assert len(per_class) == 43
    for counts in per_class.values():
        total = sum(counts.values())
        assert abs(counts["train"] / total - 0.8) < 0.01


def test_every_tenth_image_of_a_class_is_validation_and_another_tenth_test():
    """The rule the module docstring gives, and what follows from it for GTSRB's tracks."""
    with open(SPLITS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tracks = collections.defaultdict(set)
    for r in rows:
        k = int(r["name"].split("_")[1])  # the image's number within its class, from 1
        assert r["split"] == {4: "val", 6: "test"}.get(k % 10, "train"), r["name"]
        cls, file = r["source"].split("/")
        tracks[cls, file.split("_")[0]].add(r["split"])
    assert len(tracks) == 1307
    assert all(splits == {"train", "val", "test"} for splits in tracks.values())


def test_a_wrong_archive_is_named_with_what_to_do(tmp_path):
    archive = tmp_path / "GTSRB_Final_Training_Images.zip"
    archive.write_bytes(b"not the archive")
    with pytest.raises(SystemExit, match=f"delete {archive} and run again"):
        fetch(archive)


def test_build_removes_label_caches_left_from_an_earlier_build(tmp_path):
    pytest.importorskip("cv2")
    archive = tmp_path / "archive.zip"
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    with zipfile.ZipFile(archive, "w") as z:
        folder = "GTSRB/Final_Training/Images/00000"
        z.writestr(f"{folder}/GT-00000.csv", "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n"
                   "00000_00000.ppm;3;2;0;0;2;1;0\n")  # fmt: skip
        z.writestr(f"{folder}/00000_00000.ppm", b"P6\n3 2\n255\n" + pixels.tobytes())
    splits = tmp_path / "splits.csv"
    splits.write_text("name,split,source,label_2023\n00000_1,train,00000/00000_00000.ppm,0 0.5 0.5 0.5 0.5\n")
    out = tmp_path / "out"
    stale = [out / "labels" / f"{split}.cache" for split in ("train", "val", "test")]
    for cache in stale:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"old labels")
    build(archive, out, as_2023=False, splits=splits)
    assert not any(cache.exists() for cache in stale)
    assert (out / "labels" / "train" / "00000_1.txt").read_text() == f"0 {1 / 3} 0.25 {2 / 3} 0.5\n"
