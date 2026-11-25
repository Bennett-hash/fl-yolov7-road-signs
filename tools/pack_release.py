"""Pack the archives of the runs for the release.

    python tools/pack_release.py <folder with the archives as made on Colab> <folder for the release>

``colab/archive.py`` makes one archive per run on the machine that ran it,
named after the experiment and the dataset the run was started with
(``exp4-gtsrb.tar.gz``, ``exp1-2023.run2.tar.gz``). This gives each the name
used in ``results/`` (``as-described-exp4.tar.gz``, ``2023-data-exp1.tar.gz``),
leaves out the pictures of test images that YOLOv7 saves with a test, because
they show GTSRB images, and writes the SHA-256 of every archive to
``SHA256SUMS``. Nothing else in an archive is changed.
"""

from __future__ import annotations

import hashlib
import io
import re
import sys
import tarfile
from pathlib import Path

DATASETS = {"gtsrb": "as-described", "2023": "2023-data"}
PICTURES = re.compile(r"(.*/)?test_batch\d+_(labels|pred)\.jpg")


def new_name(old: str) -> str:
    """``exp1-gtsrb.run2`` is the centralised run made from start to finish; ``exp1-gtsrb`` the interrupted one."""
    exp, rest = old.split("-", 1)
    dataset, _, label = rest.partition(".")
    interrupted = exp == "exp1" and label != "run2"
    return f"{DATASETS[dataset]}-{exp}" + ("-interrupted" if interrupted else "")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    source, release = Path(sys.argv[1]), Path(sys.argv[2])
    release.mkdir(parents=True, exist_ok=True)
    sums = []
    for src in sorted(source.glob("exp*.tar.gz")):
        old = src.name[: -len(".tar.gz")]
        new = new_name(old)
        out = release / f"{new}.tar.gz"
        left_out = 0
        with tarfile.open(src) as tin, tarfile.open(out, "w:gz") as tout:
            for member in tin.getmembers():
                rel = member.name[len(old) + 1 :]
                if PICTURES.fullmatch(rel):
                    left_out += 1
                    continue
                data = tin.extractfile(member).read() if member.isfile() else None
                member.name = f"{new}/{rel}" if rel else new
                tout.addfile(member, io.BytesIO(data) if data is not None else None)
        sums.append(f"{hashlib.sha256(out.read_bytes()).hexdigest()}  {out.name}")
        print(f"{src.name} -> {out.name} ({left_out} pictures of test images left out)")
    (release / "SHA256SUMS").write_text(
        "\n".join(sorted(sums, key=lambda line: line.split()[1])) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
