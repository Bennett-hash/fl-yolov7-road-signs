"""Put saved experiment state onto a (new) Colab session, so the runs can carry on.

    python colab/restore.py --to SESSION runs/colab/fla/exp4=exp4-2023 runs/colab/fla/exp5=exp5-2023

Each argument is a folder that ``supervise.py`` copied from a session, then
``=`` and the folder name it should have under ``/content/out`` on the new
session. This uploads the folder's ``metrics.csv``, ``settings.json`` and
``sessions.jsonl``, and either its newest ``round_*/global.pt`` (a federated
experiment) or its ``train/`` checkpoint with ``opt.yaml`` and ``hyp.yaml``
(the centralised one). Starting the experiments again with ``launch.py``, with
the settings they were started with, then resumes them.

A centralised run must keep its folder name, because YOLOv7's ``opt.yaml``
records its paths.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def colab(*args: str, timeout: float = 600) -> None:
    proc = subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise SystemExit(f"colab {' '.join(args[:3])} failed: {(proc.stdout + proc.stderr).strip()[-400:]}")


def files_to_restore(local: Path) -> list[Path]:
    """The files that let an experiment carry on, relative to its folder."""
    wanted = [p for p in (Path("metrics.csv"), Path("settings.json"), Path("sessions.jsonl")) if (local / p).exists()]
    rounds = sorted(local.glob("round_*/global.pt"), key=lambda p: int(re.findall(r"\d+", p.parent.name)[0]))
    if rounds:
        wanted.append(rounds[-1].relative_to(local))
    for p in ("train/opt.yaml", "train/hyp.yaml", "train/weights/last.pt", "train/weights/best.pt"):
        if (local / p).exists():
            wanted.append(Path(p))
    return wanted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--to", required=True, help="the colab session to restore onto")
    ap.add_argument("items", nargs="+", help="LOCAL_FOLDER=REMOTE_FOLDER")
    args = ap.parse_args()
    key = os.environ.get("COLAB_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
    for item in args.items:
        local_s, _, remote = item.partition("=")
        local = Path(local_s)
        remote = remote or local.name
        files = files_to_restore(local)
        if (local / "train/weights/last.pt").exists() and remote != local.name:
            raise SystemExit(f"{local}: a centralised run has to keep its folder name ({local.name})")
        dirs = sorted({f"/content/out/{remote}/{f.parent}" for f in files})
        colab("ssh", "-s", args.to, "-i", key, "--", "mkdir -p " + " ".join(dirs), timeout=120)
        for f in files:
            colab("upload", "-s", args.to, str(local / f), f"/content/out/{remote}/{f}")
        print(f"{local} -> {args.to}:/content/out/{remote}: {', '.join(map(str, files))}", flush=True)


if __name__ == "__main__":
    main()
