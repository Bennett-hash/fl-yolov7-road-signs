"""Run one of the paper's experiments, picking up where an interrupted run stopped.

    python -m fedyolo.run exp4 --yolov7 /content/yolov7 --data /content/GTSRB_YOLO --out /content/out

For a federated experiment, every round trains each client from the current
global model (in turn, or at the same time with ``--parallel``) with YOLOv7's
``train.py``, averages the clients' ``best.pt`` into a new global model, and
scores it on the validation set. After the last round the global model is
scored on the test set, as in the paper. ``train.py`` and ``test.py`` get the
arguments the notebooks gave them, with ``--project``, ``--name``,
``--exist-ok`` and ``--device`` for where to write their output and what to
run on (the federated notebooks also gave ``--project``), and ``train.py``
also ``--workers`` if it is given.

Progress is kept on disk. A client is finished when its ``DONE`` file exists,
and a round when its ``global.pt`` does, which is written only after the
round's validation score is in ``metrics.csv``. Running the same command again
after an interruption repeats only what was unfinished: a client's training,
or the averaging and scoring of a round. It refuses to carry on a run that was
started with other settings (see :func:`changed_settings`). Each start adds a
line to ``sessions.jsonl`` in the experiment's folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import __version__, yolo
from .averaging import average_checkpoints, load_checkpoint
from .experiments import CLASS_NAMES, EXPERIMENTS, INITIAL_WEIGHTS, Experiment
from .partition import split_among_clients

METRIC_FIELDS = [
    "experiment", "round", "split", "images", "labels",
    "precision", "recall", "map50", "map50_95", "train_seconds", "wall_seconds", "finished_at",
]  # fmt: skip


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_yaml(path: Path, train: str, data: Path) -> None:
    names = ", ".join(repr(n) for n in CLASS_NAMES)
    path.write_text(
        f"train: {train}\n"
        f"val: {data / 'images' / 'val'}/\n"
        f"test: {data / 'images' / 'test'}/\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n",
        encoding="utf-8",
    )


def append_metric(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def metric_row(
    exp: str, rnd: int | str, split: str, s: yolo.Scores, seconds: float | None, wall: float | None = None
) -> dict:
    """One row of ``metrics.csv``.

    For a federated round, ``train_seconds`` adds up every client's training
    time and ``wall_seconds`` is the time the round took in the session that
    finished it. For a centralised run, ``train_seconds`` is the time
    ``train.py`` took in the last session only, the one that finished the
    training, so it is less than the whole training if that was resumed.
    Either is empty when it is not known.
    """
    return {
        "experiment": exp,
        "round": rnd,
        "split": split,
        **asdict(s),
        "train_seconds": "" if seconds is None else round(seconds, 1),
        "wall_seconds": "" if wall is None else round(wall, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def recorded_scores(path: Path, rnd: int | str, split: str) -> yolo.Scores | None:
    """The scores ``metrics.csv`` already holds for this round and split, if any.

    A session can stop after it has written a row and before it has marked the
    step as finished; the next one then uses the row instead of writing it again.
    """
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["round"] == str(rnd) and row["split"] == split:
                return yolo.Scores(
                    int(row["images"]), int(row["labels"]),
                    *(float(row[k]) for k in ("precision", "recall", "map50", "map50_95")),
                )  # fmt: skip
    return None


def record_scores(path: Path, row: dict) -> None:
    """Append a row to ``metrics.csv``, replacing any row of the same round and split.

    Such a row scores a model that has since been made again, as when a run
    is carried on from a ``metrics.csv`` copied later than its weights.
    """
    if recorded_scores(path, row["round"], row["split"]) is not None:
        log(f"replacing the {row['split']} row of round {row['round']} in {path.name}: its model was made again")
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if (r["round"], r["split"]) != (str(row["round"]), row["split"])]
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)
    append_metric(path, row)


def saved_epoch(checkpoint: Path) -> int | None:
    """The epoch a ``train.py`` checkpoint was saved after, or None if it cannot be read.

    It is -1 once ``train.py`` has finished: its last step strips the
    optimizer from ``last.pt`` and ``best.pt`` and sets their epoch to -1.
    """
    try:
        return int(load_checkpoint(checkpoint)["epoch"])
    except Exception:
        return None


def run_centralised(exp: Experiment, out: Path, args: argparse.Namespace) -> Path:
    yaml_path = out / "data.yaml"
    write_yaml(yaml_path, f"{args.data / 'images' / 'train'}/", args.data)
    run_dir = out / "train"
    best = run_dir / "weights" / "best.pt"
    if (out / "train.DONE").exists():
        log("training already finished")
        return best
    last = run_dir / "weights" / "last.pt"
    metrics = out / "metrics.csv"
    if last.exists() and saved_epoch(last) == -1:
        # train.py finished, but the session stopped before it wrote
        # train.DONE. train.py --resume would refuse this checkpoint ("nothing
        # to resume"). The final validation scores are in metrics.csv already,
        # or else in the log of the session that finished the training: the
        # newest log that has them.
        log("training had finished before the last session stopped")
        if recorded_scores(metrics, exp.epochs, "val") is None:
            logs = sorted(out.glob("train*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            found = (yolo.parse_scores(p.read_text(encoding="utf-8", errors="replace")) for p in logs)
            scores = next((s for s in found if s), None)
            if scores:
                append_metric(metrics, metric_row(exp.name, exp.epochs, "val", scores, None))
        (out / "train.DONE").write_text("\n")
        return best
    if last.exists():
        # Numbered, so that the folder shows how often the run was resumed.
        n = 1
        while (out / f"train.resume-{n}.log").exists():
            n += 1
        log(f"resuming centralised training from {last}")
        scores, seconds = yolo.resume(args.yolov7, last, out / f"train.resume-{n}.log")
    else:
        log(f"centralised training, {exp.epochs} epochs")
        scores, seconds = yolo.train(
            args.yolov7, yaml_path, args.yolov7 / INITIAL_WEIGHTS, exp.epochs,
            out, "train", args.device, out / "train.log", workers=args.workers,
        )  # fmt: skip
    if scores:
        record_scores(metrics, metric_row(exp.name, exp.epochs, "val", scores, seconds))
    (out / "train.DONE").write_text(f"{seconds:.1f}\n")
    log(f"centralised training done in {seconds / 3600:.2f} h")
    return best


def finished_rounds(exp: Experiment, out: Path) -> int:
    """The newest round of a federated run whose ``global.pt`` exists, or 0."""
    return max((r for r in range(1, exp.rounds + 1) if (out / f"round_{r:02d}" / "global.pt").exists()), default=0)


def run_federated(exp: Experiment, out: Path, args: argparse.Namespace) -> Path:
    train_dir = args.data / "images" / "train"
    images = sorted(str(train_dir / f) for f in os.listdir(train_dir) if f.endswith(".png"))
    parts = split_among_clients(images, exp.clients, seed=args.seed)
    clients_dir = out / "clients"
    clients_dir.mkdir(exist_ok=True)
    yamls = []
    for i, part in enumerate(parts, 1):
        lst = clients_dir / f"client_{i:02d}.txt"
        text = "\n".join(part) + "\n"
        if not lst.exists() or lst.read_text(encoding="utf-8") != text:
            lst.write_text(text, encoding="utf-8")
            # YOLOv7 caches the labels of a list in client_XX.cache beside it
            # and, at the pinned commit, never checks the cache against the
            # list. A cache of an earlier list would train on its images.
            lst.with_suffix(".cache").unlink(missing_ok=True)
        y = clients_dir / f"client_{i:02d}.yaml"
        write_yaml(y, str(lst), args.data)
        yamls.append(y)
    log(
        f"{exp.clients} clients with {len(parts[0])} images each "
        f"({len(images) - len(parts[0]) * exp.clients} left out), seed {args.seed}"
    )

    val_yaml = out / "data.yaml"
    write_yaml(val_yaml, f"{train_dir}/", args.data)

    # YOLOv7 writes a label cache beside each folder of labels the first time it
    # reads it. Write the validation and test caches now, so that clients
    # trained at the same time do not all try to write the same file.
    yolo.prime_label_caches(args.yolov7, [args.data / "images" / "val", args.data / "images" / "test"])

    # Carry on after the newest finished round. Only its global.pt is needed,
    # so a run can be moved to another machine with that one file.
    global_w = args.yolov7 / INITIAL_WEIGHTS
    finished = finished_rounds(exp, out)
    if finished:
        global_w = out / f"round_{finished:02d}" / "global.pt"
        log(f"resuming after round {finished}")
    metrics = out / "metrics.csv"
    for r in range(finished + 1, exp.rounds + 1):
        rdir = out / f"round_{r:02d}"
        g_path = rdir / "global.pt"
        rdir.mkdir(exist_ok=True)
        round_start = time.monotonic()
        # Whether every client of this round had finished in an earlier session.
        # Only then can a validation row already written for the round be the
        # score of the model averaged now.
        trained_before = all((rdir / f"client_{i:02d}.DONE").exists() for i in range(1, exp.clients + 1))

        def train_client(i: int, y: Path, r: int = r, rdir: Path = rdir, start_w: Path = global_w) -> float:
            name = f"client_{i:02d}"
            done = rdir / f"{name}.DONE"
            if done.exists():
                return float(done.read_text().split()[0])
            if (rdir / name).exists():
                shutil.rmtree(rdir / name)  # an interrupted run: start that client again
            scores, seconds = yolo.train(
                args.yolov7, y, start_w, exp.epochs, rdir, name, args.device, rdir / f"{name}.log",
                workers=args.workers,
            )  # fmt: skip
            if not (rdir / name / "weights" / "best.pt").exists():
                raise RuntimeError(f"{rdir / name} has no weights/best.pt")
            done.write_text(f"{seconds:.1f}\n")
            val = f", val mAP@.5 {scores.map50:.3f}" if scores else ""
            log(f"round {r}/{exp.rounds}, client {i}/{exp.clients}: {seconds / 60:.1f} min{val}")
            return seconds

        # Clients in a round are independent: each starts from the same global
        # model. Training several at once changes nothing but the wall time.
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(train_client, i, y) for i, y in enumerate(yamls, 1)]
            round_seconds = sum(f.result() for f in futures)
        round_wall = time.monotonic() - round_start

        client_weights = [rdir / f"client_{i:02d}" / "weights" / "best.pt" for i in range(1, exp.clients + 1)]
        if args.averaging == "fixed":
            averaged = average_checkpoints(client_weights)
        else:
            from .original import average_weights

            averaged = average_weights([str(p) for p in client_weights])
        # The round counts as finished once global.pt exists, so the model is
        # scored and its row written first, and only then renamed. test.py
        # loads the weights by path, whatever their name. torch.save names the
        # archive inside the file after the file, less its last suffix, so
        # with this name global.pt has the same bytes as one saved directly.
        tmp = rdir / "global.pt.pt"
        torch.save(averaged, tmp)
        scores = recorded_scores(metrics, r, "val") if trained_before else None
        if scores is None:
            scores = yolo.test(args.yolov7, tmp, val_yaml, "val", rdir, "eval_val", args.device, rdir / "eval_val.log")
            record_scores(metrics, metric_row(exp.name, r, "val", scores, round_seconds, round_wall))
        tmp.replace(g_path)
        global_w = g_path
        log(f"round {r}/{exp.rounds} averaged: val mAP@.5 {scores.map50:.3f}, mAP@.5:.95 {scores.map50_95:.3f}")

        # The clients' weights are no longer needed once they are averaged.
        for p in client_weights:
            for w in p.parent.glob("*.pt"):
                w.unlink()
    return global_w


def run_identity(settings: dict) -> dict:
    """The settings that decide what a run computes: the experiment, the averaging, the client split, the data."""
    e = settings.get("experiment") or {}
    return {
        "experiment": e.get("name"),
        "clients": e.get("clients"),
        "rounds": e.get("rounds"),
        "epochs": e.get("epochs"),
        "averaging": settings.get("averaging"),
        "seed": settings.get("seed"),
        # fedyolo.data's DATASET.txt names the dataset and gives its content
        # digest. Without one, only the folder's path says which data it is.
        "dataset": settings.get("dataset") or settings.get("data"),
    }


def changed_settings(before: dict, now: dict) -> list[str]:
    """The settings that differ between an earlier session's ``settings.json`` and this one, as "name: old, now new"."""
    a, b = run_identity(before), run_identity(now)
    return [f"{k}: {a[k]!r}, now {b[k]!r}" for k in a if a[k] != b[k]]


def git_commit(repo: Path) -> str | None:
    """The commit a git clone is at, or None if it is not a clone or git cannot say."""
    if not (repo / ".git").exists():
        return None
    try:
        got = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    if got.returncode != 0:
        return None
    return got.stdout.strip() or None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def progress(exp: Experiment, out: Path) -> dict:
    """How far the run had got when this session started: rounds, or epochs, finished."""
    if not exp.centralised:
        return {"rounds_done": finished_rounds(exp, out)}
    last = out / "train" / "weights" / "last.pt"
    if (out / "train.DONE").exists():
        return {"epochs_done": exp.epochs}
    if not last.exists():
        return {"epochs_done": 0}
    epoch = saved_epoch(last)
    return {"epochs_done": None if epoch is None else exp.epochs if epoch == -1 else epoch + 1}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedyolo.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment", help="exp1 to exp5, or a name of your own with --clients, --rounds and --epochs")
    ap.add_argument("--yolov7", type=Path, required=True, help="a clone of WongKinYiu/yolov7 holding yolov7-tiny.pt")
    ap.add_argument("--data", type=Path, required=True, help="the dataset in YOLO layout (images/ and labels/)")
    ap.add_argument("--out", type=Path, required=True, help="where each experiment gets a folder of its own")
    ap.add_argument("--averaging", choices=["fixed", "original"], default="fixed",
                    help="'original' re-runs the 2023 function, which does not average")  # fmt: skip
    ap.add_argument("--seed", type=int, default=0, help="seed for dividing the images among clients")
    ap.add_argument("--device", default="0", help="passed to YOLOv7: a CUDA index, or 'cpu'")
    ap.add_argument("--parallel", type=int, default=1, help="clients of a round to train at the same time")
    ap.add_argument("--workers", type=int, help="data-loading processes per train.py (YOLOv7's default is 8)")
    ap.add_argument("--clients", type=int, help="override the number of clients (0 for centralised)")
    ap.add_argument("--rounds", type=int, help="override the number of rounds")
    ap.add_argument("--epochs", type=int, help="override the epochs per client per round")
    ap.add_argument("--tag", help="added to the output folder's name, e.g. the dataset: exp4-gtsrb")
    args = ap.parse_args(argv)
    args.yolov7, args.data, args.out = args.yolov7.resolve(), args.data.resolve(), args.out.resolve()
    if args.parallel < 1:
        ap.error("--parallel must be at least 1")

    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(args.yolov7))  # torch.load needs YOLOv7's model classes

    base = EXPERIMENTS.get(args.experiment, Experiment(args.experiment, 0, 1, 1))
    exp = Experiment(
        args.experiment,
        base.clients if args.clients is None else args.clients,
        base.rounds if args.rounds is None else args.rounds,
        base.epochs if args.epochs is None else args.epochs,
        base.note,
    )
    if exp.name not in EXPERIMENTS and None in (args.clients, args.rounds, args.epochs):
        ap.error(f"{exp.name} is not one of {', '.join(EXPERIMENTS)}; give --clients, --rounds and --epochs")
    if not (args.yolov7 / INITIAL_WEIGHTS).exists():
        ap.error(f"{args.yolov7 / INITIAL_WEIGHTS} not found")

    folder = exp.name if args.averaging == "fixed" else f"{exp.name}-original-averaging"
    if args.tag:
        folder = f"{folder}-{args.tag}"
    out = args.out / folder
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "experiment": asdict(exp),
        "averaging": args.averaging,
        "seed": args.seed,
        "parallel": args.parallel,
        "workers": args.workers,
        "fedyolo": __version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version.split()[0],
        "yolov7_commit": git_commit(args.yolov7),
        "initial_weights_sha256": sha256(args.yolov7 / INITIAL_WEIGHTS),
        "data": str(args.data),
        # Written by fedyolo.data; says which version of the dataset this is.
        "dataset": (args.data / "DATASET.txt").read_text().strip() if (args.data / "DATASET.txt").exists() else None,
    }
    # A run is carried on only with what it was started with. settings.json
    # itself describes the latest session, so it is rewritten every time.
    if (out / "settings.json").exists():
        changed = changed_settings(json.loads((out / "settings.json").read_text()), settings)
        if changed:
            ap.error(
                f"{out} holds a run started with other settings ({'; '.join(changed)}). "
                "Start it again with the same settings, or give another --out or --tag."
            )
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    before = progress(exp, out)
    session = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "gpu": settings["cuda"],
        "torch": torch.__version__,
        "numpy": np.__version__,
        **before,
    }
    with open(out / "sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(session) + "\n")
    log(f"{exp.name}: {settings}")

    start = time.monotonic()
    final = run_centralised(exp, out, args) if exp.centralised else run_federated(exp, out, args)

    if not (out / "test.DONE").exists():
        # A test row already written can be used only if the final model is the
        # one that was there when this session started.
        made_before = before == ({"epochs_done": exp.epochs} if exp.centralised else {"rounds_done": exp.rounds})
        scores = recorded_scores(out / "metrics.csv", "final", "test") if made_before else None
        if scores is None:
            scores = yolo.test(args.yolov7, final, out / "data.yaml", "test", out, "eval_test", args.device,
                               out / "eval_test.log")  # fmt: skip
            record_scores(out / "metrics.csv", metric_row(exp.name, "final", "test", scores, None))
        (out / "test.DONE").write_text(json.dumps(asdict(scores)) + "\n")
        log(f"{exp.name} test: P {scores.precision:.3f}, R {scores.recall:.3f}, "
            f"mAP@.5 {scores.map50:.3f}, mAP@.5:.95 {scores.map50_95:.3f}")  # fmt: skip
    log(f"{exp.name} finished ({(time.monotonic() - start) / 3600:.2f} h in this session)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
