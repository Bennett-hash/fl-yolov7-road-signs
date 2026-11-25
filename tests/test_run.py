"""The runner's bookkeeping, with YOLOv7's train.py and test.py replaced by stand-ins."""

import argparse
import csv
import json
import sys
import threading
from pathlib import Path

import pytest
import torch

from fedyolo import run, yolo
from fedyolo.experiments import Experiment


class Interrupted(BaseException):
    """What a stopped session looks like to the code that was running."""


class FakeYOLO:
    """Stand-ins for train.py and test.py. Every test.py call gives a different score."""

    def __init__(self):
        self.trained: list[tuple[str, str]] = []
        self.tested: list[tuple[str, str]] = []
        self.stop_testing_in: str | None = None  # a round folder's name
        self.lock = threading.Lock()

    def train(self, yolov7, data_yaml, weights, epochs, project, name, device, log, workers=None):
        with self.lock:
            self.trained.append((project.name, name))
        (project / name / "weights").mkdir(parents=True, exist_ok=True)
        torch.save({"client": name}, project / name / "weights" / "best.pt")
        log.write_text("training\n")
        return None, 1.0

    def test(self, yolov7, weights, data_yaml, task, project, name, device, log):
        assert Path(weights).exists() and Path(weights).suffix == ".pt"
        if project.name == self.stop_testing_in:
            raise Interrupted(f"stopped while scoring {project.name}")
        with self.lock:
            self.tested.append((project.name, name))
            k = len(self.tested)
        return yolo.Scores(10, 10, k / 100, k / 100, k / 100, k / 100)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "images" / "train").mkdir(parents=True)
    for k in range(1, 11):
        (data / "images" / "train" / f"00000_{k}.png").write_bytes(b"")
    yolov7 = tmp_path / "yolov7"
    yolov7.mkdir()
    (yolov7 / "yolov7-tiny.pt").write_bytes(b"weights")
    f = FakeYOLO()
    monkeypatch.setattr(yolo, "train", f.train)
    monkeypatch.setattr(yolo, "test", f.test)
    monkeypatch.setattr(yolo, "prime_label_caches", lambda *a: None)
    monkeypatch.setattr(run, "average_checkpoints", lambda paths: {"averaged": [str(p) for p in paths]})
    # What run.main changes for the process, undone after each test.
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setattr(sys, "path", list(sys.path))
    f.data, f.yolov7, f.out = data, yolov7, tmp_path / "out"
    f.args = argparse.Namespace(
        data=data, yolov7=yolov7, seed=0, parallel=2, workers=None, device="cpu", averaging="fixed"
    )
    return f


def rows(out: Path) -> list[dict]:
    with open(out / "metrics.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def val_rounds(out: Path) -> list[str]:
    return [r["round"] for r in rows(out) if r["split"] == "val"]


EXP = Experiment("sim", 2, 3, 1)


def test_a_session_stopped_while_scoring_a_round_scores_it_on_resuming(fake):
    fake.out.mkdir()
    fake.stop_testing_in = "round_02"
    with pytest.raises(Interrupted):
        run.run_federated(EXP, fake.out, fake.args)
    assert not (fake.out / "round_02" / "global.pt").exists()  # not counted as finished
    fake.stop_testing_in = None
    run.run_federated(EXP, fake.out, fake.args)
    assert val_rounds(fake.out) == ["1", "2", "3"]
    assert [t[0] for t in fake.tested] == ["round_01", "round_02", "round_03"]
    assert sorted(fake.trained) == [(f"round_{r:02d}", f"client_{i:02d}") for r in (1, 2, 3) for i in (1, 2)]
    assert all((fake.out / f"round_{r:02d}" / "global.pt").exists() for r in (1, 2, 3))
    assert not list(fake.out.glob("round_*/global.pt.pt"))


def test_a_session_stopped_after_writing_the_row_does_not_write_it_twice(fake, monkeypatch):
    fake.out.mkdir()
    record = run.record_scores

    def record_then_stop(path, row):
        record(path, row)
        if row["round"] == 2:
            raise Interrupted("stopped before global.pt was renamed")

    monkeypatch.setattr(run, "record_scores", record_then_stop)
    with pytest.raises(Interrupted):
        run.run_federated(EXP, fake.out, fake.args)
    assert val_rounds(fake.out) == ["1", "2"] and not (fake.out / "round_02" / "global.pt").exists()
    monkeypatch.setattr(run, "record_scores", record)
    run.run_federated(EXP, fake.out, fake.args)
    assert val_rounds(fake.out) == ["1", "2", "3"]
    assert [t[0] for t in fake.tested] == ["round_01", "round_02", "round_03"]  # round 2 scored once
    assert len(fake.trained) == 6


def test_a_row_for_a_model_that_was_made_again_is_replaced(fake, monkeypatch):
    """As when a run is carried on elsewhere from a metrics.csv copied later than its newest global.pt."""
    fake.out.mkdir()
    record = run.record_scores

    def record_then_stop(path, row):
        record(path, row)
        if row["round"] == 2:
            raise Interrupted

    monkeypatch.setattr(run, "record_scores", record_then_stop)
    with pytest.raises(Interrupted):
        run.run_federated(EXP, fake.out, fake.args)
    monkeypatch.setattr(run, "record_scores", record)
    old = next(r for r in rows(fake.out) if r["round"] == "2")
    for done in (fake.out / "round_02").glob("*.DONE"):
        done.unlink()  # the clients of round 2 have to train again
    run.run_federated(EXP, fake.out, fake.args)
    assert val_rounds(fake.out) == ["1", "2", "3"]
    new = next(r for r in rows(fake.out) if r["round"] == "2")
    assert new["map50"] != old["map50"]  # the score of the model made again


def test_client_list_caches_go_only_when_a_list_changes(fake):
    fake.out.mkdir()
    exp = Experiment("sim", 2, 0, 1)  # no rounds: only the client lists are written
    run.run_federated(exp, fake.out, fake.args)
    caches = [fake.out / "clients" / f"client_{i:02d}.cache" for i in (1, 2)]
    for c in caches:
        c.write_bytes(b"labels")
    run.run_federated(exp, fake.out, fake.args)
    assert all(c.exists() for c in caches)
    fake.args.seed = 1
    run.run_federated(exp, fake.out, fake.args)
    assert not any(c.exists() for c in caches)


def main_args(fake, *extra):
    return [
        "sim", "--yolov7", str(fake.yolov7), "--data", str(fake.data), "--out", str(fake.out),
        "--clients", "2", "--rounds", "2", "--epochs", "1", "--device", "cpu", *extra,
    ]  # fmt: skip


def test_a_run_is_resumed_only_with_the_settings_it_was_started_with(fake, capsys):
    assert run.main(main_args(fake)) == 0
    folder = fake.out / "sim"
    for extra, field in ((["--seed", "1"], "seed"), (["--epochs", "2"], "epochs"), (["--rounds", "3"], "rounds")):
        with pytest.raises(SystemExit) as stop:
            run.main(main_args(fake, *extra))
        assert stop.value.code == 2
        assert field in capsys.readouterr().err
    (fake.data / "DATASET.txt").write_text("another dataset\n")
    with pytest.raises(SystemExit):
        run.main(main_args(fake))
    assert "dataset" in capsys.readouterr().err
    (fake.data / "DATASET.txt").unlink()
    assert run.main(main_args(fake, "--parallel", "1")) == 0  # what does not change the results may differ
    settings = json.loads((folder / "settings.json").read_text())
    assert settings["parallel"] == 1 and settings["numpy"] and settings["yolov7_commit"] is None
    assert len(settings["initial_weights_sha256"]) == 64
    sessions = [json.loads(line) for line in (folder / "sessions.jsonl").read_text().splitlines()]
    assert [s["rounds_done"] for s in sessions] == [0, 2]
    assert [r["split"] for r in rows(folder)] == ["val", "val", "test"]


def test_changed_settings_names_what_differs():
    before = {"experiment": {"name": "exp4", "clients": 4, "rounds": 17, "epochs": 10}, "averaging": "fixed",
              "seed": 0, "data": "/a", "dataset": "digest 1"}  # fmt: skip
    assert run.changed_settings(before, before | {"data": "/b", "parallel": 4, "workers": 2}) == []
    assert run.changed_settings(before, before | {"seed": 1, "averaging": "original"}) == [
        "averaging: 'fixed', now 'original'",
        "seed: 0, now 1",
    ]
    assert run.changed_settings(before, before | {"dataset": None}) == ["dataset: 'digest 1', now '/a'"]
    # Without a DATASET.txt, the folder is all that says which data it is.
    assert run.changed_settings(before | {"dataset": None}, before | {"dataset": None, "data": "/b"}) == [
        "dataset: '/a', now '/b'"
    ]


def test_parallel_must_be_at_least_1(fake):
    with pytest.raises(SystemExit):
        run.main(main_args(fake, "--parallel", "0"))


@pytest.fixture
def centralised(fake, monkeypatch):
    fake.resumed = []

    def resume(yolov7, last, log):
        fake.resumed.append(log.name)
        log.write_text("                 all          10          10       0.5       0.5       0.5       0.5\n")
        return yolo.Scores(10, 10, 0.5, 0.5, 0.5, 0.5), 2.0

    monkeypatch.setattr(yolo, "resume", resume)
    fake.out.mkdir()
    (fake.out / "train" / "weights").mkdir(parents=True)
    return fake


def test_resumed_centralised_runs_number_their_logs(centralised):
    exp = Experiment("c", 0, 1, 50)
    last = centralised.out / "train" / "weights" / "last.pt"
    for n in (1, 2):
        torch.save({"epoch": 10 * n}, last)
        run.run_centralised(exp, centralised.out, centralised.args)
        (centralised.out / "train.DONE").unlink()
    assert centralised.resumed == ["train.resume-1.log", "train.resume-2.log"]
    assert [r["round"] for r in rows(centralised.out)] == ["50"]  # the second replaced the first


def test_a_finished_centralised_run_without_train_done_is_recorded_not_resumed(centralised):
    exp = Experiment("c", 0, 1, 50)
    torch.save({"epoch": -1, "optimizer": None}, centralised.out / "train" / "weights" / "last.pt")
    (centralised.out / "train.log").write_text(
        "                 all          10          10       0.1       0.2       0.3       0.4\n"
        "      speed limit 20          10           1       0.5       0.5       0.5       0.5\n"
    )
    run.run_centralised(exp, centralised.out, centralised.args)
    assert centralised.resumed == []
    assert (centralised.out / "train.DONE").exists()
    (row,) = rows(centralised.out)
    assert (row["round"], row["map50"], row["map50_95"], row["train_seconds"]) == ("50", "0.3", "0.4", "")
