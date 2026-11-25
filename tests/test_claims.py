"""The numbers the README and the correction note state in words are the ones the recorded results give."""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
N_CLASSES = 43


def rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def text(name: str) -> str:
    return re.sub(r"\s+", " ", (ROOT / name).read_text(encoding="utf-8"))


def validation(dataset: str, exp: str) -> dict[int, str]:
    return {int(r["round"]): f"{float(r['map50']):.3f}" for r in rows(RESULTS / dataset / exp / "metrics.csv")
            if r["split"] == "val"}  # fmt: skip


def confusion(dataset: str, exp: str) -> list[list[int]]:
    with open(RESULTS / dataset / exp / "confusion_matrix.csv", encoding="utf-8") as f:
        return [[int(x) for x in line] for line in csv.reader(f)]


def test_confusion_matrix_columns_are_the_true_classes():
    """Every column, a true class, adds up to that class's number of test signs."""
    for row in rows(RESULTS / "summary.csv"):
        folder = (row["dataset"], row["experiment"])
        matrix = confusion(*folder)
        assert len(matrix) == N_CLASSES + 1 and all(len(line) == N_CLASSES + 1 for line in matrix), folder
        labels = [int(r["labels"]) for r in rows(RESULTS.joinpath(*folder) / "per_class.csv")[1:]]
        assert [sum(line[c] for line in matrix) for c in range(N_CLASSES)] == labels, folder
        assert sum(labels) == 3921


def test_share_of_signs_not_detected():
    def missed(exp: str) -> tuple[int, int]:
        matrix = confusion("as-described", exp)
        totals = [sum(line[c] for line in matrix) for c in range(N_CLASSES)]
        lost = matrix[N_CLASSES][:N_CLASSES]
        return round(100 * sum(lost) / sum(totals)), sum(x > t / 2 for t, x in zip(totals, lost, strict=True))

    assert missed("exp1") == (51, 24)
    assert missed("exp4")[0] == 38
    assert "the centralised model misses 51% of the test signs, and the best federated model 38%" in text(
        "CORRECTION.md"
    )
    assert not any(confusion("as-described", "exp2")[r][c] for r in range(N_CLASSES) for c in range(N_CLASSES + 1))


def test_scores_at_equal_training():
    described = {e: validation("as-described", e) for e in ("exp3", "exp4", "exp5")}
    earlier = {e: validation("2023-data", e) for e in ("exp3", "exp4", "exp5")}
    # 120 epochs per client: round 12 of ten local epochs, round 40 of three. 50: round 5, and round 17 (51).
    assert (described["exp4"][12], described["exp5"][40]) == ("0.352", "0.335")
    assert (described["exp4"][5], described["exp5"][17]) == ("0.223", "0.224")
    assert (earlier["exp4"][12], earlier["exp5"][40]) == ("0.470", "0.374")
    assert (described["exp3"][17], described["exp5"][17]) == ("0.177", "0.224")
    assert (earlier["exp3"][17], earlier["exp5"][17]) == ("0.236", "0.258")
    readme, correction = text("README.md"), text("CORRECTION.md")
    assert "give validation scores of 0.352 and 0.335; after 50 epochs, 0.223 and 0.224" in readme
    assert "(0.470 against 0.374 after 120 epochs)" in readme
    assert "0.177 in Experiment 3 and 0.224 in Experiment 5 on the dataset as described, and 0.236 and 0.258" in readme
    assert "give validation scores of 0.352 and 0.335" in correction


def test_the_epoch_a_client_returns():
    """How often the local epoch with the best fitness, which YOLOv7 keeps as best.pt, was the first."""
    counts = {}
    for exp in ("exp2", "exp3", "exp4", "exp5"):
        runs = defaultdict(list)
        for r in rows(RESULTS / "as-described" / exp / "clients.csv"):
            fitness = 0.1 * float(r["map50"]) + 0.9 * float(r["map50_95"])  # YOLOv7's fitness()
            runs[r["round"], r["client"]].append((fitness, -int(r["epoch"])))
        best = Counter(-max(epochs)[1] for epochs in runs.values())
        counts[exp] = (best[1], len(runs))
    assert counts == {"exp4": (30, 56), "exp5": (74, 116), "exp3": (21, 68), "exp2": (52, 272)}
    sentence = ("30 of 56 client runs of Experiment 4, 74 of 116 of Experiment 5, 21 of 68 of Experiment 3 and "
                "52 of 272 of Experiment 2")  # fmt: skip
    assert sentence in text("README.md") and sentence in text("CORRECTION.md")


def test_the_centralised_run_was_still_improving():
    epochs = {
        int(r["epoch"]): f"{float(r['map50']):.3f}" for r in rows(RESULTS / "as-described" / "exp1" / "epochs.csv")
    }
    assert (epochs[40], epochs[50]) == ("0.334", "0.365")
    assert "(0.334 at epoch 40, 0.365 at epoch 50)" in text("README.md")


def test_the_split_puts_every_track_in_all_three_sets():
    tracks = defaultdict(Counter)
    shares = []
    for r in rows(ROOT / "src" / "fedyolo" / "gtsrb_2023.csv"):
        k = int(r["name"].rsplit("_", 1)[1])
        assert r["split"] == {4: "val", 6: "test"}.get(k % 10, "train")
        folder, name = r["source"].split("/")
        tracks[folder, name.split("_")[0]][r["split"]] += 1
        shares += [float(x) for x in r["label_2023"].split()[3:]]
    assert len(tracks) == 1307
    assert all(t["val"] == 3 and t["test"] == 3 for t in tracks.values())
    assert Counter(sum(t.values()) for t in tracks.values()) == {30: 1306, 29: 1}
    assert 0.74 < sum(shares) / len(shares) < 0.76  # "about three quarters of the width and height"
