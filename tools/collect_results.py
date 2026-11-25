"""Turn the archives of the finished runs into the tables kept in ``results/``.

    python tools/collect_results.py <folder with the archives>

The archives are the files attached to this repository's release, one per run:
``<dataset>-<experiment>.tar.gz``, with ``<dataset>`` either ``as-described``
or ``2023-data``. Each becomes ``results/<dataset>/<experiment>/``. The
centralised experiment has two archives per dataset: ``exp1`` is the run made
from start to finish, and ``exp1-interrupted`` the earlier run that Colab
stopped and that was resumed. Each folder has:

- ``metrics.csv``            the global model's validation score after every round, and the final test score
- ``settings.json``          what was run, on what, with which versions (those of the session the run ended in)
- ``per_class.csv``          the final test, class by class
- ``confusion_matrix.csv``   the final test's confusion matrix, as counts: rows are the predicted class and
                             columns the true class, in the order of ``per_class.csv``; the last row counts
                             the signs that were not detected, the last column the detections where there is
                             no sign
- ``clients.csv``            the clients' training, epoch by epoch (federated runs; for a run that was
                             moved between machines, only the rounds run on the last one)
- ``epochs.csv``             the training, epoch by epoch (the centralised run)

and ``results/summary.csv`` lists every run's final test score beside the one
the 2023 notebook of the same experiment printed. The tables in the README's
Results section and in CORRECTION.md are rewritten from it. Logs and models
stay in the archives.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCHIVE = re.compile(r"(as-described|2023-data)-(exp\d)(-interrupted)?")
#: The final test scores the 2023 notebooks printed (see original/README.md): P, R, mAP@.5, mAP@.5:.95.
RUNS_2023 = {
    "exp1": (0.495, 0.779, 0.646, 0.430),
    "exp2": (0.320, 0.273, 0.0653, 0.0369),
    "exp3": (0.428, 0.399, 0.216, 0.135),
    "exp4": (0.479, 0.695, 0.584, 0.370),
    "exp5": (0.493, 0.424, 0.343, 0.195),
}
# YOLOv7's results.txt: epoch, GPU memory, three training losses and their sum,
# labels, image size, P, R, mAP@.5, mAP@.5:.95, three validation losses.
EPOCH_FIELDS = [
    "epoch", "train_box", "train_obj", "train_cls", "train_total",
    "precision", "recall", "map50", "map50_95", "val_box", "val_obj", "val_cls",
]  # fmt: skip
CLASS_ROW = re.compile(r"^\s*(\S.*?)\s+(\d+)\s+(\d+)\s+([\d.e-]+)\s+([\d.e-]+)\s+([\d.e-]+)\s+([\d.e-]+)\s*$")


def epoch_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 15:
            rows.append([str(int(f[0].split("/")[0]) + 1), *f[2:6], *f[8:15]])
    return rows


def per_class(log: str) -> list[list[str]]:
    """The class-by-class table test.py printed: class, images, labels, P, R, mAP@.5, mAP@.5:.95."""
    rows = []
    for line in log.replace("\r", "\n").splitlines():
        m = CLASS_ROW.match(line)
        if m and "it/s" not in line:
            rows.append(list(m.groups()))
    # The table is printed once; keep the last block, which starts at "all".
    starts = [i for i, r in enumerate(rows) if r[0] == "all"]
    if not starts:
        raise SystemExit("eval_test.log: the table of test results is missing")
    return rows[starts[-1] :]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


#: How often Colab stopped the first centralised run on each dataset, which then went on with
#: ``train.py --resume``. An archive holds the log of the last resumption only; the counts are
#: from the record kept while the runs were made (the events ``colab/supervise.py`` logged).
RESUMED = {"as-described": 2, "2023-data": 4}


def confusion_rows(text: str, labels: list[int]) -> list[list[str]]:
    """YOLOv7's confusion matrix with every cell as [predicted class, true class].

    YOLOv7 at the pinned commit (``utils/metrics.py``) counts a sign of class t
    that is detected as class p at [t, p], a sign that is not detected at
    [last row, t], and a detection that matches no sign at [p, last column].
    Its own figure labels the rows "Predicted" and the columns "True", which is
    right for the last row and column only. Here the class-by-class block is
    turned so that the labels are right for every cell. ``labels`` are the
    numbers of signs per class, which every column must then add up to.
    """
    m = [line.split(",") for line in text.strip().splitlines()]
    n = len(m) - 1
    if n != len(labels) or any(len(row) != n + 1 for row in m):
        raise SystemExit(f"confusion matrix: expected {len(labels) + 1} rows and columns")
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j], m[j][i] = m[j][i], m[i][j]
    if [sum(int(row[c]) for row in m) for c in range(n)] != labels:
        raise SystemExit("confusion matrix: the columns do not add up to the numbers of signs per class")
    return m


def collect(archive: Path) -> dict:
    """Unpack one archive into results/."""
    name = archive.name[: -len(".tar.gz")]
    match = ARCHIVE.fullmatch(name)
    if not match:
        raise SystemExit(f"{archive.name}: not one of the release's run archives")
    dataset, exp, interrupted = match.groups()
    shown = name[len(dataset) + 1 :]
    out = RESULTS / dataset / shown
    with tarfile.open(archive) as tar:
        members = {m.name[len(name) + 1 :]: m for m in tar.getmembers() if m.isfile()}
        resumed = any(rel.startswith("train.resume") for rel in members)
        if exp == "exp1" and not interrupted and ("train.log" not in members or resumed):
            raise SystemExit(f"{name}: not a run from start to finish (train.log missing, or it was resumed)")

        def read(rel: str) -> str:
            return tar.extractfile(members[rel]).read().decode("utf-8", errors="replace")

        out.mkdir(parents=True, exist_ok=True)
        metrics = read("metrics.csv").replace("\r\n", "\n")
        (out / "metrics.csv").write_text(metrics, encoding="utf-8", newline="\n")
        settings = json.loads(read("settings.json"))
        settings.pop("data", None)  # a path on the machine that ran it
        (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        classes = per_class(read("eval_test.log"))
        write_csv(out / "per_class.csv", ["class", "images", "labels", "precision", "recall", "map50", "map50_95"],
                  classes)  # fmt: skip
        labels = [int(row[2]) for row in classes[1:]]  # the first row is "all"
        matrix = confusion_rows(read("eval_test/confusion_matrix.csv"), labels)
        (out / "confusion_matrix.csv").write_text("".join(",".join(row) + "\n" for row in matrix), encoding="utf-8")

        rows = list(csv.DictReader(io.StringIO(metrics)))
        test = next(r for r in rows if r["split"] == "test")
        rounds = [int(r["round"]) for r in rows if r["split"] == "val"]
        e = settings["experiment"]
        if e["clients"]:
            if rounds != list(range(1, e["rounds"] + 1)):
                raise SystemExit(f"{name}: metrics.csv has rounds {rounds}, expected 1 to {e['rounds']}")
            client_rows = []
            for rel in sorted(members):
                m = re.fullmatch(r"round_(\d+)/client_(\d+)/results\.txt", rel)
                if m:
                    client_rows += [[int(m[1]), int(m[2]), *r] for r in epoch_rows(read(rel))]
            # A run that had to be moved to another machine keeps its global
            # scores for every round, but the clients' own records only for the
            # rounds run on the machine it finished on.
            covered = sorted({r[0] for r in client_rows})
            per_round = e["clients"] * e["epochs"]
            if len(client_rows) != len(covered) * per_round or covered != list(range(covered[0], e["rounds"] + 1)):
                raise SystemExit(f"{name}: client records are incomplete within rounds {covered[0]} to {covered[-1]}")
            write_csv(out / "clients.csv", ["round", "client", *EPOCH_FIELDS], client_rows)
            client_rounds = f"{covered[0]}-{covered[-1]}"
        else:
            epochs = epoch_rows(read("train/results.txt"))
            if len(epochs) != e["epochs"]:
                raise SystemExit(f"{name}: {len(epochs)} epochs recorded, expected {e['epochs']}")
            write_csv(out / "epochs.csv", EPOCH_FIELDS, epochs)
            client_rounds = ""
    print(
        f"{name} -> {out.relative_to(ROOT)}" + (f" (client records: rounds {client_rounds})" if client_rounds else "")
    )
    if interrupted:
        note = f"interrupted and resumed {RESUMED[dataset]} times"
    elif exp == "exp1":
        note = "run from start to finish without interruption"
    else:
        note = ""
    return {
        "dataset": dataset, "experiment": shown, "clients": e["clients"] or "",
        "rounds": e["rounds"] if e["clients"] else "", "epochs": e["epochs"],
        **{k: test[k] for k in ("precision", "recall", "map50", "map50_95")},
        **dict(zip(("precision_2023", "recall_2023", "map50_2023", "map50_95_2023"), RUNS_2023[exp], strict=True)),
        "client_records_rounds": client_rounds,
        "note": note,
    }  # fmt: skip


SETTINGS = {
    "exp1": "Centralised, 50 epochs",
    "exp2": "16 clients, 17 rounds, 3 local epochs",
    "exp3": "4 clients, 17 rounds, 3 local epochs",
    "exp4": "4 clients, 17 rounds, 10 local epochs",
    "exp5": "4 clients, 40 rounds, 3 local epochs",
}


SETTINGS_ZH = {
    "exp1": "集中式，50 个 epoch",
    "exp2": "16 个客户端，17 轮，每轮 3 个本地 epoch",
    "exp3": "4 个客户端，17 轮，每轮 3 个本地 epoch",
    "exp4": "4 个客户端，17 轮，每轮 10 个本地 epoch",
    "exp5": "4 个客户端，40 轮，每轮 3 个本地 epoch",
}
HEADERS = {
    False: (
        "| Experiment | Setting | 2023 runs | Averaging corrected, 2023 dataset "
        "| Averaging corrected, dataset as described |",
        "| Experiment | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | mAP@0.5 relative to Experiment 1 |",
    ),
    True: (
        "| 实验 | 设置 | 2023 年的结果 | 平均已修正，2023 年数据集 | 平均已修正，按论文所述构建的数据集 |",
        "| 实验 | 精确率 | 召回率 | mAP@0.5 | mAP@0.5:0.95 | mAP@0.5 与实验 1 之比 |",
    ),
}


def readme_tables(summary: list[dict], zh: bool = False) -> str:
    """The two tables of the README's Results section, from the rows of summary.csv."""
    score = {(r["dataset"], r["experiment"]): r for r in summary}

    def cell(dataset: str, exp: str, key: str = "map50") -> str:
        r = score.get((dataset, exp))
        return f"{float(r[key]):.3f}" if r else "not finished"

    first, second = HEADERS[zh]
    lines = [first, "| --- | --- | --: | --: | --: |"]
    for exp, setting in (SETTINGS_ZH if zh else SETTINGS).items():
        lines.append(f"| {exp[3:]} | {setting} | {RUNS_2023[exp][2]:.3f} | {cell('2023-data', exp)} "
                     f"| {cell('as-described', exp)} |")  # fmt: skip
    lines += ["", second, "| --- | --: | --: | --: | --: | --: |"]
    base = score.get(("as-described", "exp1"))
    for exp in SETTINGS:
        r = score.get(("as-described", exp))
        if r:
            share = f"{100 * float(r['map50']) / float(base['map50']):.0f}%" if base else ""
            lines.append(f"| {exp[3:]} | {cell('as-described', exp, 'precision')} "
                         f"| {cell('as-described', exp, 'recall')} | {cell('as-described', exp)} "
                         f"| {cell('as-described', exp, 'map50_95')} | {share} |")  # fmt: skip
    return "\n".join(lines)


def correction_tables(summary: list[dict]) -> dict[str, str]:
    """Tables A and B of CORRECTION.md, from the rows of summary.csv."""
    score = {(r["dataset"], r["experiment"]): r for r in summary}
    a = [
        "| Experiment | Setting | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
        "| --- | --- | --: | --: | --: | --: |",
    ]
    b = ["| Experiment | 2023 runs | Averaging corrected, 2023 dataset | Averaging and dataset corrected |",
         "| --- | --: | --: | --: |"]  # fmt: skip
    for exp, setting in SETTINGS.items():
        r, old = score[("as-described", exp)], score[("2023-data", exp)]
        a.append(f"| {exp[3:]} | {setting} | {float(r['precision']):.3f} | {float(r['recall']):.3f} "
                 f"| {float(r['map50']):.3f} | {float(r['map50_95']):.3f} |")  # fmt: skip
        b.append(f"| {exp[3:]} | {RUNS_2023[exp][2]:.3f} | {float(old['map50']):.3f} | {float(r['map50']):.3f} |")
    return {"correction-table-a": "\n".join(a), "correction-table-b": "\n".join(b)}


def fill(path: Path, blocks: dict[str, str]) -> None:
    """Replace what stands between each ``<!-- name:start -->`` and ``<!-- name:end -->`` in a file."""
    text = path.read_text(encoding="utf-8")
    for name, body in blocks.items():
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{body}\n{end}{tail}"
    path.write_text(text, encoding="utf-8")


def update_readme(summary: list[dict]) -> None:
    fill(ROOT / "README.md", {"results-table": readme_tables(summary)})
    fill(ROOT / "README.zh-CN.md", {"results-table": readme_tables(summary, zh=True)})
    fill(ROOT / "CORRECTION.md", correction_tables(summary))


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs" / "release")
    archives = sorted(folder.glob("*-exp*.tar.gz"))
    if not archives:
        raise SystemExit(f"{folder}: no run archives (<dataset>-<experiment>.tar.gz) here")
    summary = [collect(a) for a in archives]
    summary.sort(key=lambda r: (r["dataset"] != "as-described", r["experiment"]))
    write_csv(RESULTS / "summary.csv", list(summary[0]), [list(r.values()) for r in summary])
    print(f"{len(summary)} experiments -> results/summary.csv")
    update_readme(summary)
    print("README.md, README.zh-CN.md and CORRECTION.md: tables updated")


if __name__ == "__main__":
    main()
