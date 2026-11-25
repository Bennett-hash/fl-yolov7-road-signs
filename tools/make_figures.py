"""Draw the figures in ``results/figures`` from the tables in ``results``.

    python tools/make_figures.py

Every figure is written as PDF and SVG (vector, for print) and PNG (for the
README):

- ``scores``                      the final test mAP@0.5 of every experiment: the 2023 runs beside the re-runs
- ``curves-<dataset>``            the global model's validation mAP@0.5 as training goes on
- ``confusion-<dataset>-expN``    the final test's confusion matrix (the paper's Figures 2 to 6)

Needs Matplotlib and NumPy. The colours remain distinguishable for colour-blind
readers; series also differ by line style, marker or hatching, so the figures
read in black and white. The confusion matrices are drawn at the size the
article prints them, about 15 cm by 10 cm, with labels of 5 points or more at
that size.

In ``curves-<dataset>`` the centralised run's line is the validation
``train.py`` makes after every epoch, and the federated runs' markers are
``test.py`` on the global model after every round. The two differ in one
setting: the IoU threshold of the non-maximum suppression is 0.6 in the first
and 0.65 in the second (YOLOv7's defaults for each).
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)  # its remarks on system fonts are not about the figures
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fedyolo.experiments import CLASS_NAMES, EXPERIMENTS  # noqa: E402

RESULTS = ROOT / "results"
# For print, set FIGURE_DIR, FIGURE_DPI=600 and FIGURE_SVG_TEXT=path (letters as outlines).
FIGURES = Path(os.environ.get("FIGURE_DIR", RESULTS / "figures"))
DPI = int(os.environ.get("FIGURE_DPI", "220"))
DATASETS = {"as-described": "dataset as the paper describes it", "2023-data": "2023 dataset"}
BLUE, ORANGE, GREY, GREEN, PURPLE = "#2f6db3", "#d9792b", "#8c8c8c", "#3a923a", "#7b5aa6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,  # keep text as text in the PDF
    "svg.fonttype": os.environ.get("FIGURE_SVG_TEXT", "none"),
    "svg.hashsalt": "fl-yolov7-road-signs",  # the same element ids every time
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})  # fmt: skip


def rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    # No dates in the files, so that drawing them again changes nothing.
    metadata = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=DPI, metadata=metadata[ext])
    plt.close(fig)
    print(f"{FIGURES.name}/{name}.pdf .svg .png")


def describe(exp: str) -> str:
    e = EXPERIMENTS[exp]
    if e.centralised:
        return f"Centralised\n{e.epochs} epochs"
    return f"{e.clients} clients, {e.rounds} rounds\n{e.epochs} local epochs"


def scores() -> None:
    """Final test mAP@0.5: what the 2023 notebooks printed beside the re-runs."""
    summary = rows(RESULTS / "summary.csv")
    exps = sorted(EXPERIMENTS)
    runs_2023 = {r["experiment"]: float(r["map50_2023"]) for r in summary}
    series = [
        ("2023 runs", {e: runs_2023[e] for e in exps if e in runs_2023}, GREY, ""),
        ("Averaging corrected, 2023 dataset",
         {r["experiment"]: float(r["map50"]) for r in summary if r["dataset"] == "2023-data"}, ORANGE, "////"),
        ("Averaging corrected, dataset as described",
         {r["experiment"]: float(r["map50"]) for r in summary if r["dataset"] == "as-described"}, BLUE, ""),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    width = 0.26
    for k, (label, values, colour, hatch) in enumerate(series):
        xs = [i + (k - 1) * width for i, e in enumerate(exps) if e in values]
        ys = [values[e] for e in exps if e in values]
        bars = ax.bar(xs, ys, width * 0.92, label=label, color=colour, hatch=hatch, edgecolor="white", linewidth=0.5)
        for b, y in zip(bars, ys, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.008, f"{y:.3f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=6.5, color="#333333")  # fmt: skip
    ax.set_xticks(range(len(exps)))
    ax.set_xticklabels([f"Experiment {e[3:]}\n{describe(e)}" for e in exps], fontsize=7.5)
    ax.set_ylabel("Test mAP@0.5")
    ax.set_ylim(0, 0.72)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.6,
              handlelength=1.6)  # fmt: skip
    save(fig, "scores")


def curves(dataset: str) -> None:
    """Validation mAP@0.5 of the global model against the epochs each client has trained."""
    base = RESULTS / dataset
    styles = {
        "exp2": (PURPLE, ":", "v"),
        "exp3": (ORANGE, "--", "s"),
        "exp4": (BLUE, "-", "o"),
        "exp5": (GREEN, "-.", "^"),
    }
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    drawn = False
    drawn_final = None
    if (base / "exp1-interrupted" / "epochs.csv").exists():
        e0 = rows(base / "exp1-interrupted" / "epochs.csv")
        (grey,) = ax.plot([int(r["epoch"]) for r in e0], [float(r["map50"]) for r in e0], color="#999999", lw=0.8)
        ax.legend([grey], [f"Centralised run that was interrupted and resumed ({float(e0[-1]['map50']):.3f})"],
                  loc="lower right", frameon=False, fontsize=7, handlelength=1.6)  # fmt: skip
    if (base / "exp1" / "epochs.csv").exists():
        e1 = rows(base / "exp1" / "epochs.csv")
        ax.plot([int(r["epoch"]) for r in e1], [float(r["map50"]) for r in e1], color="#222222", lw=1.3)
        final = float(e1[-1]["map50"])
        ax.axhline(final, color="#222222", lw=0.5, ls=(0, (4, 3)), zorder=0)
        ax.annotate(f"Centralised, 50 epochs ({final:.3f})", (len(e1), final), xytext=(9, 5),
                    textcoords="offset points", fontsize=7.5, va="bottom")  # fmt: skip
        drawn, drawn_final = True, final
    for exp, (colour, ls, marker) in styles.items():
        path = base / exp / "metrics.csv"
        if not path.exists():
            continue
        e = EXPERIMENTS[exp]
        val = [r for r in rows(path) if r["split"] == "val"]
        xs = [int(r["round"]) * e.epochs for r in val]
        ys = [float(r["map50"]) for r in val]
        ax.plot(xs, ys, color=colour, ls=ls, lw=1.2, marker=marker, ms=2.6)
        # A label that would sit on the centralised run's line moves off it, to the curve's own side.
        dy = -2
        if drawn_final is not None and abs(ys[-1] - drawn_final) < 0.04:
            dy = 9 if ys[-1] >= drawn_final else -10
        ax.annotate(f"Exp. {exp[3:]}: {e.clients} clients, {e.epochs} local epochs ({ys[-1]:.3f})",
                    (xs[-1], ys[-1]), xytext=(5, dy), textcoords="offset points", fontsize=7.5,
                    color=colour, va="center")  # fmt: skip
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("Epochs each client has trained (one marker per communication round)")
    ax.set_ylabel("Validation mAP@0.5 of the global model")
    ax.set_xlim(0, 240)
    ax.set_xticks(range(0, 171, 10), minor=True)
    ax.set_xticks([0, 30, 50, 60, 90, 120, 150, 170])
    ax.set_ylim(0, max(0.43, ax.get_ylim()[1] + 0.03))
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.set_axisbelow(True)
    save(fig, f"curves-{dataset}")


def confusion(dataset: str, exp: str) -> None:
    """The final test's confusion matrix: each column, a true class, divided by its number of signs.

    A cell is the share of the column's signs that were detected as the row's
    class; the last row is the share that was not detected (at a confidence of
    0.25, as YOLOv7 draws it). The last column divides the detections that
    match no sign among the classes they were given. Rows carry the class's
    name and number, columns the number, which keeps the figure to the size of
    the article's own (about 15 cm by 10 cm, two to a page) with labels that
    can be read at that size.
    """
    path = RESULTS / dataset / exp / "confusion_matrix.csv"
    if not path.exists():
        return
    counts = np.loadtxt(path, delimiter=",")
    shares = counts / (counts.sum(0, keepdims=True) + 1e-6)
    shares[shares < 0.005] = np.nan
    n = len(CLASS_NAMES) + 1
    small = 5.5
    fig, ax = plt.subplots(figsize=(6.6, 4.35))
    mesh = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), shares, cmap="Blues", vmin=0, vmax=1,
                         edgecolors="face", linewidth=0.1)  # fmt: skip
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    numbers = [str(i) for i in range(n - 1)]
    ax.set_xticklabels([*numbers, "no sign"], rotation=90, fontsize=small - 0.5)
    ax.set_yticklabels([*numbers, ""], fontsize=small - 0.5)
    ax.tick_params(length=1.2, width=0.3, pad=1.2)
    # The names stand to the left of the numbers, in a column of their own.
    names = ax.get_yaxis_transform()
    for i, name in enumerate([*CLASS_NAMES, "not detected"]):
        ax.text(-0.052 if i < n - 1 else -0.012, i + 0.5, name, transform=names, ha="right", va="center_baseline",
                fontsize=small)  # fmt: skip
    for k in range(5, n, 5):  # faint lines every five classes, to follow a row or column by eye
        ax.axhline(k, color="#cccccc", lw=0.25, zorder=0)
        ax.axvline(k, color="#cccccc", lw=0.25, zorder=0)
    ax.set_xlabel("True class, by its number in the rows", fontsize=7, labelpad=2)
    ax.set_ylabel("Predicted class", fontsize=7, labelpad=98)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.025, aspect=40)
    bar.set_label("Share of the true class's signs", fontsize=7)
    bar.ax.tick_params(labelsize=6, length=2, width=0.3)
    bar.outline.set_linewidth(0.4)
    # Matplotlib would store the bar as a picture; keep the whole figure vector, without seams between its steps.
    bar.solids.set(rasterized=False, edgecolor="face", linewidth=0.4)
    save(fig, f"confusion-{dataset}-{exp}")


def main() -> None:
    # Matplotlib's PDF writer orders some objects by hash, so fix the hash seed:
    # then drawing the figures again gives the same files, byte for byte.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
    scores()
    for dataset in DATASETS:
        if (RESULTS / dataset).is_dir():
            curves(dataset)
            for exp in sorted(EXPERIMENTS):
                confusion(dataset, exp)


if __name__ == "__main__":
    main()
