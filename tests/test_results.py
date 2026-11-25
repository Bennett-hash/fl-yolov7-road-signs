"""The tables in the README and in the correction note are the ones the recorded results give."""

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_collector():
    spec = importlib.util.spec_from_file_location("collect_results", ROOT / "tools" / "collect_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_tables_match_summary_csv():
    collector = load_collector()
    with open(ROOT / "results" / "summary.csv", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert collector.readme_tables(summary) in readme
    assert collector.readme_tables(summary, zh=True) in (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    correction = (ROOT / "CORRECTION.md").read_text(encoding="utf-8")
    for table in collector.correction_tables(summary).values():
        assert table in correction


def test_every_result_has_its_tables():
    with open(ROOT / "results" / "summary.csv", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    for row in summary:
        folder = ROOT / "results" / row["dataset"] / row["experiment"]
        expected = ["metrics.csv", "settings.json", "per_class.csv", "confusion_matrix.csv"]
        expected.append("clients.csv" if row["clients"] else "epochs.csv")
        for name in expected:
            assert (folder / name).exists(), f"{folder.relative_to(ROOT)}/{name}"
        with open(folder / "metrics.csv", encoding="utf-8") as f:
            test = next(r for r in csv.DictReader(f) if r["split"] == "test")
        assert test["map50"] == row["map50"]
