import csv
import hashlib
from pathlib import Path

import pytest

from fedyolo.data import SPLITS
from fedyolo.experiments import CLASS_NAMES, EXPERIMENTS
from fedyolo.partition import split_among_clients
from fedyolo.yolo import parse_scores

IMAGES = [f"{c:05d}_{k}.png" for c in range(43) for k in range(1, 730)][:31367]


@pytest.mark.parametrize("n", [4, 16])
def test_clients_get_equal_disjoint_parts(n):
    parts = split_among_clients(IMAGES, n)
    assert len(parts) == n
    assert {len(p) for p in parts} == {len(IMAGES) // n}
    seen = [x for p in parts for x in p]
    assert len(seen) == len(set(seen))
    assert len(IMAGES) - len(seen) < n  # only the remainder is left out


def test_split_is_reproducible_and_seeded():
    shuffled = list(reversed(IMAGES))
    assert split_among_clients(IMAGES, 4, seed=0) == split_among_clients(shuffled, 4, seed=0)
    assert split_among_clients(IMAGES, 4, seed=0) != split_among_clients(IMAGES, 4, seed=1)


#: sha256 of the image names of all the clients' lists, client 1 first, one name per line (no newline at the
#: end), for the seed-0 split of the 31,367 training images. These are the lists the federated runs behind
#: results/ trained on (clients/client_*.txt in each run's archive, with the folder taken off each path):
#: 4 clients for exp3 to exp5, 16 for exp2, on both datasets.
ARCHIVED_SPLITS = {
    4: "aed79984bafd9264b3703acc595b9e5e9f6c8182410e9b00b1620a50ef2f0966",
    16: "200a8bf0136d75edcdc52ad78f3a9158965b7e1688d1b61acc9fc9718134d89e",
}


@pytest.mark.parametrize("n", sorted(ARCHIVED_SPLITS))
def test_seed_0_split_is_the_one_the_archived_runs_used(n):
    with open(SPLITS, encoding="utf-8") as f:
        names = [r["name"] + ".png" for r in csv.DictReader(f) if r["split"] == "train"]
    parts = split_among_clients(names, n, seed=0)
    flat = "\n".join(name for part in parts for name in part)
    assert hashlib.sha256(flat.encode()).hexdigest() == ARCHIVED_SPLITS[n]
    # fedyolo.run passes the images as paths in one folder. They sort as their
    # names do, so the split is the same with the folder in front.
    folder = "/content/data/exp4-gtsrb/GTSRB_YOLO/images/train/"
    assert split_among_clients([folder + x for x in names], n, seed=0) == [[folder + x for x in p] for p in parts]


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        split_among_clients(["a", "a"], 1)
    with pytest.raises(ValueError):
        split_among_clients(["a"], 2)
    with pytest.raises(ValueError):
        split_among_clients(["a"], 0)


def test_experiments_match_table_2_of_the_paper():
    table = {name: (e.clients, e.rounds, e.epochs) for name, e in EXPERIMENTS.items()}
    assert table == {
        "exp1": (0, 1, 50),
        "exp2": (16, 17, 3),
        "exp3": (4, 17, 3),
        "exp4": (4, 17, 10),
        "exp5": (4, 40, 3),
    }
    assert len(CLASS_NAMES) == 43


def test_parse_scores_reads_the_last_summary_row():
    # Lines as printed in original/notebooks/exp1_centralised.ipynb.
    text = (
        "               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95\n"
        "                 all        3921        3921       0.527       0.752       0.642       0.425\n"
        "      speed limit 20        3921          21       0.338       0.952       0.711       0.493\n"
        "                 all        3921        3921       0.495       0.779       0.646        0.43\n"
        "Speed: 0.3/1.0/1.2 ms inference/NMS/total per 64x64 image at batch-size 32\n"
    )
    s = parse_scores(text)
    assert (s.images, s.labels, s.precision, s.recall, s.map50, s.map50_95) == (3921, 3921, 0.495, 0.779, 0.646, 0.43)
    assert parse_scores("no summary here") is None


def test_original_notebooks_are_kept():
    notebooks = sorted(p.name for p in (Path(__file__).parents[1] / "original" / "notebooks").glob("*.ipynb"))
    assert notebooks == [
        "exp1_centralised.ipynb",
        "exp2_16clients_17rounds_3epochs.ipynb",
        "exp3_4clients_17rounds_3epochs.ipynb",
        "exp4_4clients_17rounds_10epochs.ipynb",
        "exp5_4clients_40rounds_3epochs.ipynb",
    ]
