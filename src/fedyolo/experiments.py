"""The five experiments of the paper (Table 2), and the settings they share."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    #: 0 for centralised training on the whole training set.
    clients: int
    #: Communication rounds ("global iterations"). 1 for centralised training.
    rounds: int
    #: Epochs per client per round, or the total for centralised training.
    epochs: int
    note: str = ""

    @property
    def centralised(self) -> bool:
        return self.clients == 0


EXPERIMENTS: dict[str, Experiment] = {
    e.name: e
    for e in (
        Experiment("exp1", 0, 1, 50, "Centralised baseline"),
        Experiment("exp2", 16, 17, 3, "16 simulated edge devices"),
        Experiment("exp3", 4, 17, 3, "Fewer clients"),
        Experiment("exp4", 4, 17, 10, "More local epochs"),
        Experiment("exp5", 4, 40, 3, "More rounds"),
    )
}

#: Settings every run in the notebooks used (Section 2.3 of the paper).
IMG_SIZE = 64
BATCH_SIZE = 16
INITIAL_WEIGHTS = "yolov7-tiny.pt"

#: The 43 GTSRB classes, in the order of the notebooks' data.yaml.
CLASS_NAMES = [
    "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60",
    "speed limit 70", "speed limit 80", "restriction ends 80", "speed limit 100",
    "speed limit 120", "no overtaking", "no overtaking (trucks)",
    "priority at next intersection", "priority road", "give way", "stop",
    "no traffic both ways", "no trucks", "no entry", "danger", "bend left",
    "bend right", "bend", "uneven road", "slippery road", "road narrows",
    "construction", "traffic signal", "pedestrian crossing", "school crossing",
    "cycles crossing", "snow", "animals", "restriction ends", "go right",
    "go left", "go straight", "go right or straight", "go left or straight",
    "keep right", "keep left", "roundabout", "restriction ends (overtaking)",
    "restriction ends (overtaking (trucks))",
]  # fmt: skip
