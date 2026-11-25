"""Pack finished experiments on the VM into one file each, for copying off it.

    python3 /content/archive.py exp3-gtsrb exp2-gtsrb      # run on the VM

Writes ``/content/archive/<experiment>.tar.gz`` and prints its size. It keeps
the record of the run: every log, ``metrics.csv``, settings, the list of
sessions, the client lists, each client's per-epoch ``results.txt``, every
confusion matrix as numbers, everything the final test saved, and the final
model. It leaves out what can be made again from those or is only a
by-product: the earlier rounds' models, YOLOv7's sample pictures and plots for
each client run, and TensorBoard files.
"""

import os
import re
import sys
import tarfile

OUT = "/content/out"
DEST = "/content/archive"
KEEP_SUFFIXES = (".log", ".csv", ".txt", ".yaml", ".json", ".jsonl", ".DONE")


def wanted(rel: str, last_round: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    if parts[0] == "eval_test":
        return True  # the final test: plots and sample pictures included
    if name.endswith(".pt"):
        if parts[:2] == ["train", "weights"]:
            return name in ("best.pt", "last.pt")
        return parts[0] == last_round and name == "global.pt"
    if parts[0] == "train" and name.endswith(".png"):
        return True  # the centralised run's own curves
    return name.endswith(KEEP_SUFFIXES)


os.makedirs(DEST, exist_ok=True)
for exp in sys.argv[1:]:
    root = f"{OUT}/{exp}"
    rounds = sorted(d for d in os.listdir(root) if re.fullmatch(r"round_\d+", d))
    last_round = rounds[-1] if rounds else ""
    target = f"{DEST}/{exp}.tar.gz"
    count = 0
    with tarfile.open(target, "w:gz") as tar:
        for folder, _, files in sorted(os.walk(root)):
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(folder, f), root)
                if wanted(rel, last_round):
                    tar.add(os.path.join(folder, f), arcname=f"{exp}/{rel}")
                    count += 1
        log = f"{OUT}/{exp}.log"
        if os.path.exists(log):
            tar.add(log, arcname=f"{exp}/run.log")
            count += 1
    print(f"{exp}: {count} files, {os.path.getsize(target) / 1e6:.1f} MB", flush=True)
