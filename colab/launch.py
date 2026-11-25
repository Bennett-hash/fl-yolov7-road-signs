r"""Start experiments on the VM in the background, each on its own copy of the data.

    colab exec -s fl --env EXPERIMENTS="exp4 exp5 exp1+exp3+exp2" \
        --env FEDYOLO_ARGS="--parallel 4 --workers 2" -f colab/launch.py

Words separated by spaces start at the same time. Experiments joined with
``+`` run one after another in a single chain, which keeps a VM busy without
anyone having to start the next one; an experiment that fails stops the rest
of its chain. ``exp3@gtsrb`` runs exp3 on ``/content/GTSRB_YOLO_gtsrb`` into
``/content/out/exp3-gtsrb``; a plain ``exp3`` uses ``/content/GTSRB_YOLO``. ``exp1@gtsrb.run2`` is another run on
the same dataset, into ``/content/out/exp1-gtsrb.run2``. Output goes to
``/content/out/<folder>.log``.

With ``AFTER="exp5@gtsrb"`` (names of jobs started earlier by this script),
the new jobs wait until those have ended (whether or not they failed) before
they begin.

Every experiment gets its own copy of the dataset, because YOLOv7 writes label
caches beside the labels and parallel runs would otherwise write the same
files. Starting something that is already running does nothing; starting
something that stopped picks it up where it stopped.
"""

import os
import shlex
import shutil
import subprocess
import sys

CONTENT = "/content"
OUT = f"{CONTENT}/out"
EXTRA = shlex.split(os.environ.get("FEDYOLO_ARGS", ""))
WAIT = []
for job in os.environ.get("AFTER", "").split():
    if not os.path.exists(f"{OUT}/{job}.pid"):
        raise SystemExit(f"AFTER names {job}, but {OUT}/{job}.pid does not exist: no job of that name was started here")
    with open(f"{OUT}/{job}.pid") as f:
        WAIT.append(int(f.read()))

os.makedirs(OUT, exist_ok=True)
for job in os.environ.get("EXPERIMENTS", "").split():
    experiments = job.split("+")
    pid_file = f"{OUT}/{job}.pid"
    if os.path.exists(pid_file):
        pid = int(open(pid_file).read())
        if os.path.exists(f"/proc/{pid}"):
            print(f"{job}: already running (pid {pid})")
            continue
    steps = []
    for item in experiments:
        exp, _, dataset = item.partition("@")
        folder = f"{exp}-{dataset}" if dataset else exp
        # "gtsrb.run2" is a second run on the gtsrb dataset, kept in a folder of its own.
        source = f"{CONTENT}/GTSRB_YOLO_{dataset.split('.')[0]}" if dataset else f"{CONTENT}/GTSRB_YOLO"
        data = f"{CONTENT}/data/{folder}/GTSRB_YOLO"
        if not os.path.isdir(data):
            shutil.copytree(source, data)
        cmd = [sys.executable, "-m", "fedyolo.run", exp, "--yolov7", f"{CONTENT}/yolov7",
               "--data", data, "--out", OUT, *(["--tag", dataset] if dataset else []), *EXTRA]  # fmt: skip
        steps.append(f"{shlex.join(cmd)} >> {shlex.quote(f'{OUT}/{folder}.log')} 2>&1")
    # Detached from this process's input and output, so that whatever started
    # it (a notebook cell, an ssh command) can finish while the runs go on.
    if WAIT:
        pids = " ".join(map(str, WAIT))
        steps.insert(0, f"for p in {pids}; do while kill -0 $p 2>/dev/null; do sleep 60; done; done")
    proc = subprocess.Popen(["bash", "-c", " && ".join(steps)], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})  # fmt: skip
    open(pid_file, "w").write(str(proc.pid))
    print(f"{job}: started (pid {proc.pid})" + (f", after {os.environ['AFTER']}" if WAIT else ""))
