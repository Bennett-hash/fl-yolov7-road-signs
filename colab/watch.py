"""Print the experiments' progress on the VM, for up to WATCH_MINUTES minutes.

    colab exec -s fl --timeout 0 --env WATCH_MINUTES=55 -f colab/watch.py

It prints the progress lines that ``fedyolo.run`` has written since the last
time it ran (it remembers where it stopped in ``/content/out/.watch.json``),
then keeps printing new ones until the time is up or no experiment is left
running. A running cell also tells Colab the VM is in use. Stopping this
cell does not stop the experiments.
"""

import glob
import json
import os
import time

OUT = "/content/out"
STATE = f"{OUT}/.watch.json"
INTERVAL = int(os.environ.get("WATCH_INTERVAL", "60"))
DEADLINE = time.time() + 60 * float(os.environ.get("WATCH_MINUTES", "55"))

try:
    offsets = json.load(open(STATE))
except (OSError, ValueError):
    offsets = {}


def running() -> list[str]:
    alive = []
    for pid_file in glob.glob(f"{OUT}/*.pid"):
        pid = int(open(pid_file).read())
        if os.path.exists(f"/proc/{pid}"):
            alive.append(os.path.basename(pid_file)[:-4])
    return sorted(alive)


def drain() -> None:
    for log in sorted(glob.glob(f"{OUT}/*.log")):
        # Read bytes, so that the offset stays right when a character of
        # several bytes is cut off at the end.
        with open(log, "rb") as f:
            f.seek(offsets.get(log, 0))
            new = f.read()
        # Keep a partial last line for next time.
        cut = new.rfind(b"\n") + 1
        offsets[log] = offsets.get(log, 0) + cut
        name = os.path.basename(log)[:-4]
        for line in new[:cut].decode("utf-8", "replace").splitlines():
            # fedyolo.run's own lines start with a [HH:MM:SS] stamp.
            if line.startswith("[") or "Error" in line or "Traceback" in line:
                print(f"{name}: {line}", flush=True)
    json.dump(offsets, open(STATE, "w"))


while True:
    drain()
    alive = running()
    if not alive:
        print("no experiment is running", flush=True)
        break
    if time.time() >= DEADLINE:
        print(f"watch ended; still running: {', '.join(alive)}", flush=True)
        break
    time.sleep(INTERVAL)
