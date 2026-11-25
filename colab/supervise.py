"""Look after experiments running on Colab sessions, from your own computer.

    python colab/supervise.py fl fl2

For each named ``colab`` session, in turn and for as long as it runs, this:

- runs ``colab/watch.py`` in the session for 12 minutes at a time, which shows
  progress and keeps the session in use, appending what it prints to
  ``runs/colab/<session>/watch.log``;
- copies every experiment's ``metrics.csv``, ``settings.json``,
  ``sessions.jsonl`` and ``test.DONE`` (those it has so far), and its newest
  ``global.pt`` or, for the centralised run, ``train/weights/last.pt`` and
  ``best.pt`` with ``train/opt.yaml`` and ``train/hyp.yaml``, to
  ``runs/colab/<session>/<experiment>/``, so that a lost session can be
  resumed elsewhere. Each file is downloaded beside its last copy and only
  then put in its place, so a failed download leaves the last good copy;
- checks every 3 minutes that Colab still lists every session;
- stops, printing why, when an experiment finishes, an error appears, a
  session is gone or can no longer be reached, or nothing is left running.

With ``--forever`` it does not stop for these: it appends each one, with the
time, to ``runs/colab/events.log`` and carries on, dropping a session only
when Colab no longer lists it. That is the way to leave it running unattended:

    nohup python colab/supervise.py --forever fl fl2 > runs/colab/supervise.out 2>&1 &
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Minutes between copies. A session Colab takes away loses at most what was
#: done since the last copy, so keep it shorter than a round.
WATCH_MINUTES = int(os.environ.get("WATCH_MINUTES", "12"))
STOP_PATTERNS = re.compile(
    r"finished \(|Traceback|Error|no experiment is running|authoriz|Session not found|appears to be lost|not found",
    re.I,
)


def colab(*args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def sync(session: str) -> None:
    """Copy the small files and the newest weights of every experiment."""
    listing = colab(
        "ssh", "-s", session, "-i", KEY, "--",
        "cd /content/out && for d in exp*/; do d=${d%/}; echo $d; "
        "ls $d/metrics.csv $d/settings.json $d/sessions.jsonl $d/test.DONE 2>/dev/null; "
        "ls -d $d/round_*/global.pt 2>/dev/null | tail -1; "
        "ls $d/train/weights/last.pt $d/train/weights/best.pt $d/train/opt.yaml $d/train/hyp.yaml 2>/dev/null; "
        # ls fails for whatever an experiment does not have; that is not an error here.
        "done; true",
        timeout=120,
    )  # fmt: skip
    if listing.returncode != 0:
        raise RuntimeError(f"could not list files on {session}: {listing.stderr.strip()[-300:]}")
    failed = []
    for path in listing.stdout.split():
        if "/" not in path:
            continue
        local = ROOT / "runs" / "colab" / session / path
        local.parent.mkdir(parents=True, exist_ok=True)
        # Into a file of its own first, so that a download that fails part way
        # does not spoil the copy from last time.
        part = local.with_name(local.name + ".part")
        part.unlink(missing_ok=True)
        got = colab("download", "-s", session, f"/content/out/{path}", str(part), timeout=600)
        if got.returncode != 0 or not part.exists():
            part.unlink(missing_ok=True)
            failed.append(path)
        else:
            part.replace(local)
    if failed:
        raise RuntimeError(f"could not copy {', '.join(failed)}")


def online() -> bool:
    """Whether this computer can reach Colab at all."""
    try:
        socket.create_connection(("colab.research.google.com", 443), timeout=10).close()
        return True
    except OSError:
        return False


def answers(session: str) -> bool:
    """Whether the session still runs a command over ssh. Tries three times."""
    for attempt in range(3):
        if attempt:
            time.sleep(30)
        try:
            got = colab("ssh", "-s", session, "-i", KEY, "--", "true", timeout=120)
        except subprocess.TimeoutExpired:
            continue
        # Colab allows one ssh connection at a time and answers 429 to a second:
        # the session is there, someone else is using it.
        if got.returncode == 0 or "429" in got.stdout + got.stderr:
            return True
    return False


SYNC_FAILURES: dict[str, int] = {}
TROUBLES: dict[str, int] = {}


def watch(session: str, results: dict[str, str]) -> None:
    log_dir = ROOT / "runs" / "colab" / session
    log_dir.mkdir(parents=True, exist_ok=True)
    # A session that Colab takes away can leave `colab exec` waiting for ever, so
    # give it a deadline a little past the watch's own.
    try:
        proc = subprocess.run(
            ["colab", "exec", "-s", session, "--timeout", "0", "--env", f"WATCH_MINUTES={WATCH_MINUTES}",
             "--env", "WATCH_INTERVAL=60", "-f", str(ROOT / "colab" / "watch.py")],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=(WATCH_MINUTES + 30) * 60,
        )  # fmt: skip
        text = proc.stdout + proc.stderr
        failed = proc.returncode != 0
    except subprocess.TimeoutExpired:
        text, failed = "(the watch did not return in time)\n", True
    with open(log_dir / "watch.log", "a") as f:
        f.write(text)
    lines = text.splitlines()
    # What the experiments themselves report: watch.py prefixes their lines
    # with the experiment's name.
    events = [ln for ln in lines if (ln.startswith("exp") and STOP_PATTERNS.search(ln)) or "no experiment" in ln]
    # Anything else that looks wrong comes from the connection to Colab. Often
    # that is a passing network fault, so it counts only if the session has
    # stopped answering.
    trouble = failed or any(STOP_PATTERNS.search(ln) and not ln.startswith("exp") for ln in lines)
    if trouble and not events:
        if not online():
            # This computer has lost its network. That says nothing about the
            # session, so wait for the network and try again.
            with open(log_dir / "watch.log", "a") as f:
                f.write("(this computer is offline; waiting)\n")
            while not online():
                time.sleep(30)
            results[session] = ""
            return
        if not answers(session):
            results[session] = f"{session}: the connection failed and the session no longer answers over ssh"
            return
        with open(log_dir / "watch.log", "a") as f:
            f.write("(connection trouble; the session still answers over ssh, carrying on)\n")
        # If the watch keeps failing, the notebook kernel's connection is stuck.
        # Restarting the kernel clears it and does not touch the experiments,
        # which run as processes of their own.
        TROUBLES[session] = TROUBLES.get(session, 0) + 1
        if TROUBLES[session] >= 3:
            try:
                colab("restart-kernel", "-s", session, timeout=300)
            except subprocess.TimeoutExpired:
                pass  # try again after three more failures
            TROUBLES[session] = 0
            if FOREVER:
                record(session, "restarted the notebook kernel after three failed watches in a row")
        time.sleep(60)
    else:
        TROUBLES[session] = 0
    stops = list(events)
    try:
        sync(session)
        SYNC_FAILURES[session] = 0
    except Exception as e:
        SYNC_FAILURES[session] = SYNC_FAILURES.get(session, 0) + 1
        if SYNC_FAILURES[session] >= 3:  # one bad copy is not worth waking anyone for
            stops.append(f"sync failed three times in a row: {e}")
    results[session] = "\n".join(stops)


KEY = ""  # set in main()
FOREVER = False
EVENTS = ROOT / "runs" / "colab" / "events.log"


def record(session: str, what: str) -> None:
    """Note an event in events.log, one line each, with the time."""
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(EVENTS, "a") as f:
        for line in what.splitlines():
            f.write(f"{stamp} UTC {session}: {line}\n")


def check_sessions_exist(sessions: list[str]) -> None:
    """Every 3 minutes, make sure Colab still lists every session; end the program at once if one is gone."""
    misses: dict[str, int] = {}
    while True:
        time.sleep(180)
        try:
            got = colab("sessions", timeout=120)
        except Exception:
            continue  # a failed check is not a lost session
        listing = got.stdout
        # Only a listing that worked can show that a session is gone: it names
        # some session, or says there are none.
        if got.returncode != 0 or not ("[" in listing or "No active sessions" in listing) or not online():
            continue
        for s in sessions:
            # Two misses in a row, so that one bad listing does not raise an alarm.
            misses[s] = 0 if f"[{s}]" in listing else misses.get(s, 0) + 1
            if misses[s] >= 2:
                if FOREVER:
                    record(s, f"Colab no longer lists session {s}")
                    sessions.remove(s)
                    break  # the list changed under the loop; look again in 3 minutes
                print(f"== {s}\nColab no longer lists session {s}", flush=True)
                os._exit(0)


def main() -> None:
    global KEY, FOREVER
    sessions = [a for a in sys.argv[1:] if a != "--forever"]
    FOREVER = "--forever" in sys.argv[1:]
    if not sessions:
        raise SystemExit("usage: python colab/supervise.py [--forever] SESSION [SESSION...]")
    # The key whose public half is offered to the session for `colab ssh`.
    KEY = os.environ.get("COLAB_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
    threading.Thread(target=check_sessions_exist, args=(sessions,), daemon=True).start()
    seen: set[tuple[str, str]] = set()
    while sessions:
        results: dict[str, str] = {}
        threads = [threading.Thread(target=watch, args=(s, results)) for s in list(sessions)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        reasons = {s: r for s, r in results.items() if r}
        if not FOREVER:
            if reasons:
                for s, r in reasons.items():
                    print(f"== {s}\n{r}", flush=True)
                return
            continue
        # A session is given up only when Colab stops listing it, which the
        # thread above sees. One that does not answer may be back in a minute.
        for s, r in reasons.items():
            for line in r.splitlines():
                if (s, line) not in seen:  # say each thing once
                    seen.add((s, line))
                    record(s, line)
        if reasons:
            time.sleep(60)  # an idle or unreachable session should not be asked in a tight loop
    record("supervise", "no session left to look after")


if __name__ == "__main__":
    main()
