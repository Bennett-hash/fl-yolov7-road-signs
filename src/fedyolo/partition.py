"""Dividing the training images among simulated clients.

The 2023 notebooks took the images in the order ``os.listdir`` returned them
and cut that list into equal consecutive parts, dropping the few left over
(``split_data`` in cell 9). That order depends on the file system and was not
recorded. Here the list is sorted and then shuffled with a fixed seed, which
gives random, equal, non-overlapping parts, the same on every machine.
"""

from __future__ import annotations

import random
from collections.abc import Iterable


def split_among_clients(images: Iterable[str], n_clients: int, seed: int = 0) -> list[list[str]]:
    """Return ``n_clients`` disjoint lists of ``len(images) // n_clients`` images each.

    The remainder, fewer than ``n_clients`` images, is left out, as it was in
    2023.
    """
    if n_clients < 1:
        raise ValueError("n_clients must be at least 1")
    pool = sorted(images)
    if len(set(pool)) != len(pool):
        raise ValueError("the image list has duplicates")
    random.Random(seed).shuffle(pool)
    size = len(pool) // n_clients
    if size == 0:
        raise ValueError(f"{len(pool)} images cannot be divided among {n_clients} clients")
    return [pool[i * size : (i + 1) * size] for i in range(n_clients)]
