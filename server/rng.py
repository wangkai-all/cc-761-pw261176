"""Deterministic pseudo-random helpers shared by the synthetic track.

All randomness in this project is seeded and reproducible.  The track is a
synthetic experiment; no real loss data is involved anywhere.
"""

from __future__ import annotations

import numpy as np


def rng_for(*parts: object, seed: int = 2611) -> np.random.Generator:
    """Build a generator whose stream is determined only by the given keys.

    Using a stable hash of the stringified parts keeps results reproducible
    across processes (unlike Python's salted ``hash``), which matters because
    the API and the CLI both regenerate the same synthetic world.
    """
    import hashlib

    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") ^ seed
    return np.random.default_rng(seed)
