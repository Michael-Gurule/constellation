"""Seeded random number generators. There is no global random state."""

import numpy as np


def make_rng(seed: int | np.random.SeedSequence) -> np.random.Generator:
    """Create a generator from an explicit seed or seed sequence."""
    return np.random.default_rng(seed)


def spawn(seed: int | np.random.SeedSequence, n: int) -> list[np.random.Generator]:
    """Create `n` statistically independent generators from one seed."""
    sequence = (
        seed
        if isinstance(seed, np.random.SeedSequence)
        else np.random.SeedSequence(seed)
    )
    return [np.random.default_rng(child) for child in sequence.spawn(n)]
