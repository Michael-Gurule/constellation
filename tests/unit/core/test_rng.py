import numpy as np

from constellation.core.rng import make_rng, spawn


def test_same_seed_gives_same_stream():
    assert (
        make_rng(7).integers(0, 1000, 5).tolist()
        == make_rng(7).integers(0, 1000, 5).tolist()
    )


def test_different_seeds_give_different_streams():
    assert make_rng(1).random() != make_rng(2).random()


def test_seed_sequence_is_accepted():
    sequence = np.random.SeedSequence(5)

    assert make_rng(sequence).random() == make_rng(np.random.SeedSequence(5)).random()


def test_spawn_returns_independent_reproducible_generators():
    first = [g.random() for g in spawn(11, 3)]
    second = [g.random() for g in spawn(11, 3)]

    assert first == second
    assert len(set(first)) == 3


def test_spawn_count():
    assert spawn(0, 0) == []
    assert len(spawn(0, 4)) == 4
