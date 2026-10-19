"""`config_hash` depends on content, not on key order."""

import random
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from constellation.core.config import config_hash

LEAVES = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=8)
)
JSON_VALUES = st.recursive(
    LEAVES,
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(max_size=6), children, max_size=4)
    ),
    max_leaves=12,
)
CONFIGS = st.dictionaries(st.text(max_size=6), JSON_VALUES, min_size=1, max_size=6)


def _shuffled(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        items = [(k, _shuffled(v, rng)) for k, v in value.items()]
        rng.shuffle(items)
        return dict(items)
    if isinstance(value, list):
        return [_shuffled(v, rng) for v in value]
    return value


@given(CONFIGS, st.randoms(use_true_random=False))
def test_hash_is_invariant_to_key_order(
    config: dict[str, Any], rng: random.Random
) -> None:
    assert config_hash(config) == config_hash(_shuffled(config, rng))


@given(CONFIGS, st.text(min_size=1, max_size=6), st.integers())
def test_hash_changes_when_a_value_is_added(
    config: dict[str, Any], key: str, value: int
) -> None:
    extended = {**config, key: value}
    if extended == config:
        return
    assert config_hash(extended) != config_hash(config)
