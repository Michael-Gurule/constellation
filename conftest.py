"""Shared pytest configuration."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that need network access (marked 'live')",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs network; pass --run-live to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
