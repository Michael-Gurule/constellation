import json
import logging

import pytest

from constellation.core.logging import configure_logging


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _flagged(root):
    return [h for h in root.handlers if getattr(h, "_constellation_handler", False)]


def test_text_output_goes_to_stderr(capsys):
    configure_logging("INFO")

    logging.getLogger("constellation.test").info("hello %s", "world")

    err = capsys.readouterr().err
    assert "INFO constellation.test: hello world" in err


def test_json_output_is_one_object_per_line(capsys):
    configure_logging("INFO", json=True)

    logging.getLogger("constellation.test").warning("gap of %d s", 400)

    record = json.loads(capsys.readouterr().err.strip())
    assert record["level"] == "WARNING"
    assert record["logger"] == "constellation.test"
    assert record["message"] == "gap of 400 s"
    assert record["ts"].endswith("+00:00")


def test_json_output_includes_exceptions(capsys):
    configure_logging("INFO", json=True)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger("constellation.test").exception("failed")

    record = json.loads(capsys.readouterr().err.strip())
    assert "RuntimeError: boom" in record["exc"]


def test_level_filters_records(capsys):
    configure_logging("WARNING")

    logging.getLogger("constellation.test").info("quiet")

    assert capsys.readouterr().err == ""


def test_reconfiguring_replaces_the_handler():
    configure_logging("INFO")
    configure_logging("DEBUG", json=True)

    assert len(_flagged(logging.getLogger())) == 1
