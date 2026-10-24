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


def test_file_logging_writes_records_and_survives_rotation(tmp_path):
    log = tmp_path / "logs" / "collect.log"
    configure_logging("INFO", file=log)

    logging.getLogger("constellation.test").info("before rotation")
    log.rename(tmp_path / "logs" / "collect.log.1")  # what newsyslog does
    logging.getLogger("constellation.test").info("after rotation")

    assert "before rotation" in (tmp_path / "logs" / "collect.log.1").read_text()
    assert "after rotation" in log.read_text()  # the handler reopened the new file
    assert "before rotation" not in log.read_text()


def test_file_logging_replaces_the_stream_handler(tmp_path, capsys):
    configure_logging("INFO")
    configure_logging("INFO", file=tmp_path / "x.log")

    logging.getLogger("constellation.test").info("only in the file")

    assert capsys.readouterr().err == ""
    assert len(_flagged(logging.getLogger())) == 1
