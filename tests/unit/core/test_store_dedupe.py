import numpy as np
import pandas as pd

from constellation.core.store import ParquetStore, drop_duplicate_readings


def at(frame, ingested):
    frame = frame.copy()
    frame["ingested_at"] = pd.Timestamp(ingested, tz="UTC").as_unit("ns")
    return frame


def test_exact_duplicates_across_part_files_count_once(tmp_path, make_readings):
    rows = [("2026-03-01T00:00:00", "a", 1.0), ("2026-03-01T00:00:10", "a", 2.0)]
    store = ParquetStore(tmp_path)
    store.write_readings(at(make_readings(rows), "2026-03-05T10:00:00"))
    store.write_readings(
        at(make_readings(rows), "2026-03-05T11:00:00")
    )  # a synced copy, say

    deduped = store.read_readings()
    raw = store.read_readings(dedupe=False)

    assert len(raw) == 4
    assert len(deduped) == 2
    assert deduped["value"].tolist() == [1.0, 2.0]


def test_the_earliest_ingested_at_is_kept(tmp_path, make_readings):
    rows = [("2026-03-01T00:00:00", "a", 1.0)]
    store = ParquetStore(tmp_path)
    store.write_readings(at(make_readings(rows), "2026-03-06T00:00:00"))
    store.write_readings(at(make_readings(rows), "2026-03-05T00:00:00"))
    store.write_readings(at(make_readings(rows), "2026-03-07T00:00:00"))

    (row,) = store.read_readings().itertuples()

    assert row.ingested_at == pd.Timestamp("2026-03-05T00:00:00", tz="UTC")


def test_readings_that_differ_in_any_compared_column_are_all_kept(
    tmp_path, make_readings
):
    base = make_readings([("2026-03-01T00:00:00", "a", 1.0)])
    store = ParquetStore(tmp_path)
    variants = [base]
    for column, value in (("value", 2.0), ("value_text", "ON"), ("status", "2")):
        changed = base.copy()
        changed[column] = (
            pd.Series([value], dtype=object) if column != "value" else value
        )
        variants.append(changed)
    variants.append(base.assign(channel="b"))
    variants.append(make_readings([("2026-03-01T00:00:01", "a", 1.0)]))
    for variant in variants:
        store.write_readings(variant)

    assert len(store.read_readings()) == len(variants)


def test_missing_values_compare_equal(tmp_path, make_readings):
    frame = make_readings([("2026-03-01T00:00:00", "state", float("nan"))])
    frame["status"] = pd.Series([None], dtype=object)
    frame["value_text"] = pd.Series(["ON"], dtype=object)
    store = ParquetStore(tmp_path)
    store.write_readings(at(frame, "2026-03-05T10:00:00"))
    store.write_readings(at(frame, "2026-03-05T11:00:00"))

    result = store.read_readings()

    assert len(result) == 1
    assert np.isnan(result["value"].iloc[0])


def test_stats_count_each_reading_once_unless_asked_otherwise(tmp_path, make_readings):
    rows = [("2026-03-01T00:00:00", "a", 1.0), ("2026-03-01T00:00:10", "a", 2.0)]
    store = ParquetStore(tmp_path)
    store.write_readings(at(make_readings(rows), "2026-03-05T10:00:00"))
    store.write_readings(at(make_readings(rows), "2026-03-05T11:00:00"))

    assert store.stats()["rows"].tolist() == [2]
    assert store.stats(dedupe=False)["rows"].tolist() == [4]


def test_drop_duplicate_readings_handles_empty_frames(make_readings):
    empty = make_readings([("2026-03-01", "a", 1.0)]).iloc[0:0]

    assert drop_duplicate_readings(empty).empty
