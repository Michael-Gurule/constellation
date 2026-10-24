from datetime import date

import pandas as pd
import pyarrow.parquet as pq
import pytest

from constellation.core import compact as compact_module
from constellation.core.compact import compact, plan_compaction
from constellation.core.errors import ConfigError, StoreError
from constellation.core.store import ParquetStore

TODAY = date(2026, 3, 3)


def at(frame, ingested):
    frame = frame.copy()
    frame["ingested_at"] = pd.Timestamp(ingested, tz="UTC").as_unit("ns")
    return frame


def part_files(store):
    return sorted(
        p.relative_to(store.root).as_posix() for p in store.root.rglob("part-*.parquet")
    )


@pytest.fixture
def store(tmp_path, make_readings):
    """Two finished days with several part files each, one with duplicates, and 'today'."""
    store = ParquetStore(tmp_path / "readings")
    for hour in range(4):  # 2026-03-01: four flushes, the last one repeats a reading
        rows = [
            (f"2026-03-01T0{hour}:00:00", "b", float(hour)),
            (f"2026-03-01T0{hour}:00:00", "a", float(hour) + 0.5),
        ]
        if hour == 3:
            rows.append(
                ("2026-03-01T00:00:00", "b", 0.0)
            )  # duplicate of the first flush
        store.write_readings(at(make_readings(rows), f"2026-03-01T0{hour}:30:00"))
    for hour in range(3):  # 2026-03-02
        store.write_readings(
            at(
                make_readings([(f"2026-03-02T0{hour}:00:00", "a", float(hour))]),
                "2026-03-02T05:00:00",
            )
        )
    for hour in range(2):  # 2026-03-03 is today, still open
        store.write_readings(
            at(
                make_readings([(f"2026-03-03T0{hour}:00:00", "a", float(hour))]),
                "2026-03-03T05:00:00",
            )
        )
    store.write_readings(
        make_readings(
            [("2026-03-01T00:00:00", "x", 1.0)], source="sim", asset_id="SIM-1"
        )
    )
    store.write_readings(
        make_readings(
            [("2026-03-01T01:00:00", "x", 2.0)], source="sim", asset_id="SIM-1"
        )
    )
    return store


def test_compaction_keeps_exactly_the_deduplicated_readings(store):
    before = store.read_readings()

    report = compact(store, today=TODAY)

    after = store.read_readings()
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    assert report.rows_before > report.rows_after
    assert len(store.read_readings(dedupe=False)) == len(
        after
    )  # no duplicates remain on disk


def test_each_finished_day_becomes_one_sorted_file(store):
    compact(store, today=TODAY)

    files = part_files(store)
    iss_march_1 = [
        f for f in files if f.startswith("source=iss/asset=ISS/date=2026-03-01")
    ]
    assert len(iss_march_1) == 1
    assert len([f for f in files if "date=2026-03-02" in f and "source=iss" in f]) == 1
    table = pq.read_table(store.root / iss_march_1[0]).to_pandas()
    assert table[["channel", "t"]].apply(tuple, axis=1).is_monotonic_increasing
    assert (
        len(table) == 8
    )  # four flushes of two readings, plus a repeat that was dropped


def test_today_and_open_partitions_are_left_alone(store):
    open_dir = store.root / "source=iss" / "asset=ISS" / "date=2026-03-02"
    (open_dir / ".part-writer.parquet.tmp").write_bytes(b"in progress")

    report = compact(store, today=TODAY)

    assert report.skipped_open == ["iss/ISS/2026-03-02"]
    files = part_files(store)
    assert (
        len([f for f in files if "date=2026-03-02" in f and "source=iss" in f]) == 3
    )  # untouched
    assert len([f for f in files if "date=2026-03-03" in f]) == 2  # today
    assert (open_dir / ".part-writer.parquet.tmp").exists()


def test_before_source_and_asset_limit_the_scope(store):
    report = compact(store, before="2026-03-02", source="iss", asset="ISS", today=TODAY)

    assert [p.date for p in report.partitions] == ["2026-03-01"]
    assert len([f for f in part_files(store) if "source=sim" in f]) == 2
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        plan_compaction(store, before="yesterday")


def test_compaction_is_idempotent(store):
    compact(store, today=TODAY)
    files = part_files(store)

    second = compact(store, today=TODAY)

    assert second.partitions == []
    assert part_files(store) == files


def test_late_files_in_a_compacted_day_merge_on_the_next_run(store, make_readings):
    compact(store, today=TODAY)
    store.write_readings(
        at(make_readings([("2026-03-01T09:00:00", "a", 9.0)]), "2026-03-03T01:00:00")
    )
    rows_before = len(store.read_readings())

    report = compact(store, today=TODAY)

    assert [p.date for p in report.partitions] == ["2026-03-01"]
    assert len(store.read_readings()) == rows_before
    assert (
        len(
            [
                f
                for f in part_files(store)
                if "source=iss" in f and "date=2026-03-01" in f
            ]
        )
        == 1
    )


def test_dry_run_reports_and_changes_nothing(store):
    files = part_files(store)

    report = compact(store, dry_run=True, today=TODAY)

    assert report.dry_run
    assert part_files(store) == files
    assert (
        report.files_before == 7 + 2
    )  # three days' worth of finished files: 4 + 3, and 2 for sim
    assert report.files_after == 3
    assert report.rows_before - report.rows_after == 1  # the one repeated reading
    assert [p.rows_removed for p in report.partitions] == [1, 0, 0]


def test_a_failure_after_writing_leaves_the_originals_and_no_stray_file(
    store, monkeypatch
):
    files = part_files(store)
    before = store.read_readings(dedupe=False)
    real = pq.read_table

    def fail_on_verification(source, *args, **kwargs):
        if "schema" not in kwargs:  # the read-back check, not the read of the originals
            raise OSError("disk went away")
        return real(source, *args, **kwargs)

    monkeypatch.setattr(compact_module.pq, "read_table", fail_on_verification)

    with pytest.raises(OSError, match="disk went away"):
        compact(store, today=TODAY)

    monkeypatch.undo()
    assert part_files(store) == files
    assert list(store.root.rglob(".*.tmp")) == []
    pd.testing.assert_frame_equal(
        store.read_readings(dedupe=False), before, check_exact=True
    )


def test_a_row_count_mismatch_after_writing_keeps_the_originals(store, monkeypatch):
    files = part_files(store)
    real = pq.read_table

    class Short:
        num_rows = 1

    def short_read(source, *args, **kwargs):
        return real(source, *args, **kwargs) if "schema" in kwargs else Short()

    monkeypatch.setattr(compact_module.pq, "read_table", short_read)

    with pytest.raises(StoreError, match="row count after writing"):
        compact(store, today=TODAY)

    monkeypatch.undo()
    assert part_files(store) == files


def test_a_partition_that_changes_during_compaction_is_not_rewritten(
    store, monkeypatch
):
    plan = plan_compaction(store, today=TODAY).partitions[0]
    plan.files[0].unlink()  # simulates an overlapping compaction removing a file

    with pytest.raises(Exception):  # noqa: B017, PT011 - FileNotFoundError or StoreError
        compact_module._compact_partition(plan)

    assert len(list(plan.directory.glob("part-*.parquet"))) == len(plan.files) - 1
