import csv

import pytest

from aegis_orbit.utils.streaming_csv import StreamingCSVWriter


def test_writer_refuses_existing_file(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        StreamingCSVWriter(path, ["first", "second"])


def test_writer_preserves_header_order(tmp_path):
    path = tmp_path / "records.csv"

    with StreamingCSVWriter(path, ["second", "first"]) as writer:
        writer.write({"first": "a", "second": "b"})

    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == ["second", "first"]


def test_writer_rejects_unexpected_key(tmp_path):
    path = tmp_path / "records.csv"

    with StreamingCSVWriter(path, ["expected"]) as writer:
        with pytest.raises(ValueError, match="dict contains fields not in fieldnames"):
            writer.write({"expected": "value", "unexpected": "value"})


def test_writer_rows_round_trip_and_missing_fields_are_blank(tmp_path):
    path = tmp_path / "records.csv"
    expected = [{"first": "one", "second": ""}, {"first": "two", "second": "value"}]

    with StreamingCSVWriter(path, ["first", "second"]) as writer:
        writer.write({"first": "one"})
        writer.write(expected[1])

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert list(reader) == expected


def test_writer_closes_parseable_file_when_context_exits_with_exception(tmp_path):
    path = tmp_path / "records.csv"

    with pytest.raises(RuntimeError, match="interrupted"):
        with StreamingCSVWriter(path, ["step", "value"]) as writer:
            writer.write({"step": 1, "value": "written"})
            raise RuntimeError("interrupted")

    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{"step": "1", "value": "written"}]
