import json

from cl.common.artifacts import atomic_write_json, stable_hash, write_jsonl


def test_serialization_round_trip(tmp_path):
    value = {"b": [2, 1], "a": 3}
    path = atomic_write_json(tmp_path / "value.json", value)
    assert json.loads(path.read_text()) == value
    rows = [{"id": 1}, {"id": 2}]
    jsonl = write_jsonl(tmp_path / "rows.jsonl", rows)
    assert [json.loads(line) for line in jsonl.read_text().splitlines()] == rows
    assert stable_hash(value) == stable_hash({"a": 3, "b": [2, 1]})
