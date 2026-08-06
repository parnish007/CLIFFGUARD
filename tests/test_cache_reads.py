"""Cache reads must survive a half-written file.

Caches are written to Google Drive as each scheme finishes. That is what makes
an interrupted Colab session resumable, and it is also what makes a truncated
file possible: a disconnect can land in the middle of a write.

The failure this guards against is not a crash but a LOOP. `json.loads` on a
truncated file raises; the notebook's retry logic re-reads the same cache on
every attempt; so one bad file turned a resumable ladder into three identical
failures and then a give-up, having spent the session. Treating an unreadable
cache as a missing one costs the time to regenerate that one scheme.

No torch import here -- these helpers are pure filesystem, and the module they
live in only imports torch inside functions.
"""

from __future__ import annotations

import json

import numpy as np

from scripts.run_local_ladder import read_json_cache, read_npy_cache


def test_reads_a_good_json_cache(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert read_json_cache(path) == ["a", "b"]


def test_missing_json_cache_is_simply_absent(tmp_path) -> None:
    assert read_json_cache(tmp_path / "nothing.json") is None


def test_truncated_json_reads_as_absent_and_is_moved_aside(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text('["a", "b", trunc', encoding="utf-8")
    assert read_json_cache(path) is None
    assert not path.exists(), "the bad file must not be read again next attempt"
    assert (tmp_path / "c.json.corrupt").exists(), "kept for inspection, not deleted"


def test_empty_json_file_reads_as_absent(tmp_path) -> None:
    """A write that was interrupted before any bytes landed."""
    path = tmp_path / "c.json"
    path.write_text("", encoding="utf-8")
    assert read_json_cache(path) is None


def test_reads_a_good_npy_cache(tmp_path) -> None:
    path = tmp_path / "a.npy"
    np.save(path, np.ones(4))
    assert read_npy_cache(path).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_truncated_npy_reads_as_absent_and_is_moved_aside(tmp_path) -> None:
    path = tmp_path / "a.npy"
    path.write_bytes(b"\x93NUMPY not really")
    assert read_npy_cache(path) is None
    assert (tmp_path / "a.npy.corrupt").exists()


def test_quarantine_does_not_collide_across_two_bad_files(tmp_path) -> None:
    """A second corruption must not fail because the .corrupt name is taken."""
    for _ in range(2):
        path = tmp_path / "c.json"
        path.write_text("{bad", encoding="utf-8")
        assert read_json_cache(path) is None
    assert (tmp_path / "c.json.corrupt").exists()
