"""Tests for cliffguard/eval/results_writer.py and scripts/compare_runs.py."""

from __future__ import annotations

import importlib.util
import json
import socket
import time
from pathlib import Path

import pytest

from cliffguard.eval.results_writer import (
    list_runs,
    load_hypothesis_results,
    load_run_metadata,
    make_run_dir,
    write_calibration_summary,
    write_drift_results,
    write_fold_result,
    write_gate_verdicts,
    write_hypothesis_results,
    write_manifest,
    write_run_metadata,
)
from cliffguard.eval.runner import FoldResult
from cliffguard.types import QuantScheme, Tier

# ---------------------------------------------------------------------------
# Load compare_runs from scripts/compare_runs.py via importlib so we can test
# it without installing it as a package.
# ---------------------------------------------------------------------------
_COMPARE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"
_spec = importlib.util.spec_from_file_location("_compare_runs_script", _COMPARE_SCRIPT)
assert _spec is not None
_compare_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_compare_mod)  # type: ignore[union-attr]
_compare_runs = _compare_mod.compare_runs  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fold_result(**kwargs: object) -> FoldResult:
    defaults: dict[str, object] = {
        "fold_name": "fold_b",
        "tier": Tier.A,
        "scheme": QuantScheme.FP16,
        "n_prompts": 100,
        "n_blocked": 80,
        "n_passed": 20,
        "fpr": 0.05,
        "asr": 0.20,
        "notes": [],
    }
    defaults.update(kwargs)
    return FoldResult(**defaults)  # type: ignore[arg-type]


def _write_complete_run(run_dir: Path, tier: Tier, hyp: dict[str, object]) -> None:
    """Write metadata + hypothesis results to simulate a complete run."""
    write_run_metadata(run_dir, tier, ["FP16"], "abc123")
    write_hypothesis_results(run_dir, hyp)


# ---------------------------------------------------------------------------
# make_run_dir
# ---------------------------------------------------------------------------


class TestMakeRunDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_placed_under_runs_subdir(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        assert run_dir.parent.name == "runs"
        assert run_dir.parent.parent == tmp_path

    def test_naming_starts_with_tier(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        assert run_dir.name.startswith("A_")

    def test_naming_tier_b(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.B)
        assert run_dir.name.startswith("B_")

    def test_naming_tier_c_plus(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.C_PLUS)
        assert run_dir.name.startswith("C_PLUS_")

    def test_naming_contains_date_and_time(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        # Last two segments after splitting: YYYYMMDD and HHMMSS
        parts = run_dir.name.rsplit("_", 2)
        assert len(parts) == 3
        date_part, time_part = parts[1], parts[2]
        assert len(date_part) == 8 and date_part.isdigit()
        assert len(time_part) == 6 and time_part.isdigit()

    def test_two_calls_produce_different_dirs(self, tmp_path: Path) -> None:
        dir1 = make_run_dir(tmp_path, Tier.A)
        time.sleep(1.1)
        dir2 = make_run_dir(tmp_path, Tier.A)
        assert dir1 != dir2

    def test_hostname_spaces_replaced_with_underscores(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "gethostname", lambda: "my host name")
        run_dir = make_run_dir(tmp_path, Tier.A)
        assert " " not in run_dir.name
        assert "my_host_name" in run_dir.name

    def test_hostname_truncated_to_20_chars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        long_host = "z" * 30
        monkeypatch.setattr(socket, "gethostname", lambda: long_host)
        run_dir = make_run_dir(tmp_path, Tier.A)
        # hostname sits between tier prefix "A_" and "_YYYYMMDD_HHMMSS"
        # e.g. A_zzzzzzzzzzzzzzzzzzzz_20251001_143022
        # "z" * 21 would appear if truncation didn't happen
        assert "z" * 21 not in run_dir.name


# ---------------------------------------------------------------------------
# write_run_metadata
# ---------------------------------------------------------------------------


class TestWriteRunMetadata:
    def test_creates_file(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        path = write_run_metadata(run_dir, Tier.A, ["FP16"], "abc123")
        assert path.exists()
        assert path.name == "run_metadata.json"

    def test_all_fields_present(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_run_metadata(
            run_dir,
            Tier.A,
            ["FP16", "NF4"],
            "deadbeef",
            hardware_description="test box",
            config_path="configs/test.yaml",
        )
        data = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert data["tier"] == "A"
        assert data["schemes"] == ["FP16", "NF4"]
        assert data["git_hash"] == "deadbeef"
        assert data["hardware_description"] == "test box"
        assert data["config_path"] == "configs/test.yaml"
        assert "hostname" in data
        assert "timestamp_utc" in data

    def test_config_path_defaults_to_null(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_run_metadata(run_dir, Tier.A, [], "abc")
        data = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert data["config_path"] is None

    def test_tier_serialized_as_string(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.B)
        write_run_metadata(run_dir, Tier.B, [], "abc")
        data = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert data["tier"] == "B"


# ---------------------------------------------------------------------------
# write_fold_result
# ---------------------------------------------------------------------------


class TestWriteFoldResult:
    def test_creates_summary_file(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        path = write_fold_result(run_dir, "fold_b", [_fold_result()])
        assert path.exists()
        assert path.name == "fold_b_summary.json"

    def test_creates_fold_subdirectory(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_fold_result(run_dir, "fold_c", [_fold_result()])
        assert (run_dir / "fold_c").is_dir()

    def test_n_results_field(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_fold_result(run_dir, "fold_b", [_fold_result(), _fold_result()])
        data = json.loads((run_dir / "fold_b" / "fold_b_summary.json").read_text(encoding="utf-8"))
        assert data["n_results"] == 2

    def test_tpr_and_abr_included(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        r = _fold_result(n_prompts=100, n_blocked=80, asr=0.20)
        write_fold_result(run_dir, "fold_b", [r])
        data = json.loads((run_dir / "fold_b" / "fold_b_summary.json").read_text(encoding="utf-8"))
        s = data["results"][0]
        assert "tpr" in s
        assert "abr" in s
        assert abs(s["tpr"] - 0.80) < 1e-9
        assert abs(s["abr"] - 0.80) < 1e-9

    def test_tier_and_scheme_as_strings(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        r = _fold_result(tier=Tier.B, scheme=QuantScheme.NF4)
        write_fold_result(run_dir, "fold_b", [r])
        data = json.loads((run_dir / "fold_b" / "fold_b_summary.json").read_text(encoding="utf-8"))
        s = data["results"][0]
        assert s["tier"] == "B"
        assert s["scheme"] == "NF4"

    def test_extra_field(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_fold_result(run_dir, "fold_b", [], extra={"cliff_at": "Q3_K_M"})
        data = json.loads((run_dir / "fold_b" / "fold_b_summary.json").read_text(encoding="utf-8"))
        assert data["extra"] == {"cliff_at": "Q3_K_M"}

    def test_extra_none_by_default(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_fold_result(run_dir, "fold_b", [])
        data = json.loads((run_dir / "fold_b" / "fold_b_summary.json").read_text(encoding="utf-8"))
        assert data["extra"] is None


# ---------------------------------------------------------------------------
# write_gate_verdicts
# ---------------------------------------------------------------------------


class TestWriteGateVerdicts:
    def test_creates_jsonl_file(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        path = write_gate_verdicts(run_dir, "fold_b", [{"gate": "VESTIBULE-LZ", "fired": False}])
        assert path.exists()
        assert path.name == "gate_verdicts.jsonl"

    def test_one_line_per_verdict(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        verdicts = [{"gate": "A", "score": 0.1}, {"gate": "B", "score": 0.9}]
        write_gate_verdicts(run_dir, "fold_b", verdicts)
        lines = (run_dir / "fold_b" / "gate_verdicts.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "gate" in parsed

    def test_appends_on_second_call(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_gate_verdicts(run_dir, "fold_b", [{"gate": "X", "score": 1.0}])
        write_gate_verdicts(run_dir, "fold_b", [{"gate": "Y", "score": 2.0}])
        lines = (run_dir / "fold_b" / "gate_verdicts.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        gates = [json.loads(line)["gate"] for line in lines]
        assert gates == ["X", "Y"]

    def test_creates_fold_subdirectory(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_gate_verdicts(run_dir, "fold_c", [])
        assert (run_dir / "fold_c").is_dir()


# ---------------------------------------------------------------------------
# write_calibration_summary
# ---------------------------------------------------------------------------


class TestWriteCalibrationSummary:
    def test_creates_file_in_fold_a(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        thresholds = {"PROBE-RM": {"FP16": 0.5, "NF4": 0.42}}
        path = write_calibration_summary(run_dir, thresholds)
        assert path.exists()
        assert path.name == "calibration_summary.json"
        assert path.parent.name == "fold_a"

    def test_content(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        thresholds = {"PROBE-HD": {"FP16": 0.6, "GGUF_Q3_K_M": 0.55}}
        write_calibration_summary(run_dir, thresholds)
        data = json.loads(
            (run_dir / "fold_a" / "calibration_summary.json").read_text(encoding="utf-8")
        )
        assert data["thresholds_by_primitive_scheme"] == thresholds


# ---------------------------------------------------------------------------
# write_drift_results
# ---------------------------------------------------------------------------


class TestWriteDriftResults:
    def test_creates_file_in_fold_d(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        path = write_drift_results(run_dir, {"adwin_detections": 3})
        assert path.exists()
        assert path.name == "drift_results.json"
        assert path.parent.name == "fold_d"

    def test_content_roundtrips(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        drift = {"adwin_detections": 3, "mean_latency_steps": 42.5, "scheme": "NF4"}
        write_drift_results(run_dir, drift)
        data = json.loads((run_dir / "fold_d" / "drift_results.json").read_text(encoding="utf-8"))
        assert data == drift


# ---------------------------------------------------------------------------
# write_hypothesis_results
# ---------------------------------------------------------------------------


class TestWriteHypothesisResults:
    def test_creates_file(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        path = write_hypothesis_results(run_dir, {"h1_accepted": True})
        assert path.exists()
        assert path.name == "hypothesis_results.json"

    def test_written_at_added_automatically(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_hypothesis_results(run_dir, {"h1_accepted": True})
        data = json.loads((run_dir / "hypothesis_results.json").read_text(encoding="utf-8"))
        assert "written_at" in data

    def test_original_fields_preserved(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        results: dict[str, object] = {
            "h1_accepted": False,
            "h1_summary": "Rejected",
            "h4_accepted": True,
            "h4_p": 0.003,
        }
        write_hypothesis_results(run_dir, results)
        data = json.loads((run_dir / "hypothesis_results.json").read_text(encoding="utf-8"))
        assert data["h1_accepted"] is False
        assert data["h1_summary"] == "Rejected"
        assert data["h4_accepted"] is True
        assert abs(data["h4_p"] - 0.003) < 1e-12

    def test_does_not_mutate_input(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        results: dict[str, object] = {"h1_accepted": True}
        write_hypothesis_results(run_dir, results)
        assert "written_at" not in results


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------


class TestWriteManifest:
    def test_creates_file(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        path = write_manifest(run_dir, {"git_hash": "abc", "tier": "A"})
        assert path.exists()
        assert path.name == "manifest.json"

    def test_content_roundtrips(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        manifest: dict[str, object] = {"git_hash": "deadbeef", "timestamp": "2025-01-01T00:00:00+00:00"}
        write_manifest(run_dir, manifest)
        data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert data == manifest


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_when_runs_dir_missing(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path) == []

    def test_empty_when_no_run_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "runs").mkdir()
        assert list_runs(tmp_path) == []

    def test_returns_sorted_by_name(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        # Create dirs with names that sort alphabetically in a known order
        b_dir = runs_dir / "B_host_20250101_000002"
        a_dir = runs_dir / "A_host_20250101_000001"
        b_dir.mkdir()
        a_dir.mkdir()
        result = list_runs(tmp_path)
        assert len(result) == 2
        assert result[0].name == "A_host_20250101_000001"
        assert result[1].name == "B_host_20250101_000002"

    def test_two_real_runs_returned_in_order(self, tmp_path: Path) -> None:
        dir1 = make_run_dir(tmp_path, Tier.A)
        time.sleep(1.1)
        dir2 = make_run_dir(tmp_path, Tier.A)
        result = list_runs(tmp_path)
        assert len(result) == 2
        assert result[0] == dir1
        assert result[1] == dir2

    def test_ignores_files_in_runs_dir(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "not_a_dir.json").write_text("{}")
        make_run_dir(tmp_path, Tier.A)
        result = list_runs(tmp_path)
        assert len(result) == 1
        assert result[0].is_dir()


# ---------------------------------------------------------------------------
# load_run_metadata
# ---------------------------------------------------------------------------


class TestLoadRunMetadata:
    def test_loads_written_metadata(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_run_metadata(run_dir, Tier.A, ["FP16"], "abc123")
        data = load_run_metadata(run_dir)
        assert data["tier"] == "A"
        assert data["git_hash"] == "abc123"

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "no_metadata_run"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_run_metadata(run_dir)


# ---------------------------------------------------------------------------
# load_hypothesis_results
# ---------------------------------------------------------------------------


class TestLoadHypothesisResults:
    def test_loads_written_results(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_hypothesis_results(run_dir, {"h1_accepted": True, "h4_p": 0.001})
        data = load_hypothesis_results(run_dir)
        assert data["h1_accepted"] is True
        assert abs(data["h4_p"] - 0.001) < 1e-12  # type: ignore[arg-type]

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_hypothesis_results(run_dir)


# ---------------------------------------------------------------------------
# compare_runs (from scripts/compare_runs.py)
# ---------------------------------------------------------------------------


class TestCompareRuns:
    _FULL_HYP: dict[str, object] = {
        "h1_accepted": True,
        "h1_summary": "Accepted in 2/3 families",
        "h2_accepted": True,
        "h3_accepted": False,
        "h4_accepted": True,
        "h5_tier_c_accepted": True,
    }

    def _make_complete_run(self, tmp_path: Path, tier: Tier, hyp: dict[str, object]) -> Path:
        run_dir = make_run_dir(tmp_path, tier)
        _write_complete_run(run_dir, tier, hyp)
        return run_dir

    def test_agreement_all_true_for_identical_results(self, tmp_path: Path) -> None:
        dir1 = self._make_complete_run(tmp_path, Tier.A, self._FULL_HYP)
        time.sleep(1.1)
        dir2 = self._make_complete_run(tmp_path, Tier.A, self._FULL_HYP)
        comparison = _compare_runs([dir1, dir2])
        agreement = comparison["agreement"]
        assert agreement["H1"] is True
        assert agreement["H4"] is True
        assert agreement["H5"] is True

    def test_agreement_false_when_results_differ(self, tmp_path: Path) -> None:
        hyp1: dict[str, object] = dict(self._FULL_HYP)
        hyp2: dict[str, object] = dict(self._FULL_HYP)
        hyp2["h1_accepted"] = False  # differs
        dir1 = self._make_complete_run(tmp_path, Tier.A, hyp1)
        time.sleep(1.1)
        dir2 = self._make_complete_run(tmp_path, Tier.A, hyp2)
        comparison = _compare_runs([dir1, dir2])
        assert comparison["agreement"]["H1"] is False

    def test_skips_missing_hypothesis_results_without_raising(self, tmp_path: Path) -> None:
        # Run dir with metadata but no hypothesis_results.json
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_run_metadata(run_dir, Tier.A, [], "abc")
        # Should not raise; the run should be skipped
        comparison = _compare_runs([run_dir])
        assert isinstance(comparison["runs"], list)
        assert len(comparison["runs"]) == 0

    def test_hypotheses_list_always_present(self, tmp_path: Path) -> None:
        comparison = _compare_runs([])
        assert comparison["hypotheses"] == ["H1", "H2", "H3", "H4", "H5"]

    def test_runs_list_populated_for_complete_run(self, tmp_path: Path) -> None:
        dir1 = self._make_complete_run(tmp_path, Tier.B, {"h1_accepted": True})
        comparison = _compare_runs([dir1])
        assert len(comparison["runs"]) == 1
        assert comparison["runs"][0]["tier"] == "B"  # type: ignore[index]

    def test_agreement_none_for_single_run(self, tmp_path: Path) -> None:
        dir1 = self._make_complete_run(tmp_path, Tier.A, self._FULL_HYP)
        comparison = _compare_runs([dir1])
        # Can't determine agreement with only one run
        assert comparison["agreement"]["H1"] is None

    def test_run_dir_and_git_hash_in_output(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, Tier.A)
        write_run_metadata(run_dir, Tier.A, ["FP16"], "cafebabe")
        write_hypothesis_results(run_dir, {"h1_accepted": True})
        comparison = _compare_runs([run_dir])
        run_entry = comparison["runs"][0]  # type: ignore[index]
        assert run_entry["git_hash"] == "cafebabe"  # type: ignore[index]
        assert str(run_dir) in str(run_entry["run_dir"])  # type: ignore[index]
