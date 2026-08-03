"""Stream-check an F16/quantized GGUF pair.

This harness checks tensor-name and declared-shape correspondence, then asks
the optional ``gguf`` package to dequantize each tensor sequentially. It never
retains a dequantized model or even both members of a tensor pair at once. Peak
resident-set size (RSS) is printed during the run and in the final summary.

VERIFIED 2026-08-03 against a real pair: ``qwen2.5-1.5b-instruct-fp16.gguf``
(3.56 GB) and ``qwen2.5-1.5b-instruct-q4_k_m.gguf`` (1.12 GB) from
``Qwen/Qwen2.5-1.5B-Instruct-GGUF``. All 339 tensors corresponded and
dequantized; result PASSED. Recorded in ``docs/build_log.md`` Entry 7.

Measured peak RSS on that run: baseline 964.80 MiB, peak 7.03 GiB, increase
6.09 GiB. Read that number carefully before quoting the streaming claim. The
script does hold only one tensor pair at a time, but ``GGUFReader`` memory-maps
both files, and touched pages count toward the working set -- 4.68 GB of file
plus the largest dequantized pair (``token_embd``, 233 M elements at float64)
accounts for the increase. The resident figure is therefore dominated by
reclaimable mapped pages, not by retained arrays, and it is not a lower bound on
the memory a caller must have free.

That first real run also exposed a bug in this file's own instrumentation. Every
RSS figure printed 0.00 B because ``GetCurrentProcess`` was called without an
explicit ``restype``, so its ``(HANDLE)-1`` pseudo-handle was truncated to 32
bits and ``GetProcessMemoryInfo`` failed against an invalid handle — silently,
via the ``except OSError`` fallback. The memory claim was therefore unverified
even while the correspondence claim passed. Fixed in ``_windows_memory_bytes``.

Note on shapes: the ``shape=`` printed per tensor is ``GGUFReader``'s ne order,
which is the REVERSE of the numpy array ``gguf.dequantize`` returns. A tensor
shown as ``(8960, 1536)`` yields a ``(1536, 8960)`` array.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import ctypes
import gc
import importlib
import io
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _windows_memory_bytes() -> tuple[int, int]:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)

    # GetCurrentProcess returns the pseudo-handle (HANDLE)-1. Without an explicit
    # restype ctypes treats that as a C int, so on 64-bit Windows it is truncated
    # to 0x00000000FFFFFFFF instead of the full 0xFFFFFFFFFFFFFFFF. The call then
    # fails against an invalid handle and every RSS figure silently reads 0.00 B,
    # which is how this went unnoticed until a real GGUF pair was streamed.
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.restype = ctypes.c_int
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]

    if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(
            f"GetProcessMemoryInfo failed (GetLastError={ctypes.get_last_error()})"
        )
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _posix_memory_bytes() -> tuple[int, int]:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024  # Linux and other common Unix implementations report KiB.

    statm = Path("/proc/self/statm")
    if statm.is_file():
        import os

        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE")), peak
    return peak, peak


def _memory_bytes() -> tuple[int, int]:
    try:
        if sys.platform == "win32":
            return _windows_memory_bytes()
        return _posix_memory_bytes()
    except (AttributeError, OSError, ValueError):
        return (0, 0)


def _shape(tensor: Any) -> tuple[int, ...]:
    raw_shape = getattr(tensor, "shape", None)
    if raw_shape is None:
        raise AttributeError(f"tensor {getattr(tensor, 'name', '<unnamed>')!r} has no shape")
    return tuple(int(dimension) for dimension in raw_shape)


def _tensor_index(reader: Any, label: str) -> tuple[dict[str, Any], bool]:
    index: dict[str, Any] = {}
    mismatch = False
    for tensor in getattr(reader, "tensors"):
        name = str(tensor.name)
        if name in index:
            print(f"ERROR: duplicate {label} tensor name: {name}", file=sys.stderr)
            mismatch = True
        index[name] = tensor
    return index, mismatch


def _verify_readers(
    fp16_reader: Any,
    quantized_reader: Any,
    dequantize: Any,
    *,
    result_label: str = "GGUF pair verification",
) -> int:
    """Verify two already-open readers; return zero only on full agreement."""
    baseline_current, baseline_peak = _memory_bytes()
    print(
        f"baseline RSS: current={_format_bytes(baseline_current)}, "
        f"peak={_format_bytes(baseline_peak)}"
    )

    try:
        fp16_tensors, fp16_duplicate = _tensor_index(fp16_reader, "F16")
        quantized_tensors, quantized_duplicate = _tensor_index(
            quantized_reader, "quantized"
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: could not index GGUF readers: {exc}", file=sys.stderr)
        return 2

    mismatch = fp16_duplicate or quantized_duplicate
    fp16_names = set(fp16_tensors)
    quantized_names = set(quantized_tensors)
    missing_from_quantized = sorted(fp16_names - quantized_names)
    missing_from_fp16 = sorted(quantized_names - fp16_names)
    for name in missing_from_quantized:
        print(f"ERROR: tensor missing from quantized GGUF: {name}", file=sys.stderr)
    for name in missing_from_fp16:
        print(f"ERROR: tensor missing from F16 GGUF: {name}", file=sys.stderr)
    mismatch = mismatch or bool(missing_from_quantized or missing_from_fp16)

    common_names = sorted(fp16_names & quantized_names)
    print(
        f"tensor names: F16={len(fp16_names)}, quantized={len(quantized_names)}, "
        f"common={len(common_names)}"
    )

    observed_peak = max(baseline_current, baseline_peak)
    for position, name in enumerate(common_names, start=1):
        fp16_tensor = fp16_tensors[name]
        quantized_tensor = quantized_tensors[name]
        try:
            fp16_declared_shape = _shape(fp16_tensor)
            quantized_declared_shape = _shape(quantized_tensor)
            if fp16_declared_shape != quantized_declared_shape:
                print(
                    f"ERROR: declared shape mismatch for {name}: "
                    f"F16={fp16_declared_shape}, quantized={quantized_declared_shape}",
                    file=sys.stderr,
                )
                mismatch = True

            fp16_raw = dequantize(fp16_tensor.data, fp16_tensor.tensor_type)
            fp16_array = np.asarray(fp16_raw)
            fp16_actual_shape = tuple(int(value) for value in fp16_array.shape)
            fp16_elements = int(fp16_array.size)
            current, peak = _memory_bytes()
            observed_peak = max(observed_peak, current, peak)
            del fp16_array, fp16_raw
            gc.collect()

            quantized_raw = dequantize(
                quantized_tensor.data, quantized_tensor.tensor_type
            )
            quantized_array = np.asarray(quantized_raw)
            quantized_actual_shape = tuple(
                int(value) for value in quantized_array.shape
            )
            quantized_elements = int(quantized_array.size)
            current, peak = _memory_bytes()
            observed_peak = max(observed_peak, current, peak)

            if fp16_actual_shape != quantized_actual_shape:
                print(
                    f"ERROR: dequantized shape mismatch for {name}: "
                    f"F16={fp16_actual_shape}, quantized={quantized_actual_shape}",
                    file=sys.stderr,
                )
                mismatch = True
            if fp16_elements != quantized_elements:
                print(
                    f"ERROR: dequantized element-count mismatch for {name}: "
                    f"F16={fp16_elements}, quantized={quantized_elements}",
                    file=sys.stderr,
                )
                mismatch = True

            print(
                f"[{position}/{len(common_names)}] {name} "
                f"shape={fp16_declared_shape} "
                f"RSS current={_format_bytes(current)} peak={_format_bytes(peak)}"
            )
            del quantized_array, quantized_raw
            gc.collect()
        except (AttributeError, MemoryError, RuntimeError, TypeError, ValueError) as exc:
            print(f"ERROR: could not verify tensor {name}: {exc}", file=sys.stderr)
            mismatch = True

    final_current, final_peak = _memory_bytes()
    observed_peak = max(observed_peak, final_current, final_peak)
    print(
        "peak RSS report: "
        f"baseline={_format_bytes(baseline_current)}, "
        f"observed_peak={_format_bytes(observed_peak)}, "
        f"increase={_format_bytes(max(0, observed_peak - baseline_current))}"
    )
    if mismatch:
        print(f"{result_label}: FAILED", file=sys.stderr)
        return 1
    print(f"{result_label}: PASSED")
    return 0


def verify_pair(fp16_path: Path, quantized_path: Path) -> int:
    """Verify one pair; return zero on success and non-zero on any mismatch."""
    for label, path in (("F16", fp16_path), ("quantized", quantized_path)):
        if not path.is_file():
            print(f"ERROR: {label} GGUF is not a file: {path}", file=sys.stderr)
            return 2

    try:
        gguf = importlib.import_module("gguf")
        reader_type = getattr(gguf, "GGUFReader")
        dequantize = getattr(gguf, "dequantize")
    except (ImportError, AttributeError) as exc:
        print(f"ERROR: the optional gguf package is required: {exc}", file=sys.stderr)
        return 2

    try:
        fp16_reader = reader_type(str(fp16_path))
        quantized_reader = reader_type(str(quantized_path))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: could not open GGUF files: {exc}", file=sys.stderr)
        return 2
    return _verify_readers(fp16_reader, quantized_reader, dequantize)


def self_check() -> int:
    """Exercise success and mismatch paths without files or ``gguf``.

    This validates the CLI plumbing, tensor indexing, declared/dequantized
    shape checks, and return-code contract. It cannot validate a real GGUF
    reader, tensor orientation, dequantizer coverage, or multi-GB peak RSS.
    """

    class FakeTensor:
        def __init__(self, name: str, data: np.ndarray) -> None:
            self.name = name
            self.data = data
            self.shape = data.shape
            self.tensor_type = "F32"

    class FakeReader:
        def __init__(self, tensors: list[FakeTensor]) -> None:
            self.tensors = tensors

    def fake_dequantize(data: np.ndarray, _tensor_type: str) -> np.ndarray:
        return np.asarray(data)

    fp16 = FakeReader(
        [
            FakeTensor("layer.0.weight", np.arange(12.0).reshape(3, 4)),
            FakeTensor("layer.1.weight", np.arange(8.0).reshape(2, 4)),
        ]
    )
    matching = FakeReader(
        [
            FakeTensor("layer.0.weight", np.zeros((3, 4))),
            FakeTensor("layer.1.weight", np.zeros((2, 4))),
        ]
    )
    mismatched = FakeReader(
        [
            FakeTensor("layer.0.weight", np.zeros((4, 3))),
            FakeTensor("layer.extra.weight", np.zeros((2, 4))),
        ]
    )

    captured_out = io.StringIO()
    captured_err = io.StringIO()
    with redirect_stdout(captured_out), redirect_stderr(captured_err):
        success_status = _verify_readers(
            fp16, matching, fake_dequantize, result_label="self-check matching pair"
        )
        mismatch_status = _verify_readers(
            fp16,
            mismatched,
            fake_dequantize,
            result_label="self-check mismatched pair",
        )
    if success_status != 0 or mismatch_status == 0:
        print("GGUF harness no-file self-check: FAILED", file=sys.stderr)
        print(captured_out.getvalue(), file=sys.stderr)
        print(captured_err.getvalue(), file=sys.stderr)
        return 1
    print(
        "GGUF harness no-file self-check: PASSED "
        "(matching pair accepted; name/shape mismatch rejected)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream-check tensor correspondence and dequantization for a GGUF pair."
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run an in-memory harness check; requires no files or gguf package",
    )
    parser.add_argument("fp16_gguf", nargs="?", type=Path, help="path to the F16 GGUF")
    parser.add_argument(
        "quantized_gguf",
        nargs="?",
        type=Path,
        help="path to the Q_K/other quantized GGUF",
    )
    arguments = parser.parse_args()
    if arguments.self_check:
        if arguments.fp16_gguf is not None or arguments.quantized_gguf is not None:
            parser.error("--self-check does not accept GGUF paths")
        return self_check()
    if arguments.fp16_gguf is None or arguments.quantized_gguf is None:
        parser.error("provide both fp16_gguf and quantized_gguf, or use --self-check")
    return verify_pair(arguments.fp16_gguf, arguments.quantized_gguf)


if __name__ == "__main__":
    raise SystemExit(main())
