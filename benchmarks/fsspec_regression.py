#!/usr/bin/env python
"""Performance regression benchmarks for fsspec I/O operations.

This script benchmarks segy library performance with different fsspec versions
to identify regressions in I/O operations like cat_ranges, read_block, etc.

Usage:
    python benchmarks/fsspec_regression.py --output results.json
    python benchmarks/fsspec_regression.py --data-path /local/file.sgy
    python benchmarks/fsspec_regression.py --concurrency-test --workers 1,2,4,8
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import fsspec
import numpy as np
from fsspec.core import url_to_fs

# Add parent to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from segy import SegyFile
from segy.config import SegyFileSettings

from instrumented_fs import InstrumentedFileSystem
from instrumented_fs import IOMetrics
from instrumented_fs import NetworkBytesTracker

if TYPE_CHECKING:
    from collections.abc import Callable

# Default Parihaka dataset paths
DEFAULT_S3_PATH = (
    "s3://open.source.geoscience/open_data/newzealand/"
    "Taranaiki_Basin/PARIHAKA-3D/Parihaka_PSTM_full_angle.sgy"
)
DEFAULT_HTTP_PATH = (
    "http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/"
    "Taranaiki_Basin/PARIHAKA-3D/Parihaka_PSTM_full_angle.sgy"
)


@dataclass
class ThroughputMetrics:
    """Throughput and I/O operation metrics.

    Tracks multiple levels of throughput measurement:
    - Logical throughput: Based on bytes_requested (what the application asked for)
    - API throughput: Based on bytes_transferred (what cat_ranges returned)
    - Network throughput: Based on network_bytes_recv (actual OS-level network I/O)

    The network_overhead ratio reveals true prefetching behavior that is invisible
    at the API level.

    Attributes:
        bytes_requested: Total bytes requested by the application (logical bytes).
        bytes_transferred: Total bytes returned by cat_ranges (API level).
        network_bytes_recv: Actual bytes received at OS level (from psutil).
        network_bytes_sent: Actual bytes sent at OS level (from psutil).
        throughput_mbps: Throughput based on bytes_requested (logical throughput).
        throughput_actual_mbps: Throughput based on bytes_transferred (API throughput).
        throughput_network_mbps: Throughput based on network_bytes_recv (true network).
        prefetch_ratio: Ratio of bytes_transferred to bytes_requested (API level).
        network_overhead: Ratio of network_bytes_recv to bytes_requested.
            Values > 1.0 indicate prefetching at the network level.
        read_block_calls: Number of read_block() calls.
        cat_ranges_calls: Number of cat_ranges() calls.
        ranges_requested: Total individual byte ranges requested.
    """

    bytes_requested: int
    bytes_transferred: int
    network_bytes_recv: int
    network_bytes_sent: int
    throughput_mbps: float
    throughput_actual_mbps: float
    throughput_network_mbps: float
    prefetch_ratio: float
    network_overhead: float
    read_block_calls: int
    cat_ranges_calls: int
    ranges_requested: int

    # Backward compatibility alias
    @property
    def bytes_read(self) -> int:
        """Alias for bytes_requested for backward compatibility."""
        return self.bytes_requested

    @classmethod
    def from_io_metrics(
        cls,
        metrics: IOMetrics,
        elapsed_seconds: float,
        network_tracker: NetworkBytesTracker | None = None,
    ) -> ThroughputMetrics:
        """Create from IOMetrics, elapsed time, and optional network tracker.

        Calculates throughput at multiple levels:
        - Logical (based on requested bytes)
        - API (based on transferred bytes from cat_ranges)
        - Network (based on actual OS-level network I/O)
        """
        if elapsed_seconds > 0:
            throughput_logical = (metrics.bytes_requested / (1024 * 1024)) / elapsed_seconds
            throughput_actual = (metrics.bytes_transferred / (1024 * 1024)) / elapsed_seconds
        else:
            throughput_logical = 0.0
            throughput_actual = 0.0

        # Network-level metrics from psutil
        network_bytes_recv = 0
        network_bytes_sent = 0
        throughput_network = 0.0
        network_overhead = 1.0

        if network_tracker is not None:
            network_bytes_recv = network_tracker.bytes_recv
            network_bytes_sent = network_tracker.bytes_sent
            if elapsed_seconds > 0:
                throughput_network = (network_bytes_recv / (1024 * 1024)) / elapsed_seconds
            if metrics.bytes_requested > 0:
                network_overhead = network_bytes_recv / metrics.bytes_requested

        return cls(
            bytes_requested=metrics.bytes_requested,
            bytes_transferred=metrics.bytes_transferred,
            network_bytes_recv=network_bytes_recv,
            network_bytes_sent=network_bytes_sent,
            throughput_mbps=round(throughput_logical, 2),
            throughput_actual_mbps=round(throughput_actual, 2),
            throughput_network_mbps=round(throughput_network, 2),
            prefetch_ratio=round(metrics.prefetch_ratio, 3),
            network_overhead=round(network_overhead, 3),
            read_block_calls=metrics.read_block_calls,
            cat_ranges_calls=metrics.cat_ranges_calls,
            ranges_requested=metrics.ranges_requested,
        )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark operation."""

    operation: str
    times: list[float]
    mean: float
    std: float
    min: float
    max: float
    median: float
    throughput: ThroughputMetrics | None = None

    @classmethod
    def from_times(
        cls,
        operation: str,
        times: list[float],
        throughput: ThroughputMetrics | None = None,
    ) -> BenchmarkResult:
        """Create result from list of timing measurements."""
        return cls(
            operation=operation,
            times=times,
            mean=round(statistics.mean(times), 4),
            std=round(statistics.stdev(times) if len(times) > 1 else 0.0, 4),
            min=round(min(times), 4),
            max=round(max(times), 4),
            median=round(statistics.median(times), 4),
            throughput=throughput,
        )


@dataclass
class FileInfo:
    """Information about the benchmark file."""

    size_bytes: int
    num_traces: int
    samples_per_trace: int
    trace_size_bytes: int


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    fsspec_version: str
    python_version: str
    platform_info: str
    timestamp: str
    data_path: str
    file_info: FileInfo
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""

        def throughput_to_dict(tp: ThroughputMetrics | None) -> dict[str, Any] | None:
            if tp is None:
                return None
            d = asdict(tp)
            # Add bytes_read alias for backward compatibility
            d["bytes_read"] = tp.bytes_requested
            return d

        return {
            "fsspec_version": self.fsspec_version,
            "python_version": self.python_version,
            "platform": self.platform_info,
            "timestamp": self.timestamp,
            "data_path": self.data_path,
            "file_info": asdict(self.file_info),
            "results": [
                {
                    **{k: v for k, v in asdict(r).items() if k != "throughput"},
                    "throughput": throughput_to_dict(r.throughput),
                }
                for r in self.results
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def run_benchmark(
    func: Callable[[], Any],
    warmup: int,
    iterations: int,
    instrumented_fs: InstrumentedFileSystem | None = None,
    network_tracker: NetworkBytesTracker | None = None,
) -> tuple[list[float], ThroughputMetrics | None]:
    """Run a benchmark function with warmup and iterations.

    Args:
        func: Function to benchmark (should return None or be side-effect only)
        warmup: Number of warmup iterations to discard
        iterations: Number of measured iterations
        instrumented_fs: Optional instrumented filesystem to capture API-level metrics
        network_tracker: Optional network tracker to capture OS-level network bytes

    Returns:
        Tuple of (list of times in seconds, throughput metrics or None)
    """
    # Warmup runs
    for _ in range(warmup):
        func()
        gc.collect()

    # Reset metrics after warmup
    if instrumented_fs:
        instrumented_fs.metrics.reset()
    if network_tracker:
        network_tracker.reset()

    # Measured runs
    times: list[float] = []
    for _ in range(iterations):
        gc.collect()

        # Start network tracking before the operation
        if network_tracker:
            network_tracker.start()

        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start

        # Stop network tracking after the operation
        if network_tracker:
            network_tracker.stop()

        times.append(elapsed)

    # Calculate throughput from accumulated metrics
    throughput = None
    if instrumented_fs:
        total_time = sum(times)
        throughput = ThroughputMetrics.from_io_metrics(
            instrumented_fs.metrics, total_time, network_tracker
        )

    return times, throughput


def create_instrumented_segy(
    path: str, settings: SegyFileSettings
) -> tuple[SegyFile, InstrumentedFileSystem]:
    """Create a SegyFile with an instrumented filesystem.

    This patches the SegyFile to use our instrumented filesystem wrapper.
    """
    fs, url = url_to_fs(path, **settings.storage_options)
    instrumented = InstrumentedFileSystem(fs)

    # Create SegyFile normally first
    sgy = SegyFile(path, settings=settings)

    # Replace the filesystem with instrumented version
    sgy.fs = instrumented  # type: ignore[assignment]

    return sgy, instrumented


def benchmark_file_open(
    path: str,
    settings: SegyFileSettings,
    warmup: int,
    iterations: int,
    verbose: bool = False,
) -> BenchmarkResult:
    """Benchmark file opening (spec inference, header reads)."""
    if verbose:
        print("  Benchmarking: file_open")

    def open_file() -> None:
        sgy = SegyFile(path, settings=settings)
        # Access properties to trigger lazy loading
        _ = sgy.text_header
        _ = sgy.binary_header

    times, _ = run_benchmark(open_file, warmup, iterations)
    return BenchmarkResult.from_times("file_open", times)


def benchmark_sequential_read(
    sgy: SegyFile,
    instrumented: InstrumentedFileSystem,
    start: int,
    count: int,
    warmup: int,
    iterations: int,
    verbose: bool = False,
    network_tracker: NetworkBytesTracker | None = None,
) -> BenchmarkResult:
    """Benchmark sequential (contiguous) trace reads."""
    if verbose:
        print(f"  Benchmarking: sequential_read_{count}")

    def read_sequential() -> None:
        _ = sgy.trace[start : start + count]

    times, throughput = run_benchmark(
        read_sequential, warmup, iterations, instrumented, network_tracker
    )
    return BenchmarkResult.from_times(
        f"sequential_read_{count}", times, throughput
    )


def benchmark_random_read(
    sgy: SegyFile,
    instrumented: InstrumentedFileSystem,
    count: int,
    seed: int,
    warmup: int,
    iterations: int,
    verbose: bool = False,
    network_tracker: NetworkBytesTracker | None = None,
) -> BenchmarkResult:
    """Benchmark random (non-contiguous) trace reads."""
    if verbose:
        print(f"  Benchmarking: random_read_{count}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(sgy.num_traces, size=count, replace=False)

    def read_random() -> None:
        _ = sgy.trace[indices]

    times, throughput = run_benchmark(read_random, warmup, iterations, instrumented, network_tracker)
    return BenchmarkResult.from_times(f"random_read_{count}", times, throughput)


def benchmark_header_only(
    sgy: SegyFile,
    instrumented: InstrumentedFileSystem,
    count: int,
    seed: int,
    warmup: int,
    iterations: int,
    verbose: bool = False,
    network_tracker: NetworkBytesTracker | None = None,
) -> BenchmarkResult:
    """Benchmark header-only reads (smaller payloads)."""
    if verbose:
        print(f"  Benchmarking: header_only_{count}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(sgy.num_traces, size=count, replace=False)

    def read_headers() -> None:
        _ = sgy.header[indices]

    times, throughput = run_benchmark(read_headers, warmup, iterations, instrumented, network_tracker)
    return BenchmarkResult.from_times(f"header_only_{count}", times, throughput)


def benchmark_data_only(
    sgy: SegyFile,
    instrumented: InstrumentedFileSystem,
    count: int,
    seed: int,
    warmup: int,
    iterations: int,
    verbose: bool = False,
    network_tracker: NetworkBytesTracker | None = None,
) -> BenchmarkResult:
    """Benchmark data-only reads (larger payloads per trace)."""
    if verbose:
        print(f"  Benchmarking: data_only_{count}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(sgy.num_traces, size=count, replace=False)

    def read_data() -> None:
        _ = sgy.sample[indices]

    times, throughput = run_benchmark(read_data, warmup, iterations, instrumented, network_tracker)
    return BenchmarkResult.from_times(f"data_only_{count}", times, throughput)


# Worker function for concurrent benchmarks - must be at module level for pickling
def _concurrent_worker(
    worker_id: int,
    path: str,
    storage_options: dict[str, Any],
    traces_per_worker: int,
) -> tuple[float, int]:
    """Worker function for concurrent read benchmarks."""
    settings = SegyFileSettings(storage_options=storage_options)
    sgy = SegyFile(path, settings=settings)
    start = worker_id * traces_per_worker
    t0 = time.perf_counter()
    traces = sgy.trace[start : start + traces_per_worker]
    elapsed = time.perf_counter() - t0
    bytes_read = len(traces.sample.tobytes()) + len(traces.header.tobytes())
    return elapsed, bytes_read


def benchmark_concurrent(
    path: str,
    settings: SegyFileSettings,
    num_workers: int,
    traces_per_worker: int,
    warmup: int,
    iterations: int,
    executor_type: str = "thread",
    verbose: bool = False,
) -> BenchmarkResult:
    """Benchmark concurrent trace reads with multiple workers."""
    operation = f"concurrent_{executor_type}_{num_workers}_workers"
    if verbose:
        print(f"  Benchmarking: {operation}")

    Executor = ThreadPoolExecutor if executor_type == "thread" else ProcessPoolExecutor

    def run_concurrent() -> tuple[float, int]:
        with Executor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _concurrent_worker,
                    i,
                    path,
                    settings.storage_options,
                    traces_per_worker,
                )
                for i in range(num_workers)
            ]
            results = [f.result() for f in futures]
        total_bytes = sum(r[1] for r in results)
        return 0.0, total_bytes  # Time tracked externally

    # Warmup
    for _ in range(warmup):
        run_concurrent()
        gc.collect()

    # Measured runs
    times: list[float] = []
    total_bytes = 0
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        _, bytes_read = run_concurrent()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_bytes = bytes_read  # Same each iteration

    # Calculate throughput
    # Note: In concurrent mode, we only track bytes from the application side,
    # not actual network bytes. Network tracking is not available across processes.
    total_time = sum(times)
    total_bytes_all_iterations = total_bytes * iterations
    throughput_mbps = round((total_bytes_all_iterations / (1024 * 1024)) / total_time, 2) if total_time > 0 else 0.0
    throughput = ThroughputMetrics(
        bytes_requested=total_bytes_all_iterations,
        bytes_transferred=total_bytes_all_iterations,  # Same as requested (no instrumentation)
        network_bytes_recv=0,  # Not tracked in concurrent mode
        network_bytes_sent=0,  # Not tracked in concurrent mode
        throughput_mbps=throughput_mbps,
        throughput_actual_mbps=throughput_mbps,  # Same as logical (no instrumentation)
        throughput_network_mbps=0.0,  # Not tracked in concurrent mode
        prefetch_ratio=1.0,  # Not measured in concurrent mode
        network_overhead=0.0,  # Not measured in concurrent mode
        read_block_calls=0,  # Not tracked in concurrent mode
        cat_ranges_calls=0,
        ranges_requested=0,
    )

    return BenchmarkResult.from_times(operation, times, throughput)


def run_benchmarks(args: argparse.Namespace) -> BenchmarkReport:
    """Run all benchmarks and return a report."""
    print(f"fsspec version: {fsspec.__version__}")
    print(f"Data path: {args.data_path}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Measured iterations: {args.iterations}")
    print(f"Trace count: {args.trace_count}")

    # Check if network tracking is available
    network_tracking_available = NetworkBytesTracker.is_available()
    if network_tracking_available:
        print("Network tracking: ENABLED (psutil available)")
    else:
        print("Network tracking: DISABLED (psutil not installed)")
    print()

    # Parse storage options
    storage_options = json.loads(args.storage_options) if args.storage_options else {}

    # Only use anonymous access for the default public Parihaka dataset
    # For custom S3 paths, use credentials from environment
    if args.data_path == DEFAULT_S3_PATH and not storage_options:
        storage_options = {"anon": True}

    settings = SegyFileSettings(storage_options=storage_options)

    # Initial file open to get file info
    print("Opening file to get metadata...")
    sgy = SegyFile(args.data_path, settings=settings)
    file_info = FileInfo(
        size_bytes=sgy.file_size,
        num_traces=sgy.num_traces,
        samples_per_trace=sgy.samples_per_trace,
        trace_size_bytes=sgy.spec.trace.itemsize,
    )
    print(f"  File size: {file_info.size_bytes / (1024**3):.2f} GiB")
    print(f"  Traces: {file_info.num_traces:,}")
    print(f"  Samples/trace: {file_info.samples_per_trace}")
    print()

    # Create report
    report = BenchmarkReport(
        fsspec_version=fsspec.__version__,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform_info=platform.platform(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_path=args.data_path,
        file_info=file_info,
    )

    # Run benchmarks
    print("Running benchmarks...")

    # 1. File open benchmark
    result = benchmark_file_open(
        args.data_path, settings, args.warmup, args.iterations, args.verbose
    )
    report.results.append(result)
    if args.verbose:
        print(f"    Mean: {result.mean:.3f}s ± {result.std:.3f}s")

    # Create instrumented SegyFile for subsequent benchmarks
    sgy, instrumented = create_instrumented_segy(args.data_path, settings)

    # Create network tracker if available
    network_tracker = NetworkBytesTracker() if network_tracking_available else None

    # 2. Sequential read benchmark
    start_trace = min(500_000, sgy.num_traces - args.trace_count - 1)
    instrumented.metrics.reset()
    if network_tracker:
        network_tracker.reset()
    result = benchmark_sequential_read(
        sgy, instrumented, start_trace, args.trace_count,
        args.warmup, args.iterations, args.verbose, network_tracker
    )
    report.results.append(result)
    if args.verbose:
        print(f"    Mean: {result.mean:.3f}s ± {result.std:.3f}s")
        if result.throughput:
            print(f"    Throughput: {result.throughput.throughput_mbps:.2f} MB/s")
            if result.throughput.network_bytes_recv > 0:
                print(f"    Network overhead: {result.throughput.network_overhead:.3f}x")

    # 3. Random read benchmark
    instrumented.metrics.reset()
    if network_tracker:
        network_tracker.reset()
    result = benchmark_random_read(
        sgy, instrumented, args.trace_count, args.random_seed,
        args.warmup, args.iterations, args.verbose, network_tracker
    )
    report.results.append(result)
    if args.verbose:
        print(f"    Mean: {result.mean:.3f}s ± {result.std:.3f}s")
        if result.throughput:
            print(f"    Throughput: {result.throughput.throughput_mbps:.2f} MB/s")
            print(f"    cat_ranges calls: {result.throughput.cat_ranges_calls}")
            if result.throughput.network_bytes_recv > 0:
                print(f"    Network overhead: {result.throughput.network_overhead:.3f}x")

    # 4. Header-only benchmark
    instrumented.metrics.reset()
    if network_tracker:
        network_tracker.reset()
    result = benchmark_header_only(
        sgy, instrumented, args.trace_count, args.random_seed + 1,
        args.warmup, args.iterations, args.verbose, network_tracker
    )
    report.results.append(result)
    if args.verbose:
        print(f"    Mean: {result.mean:.3f}s ± {result.std:.3f}s")
        if result.throughput and result.throughput.network_bytes_recv > 0:
            print(f"    Network overhead: {result.throughput.network_overhead:.3f}x")

    # 5. Data-only benchmark
    instrumented.metrics.reset()
    if network_tracker:
        network_tracker.reset()
    result = benchmark_data_only(
        sgy, instrumented, args.trace_count, args.random_seed + 2,
        args.warmup, args.iterations, args.verbose, network_tracker
    )
    report.results.append(result)
    if args.verbose:
        print(f"    Mean: {result.mean:.3f}s ± {result.std:.3f}s")
        if result.throughput and result.throughput.network_bytes_recv > 0:
            print(f"    Network overhead: {result.throughput.network_overhead:.3f}x")

    # 6. Concurrent benchmarks (if enabled)
    if args.concurrency_test:
        print("\nRunning concurrency benchmarks...")
        workers_list = [int(w.strip()) for w in args.workers.split(",")]
        traces_per_worker = args.trace_count // max(workers_list)

        for num_workers in workers_list:
            result = benchmark_concurrent(
                args.data_path,
                settings,
                num_workers,
                traces_per_worker,
                args.warmup,
                args.iterations,
                args.executor,
                args.verbose,
            )
            report.results.append(result)
            if args.verbose:
                print(f"    Mean: {result.mean:.3f}s ± {result.std:.3f}s")
                if result.throughput:
                    print(f"    Throughput: {result.throughput.throughput_mbps:.2f} MB/s")

    return report


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance regression benchmarks for fsspec I/O operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default Parihaka dataset (S3)
  python benchmarks/fsspec_regression.py --output results.json

  # Use local copy of dataset
  python benchmarks/fsspec_regression.py -p /data/file.sgy -O results.json

  # Run with concurrent I/O tests
  python benchmarks/fsspec_regression.py --concurrency-test --workers 1,2,4,8

  # Quick test with fewer iterations
  python benchmarks/fsspec_regression.py -n 3 --trace-count 500 -v
""",
    )

    # Data source options
    parser.add_argument(
        "-p", "--data-path",
        default=DEFAULT_S3_PATH,
        help="Path to SEG-Y file (S3, HTTP, or local path)",
    )
    parser.add_argument(
        "-o", "--storage-options",
        default="",
        help='JSON string of fsspec storage options (e.g., \'{"anon": true}\', \'{"default_cache_type": "none"}\')',
    )

    # Benchmark control
    parser.add_argument(
        "-w", "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=5,
        help="Number of measured iterations (default: 5)",
    )
    parser.add_argument(
        "--trace-count",
        type=int,
        default=1000,
        help="Number of traces per read operation (default: 1000)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for reproducible random trace indices (default: 42)",
    )

    # Concurrency testing
    parser.add_argument(
        "--concurrency-test",
        action="store_true",
        help="Enable concurrent I/O benchmarks",
    )
    parser.add_argument(
        "--workers",
        default="1,2,4",
        help="Comma-separated worker counts for concurrency tests (default: 1,2,4)",
    )
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="Executor type for concurrency tests (default: thread)",
    )

    # Output
    parser.add_argument(
        "-O", "--output",
        help="Output file path for JSON results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress during benchmarks",
    )

    args = parser.parse_args()

    try:
        report = run_benchmarks(args)

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"fsspec version: {report.fsspec_version}")
        print(f"Python: {report.python_version}")
        print(f"Platform: {report.platform_info}")
        print()

        for result in report.results:
            print(f"{result.operation}:")
            print(f"  Time: {result.mean:.3f}s ± {result.std:.3f}s (min={result.min:.3f}, max={result.max:.3f})")
            if result.throughput:
                tp = result.throughput
                print(f"  Throughput (logical): {tp.throughput_mbps:.2f} MB/s")
                print(f"  Throughput (actual):  {tp.throughput_actual_mbps:.2f} MB/s")
                if tp.network_bytes_recv > 0:
                    print(f"  Throughput (network): {tp.throughput_network_mbps:.2f} MB/s")
                print(f"  Bytes requested: {tp.bytes_requested:,}")
                print(f"  Bytes transferred: {tp.bytes_transferred:,}")
                if tp.network_bytes_recv > 0:
                    print(f"  Network bytes recv: {tp.network_bytes_recv:,}")
                    overhead_pct = (tp.network_overhead - 1.0) * 100
                    if abs(overhead_pct) >= 1:
                        print(f"  Network overhead: {tp.network_overhead:.3f}x ({overhead_pct:+.1f}% vs requested)")
                    else:
                        print(f"  Network overhead: {tp.network_overhead:.3f}x")
                if tp.prefetch_ratio != 1.0:
                    print(f"  API prefetch ratio: {tp.prefetch_ratio:.3f}x")
                if tp.cat_ranges_calls > 0:
                    print(f"  cat_ranges calls: {tp.cat_ranges_calls}")
                    print(f"  Ranges requested: {tp.ranges_requested}")
            print()

        # Save results
        if args.output:
            report.save(args.output)
            print(f"Results saved to: {args.output}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running benchmarks: {e}")
        raise


if __name__ == "__main__":
    main()
