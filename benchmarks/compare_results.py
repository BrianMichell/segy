#!/usr/bin/env python
"""Compare two benchmark result files and highlight regressions.

Usage:
    python benchmarks/compare_results.py baseline.json current.json
    python benchmarks/compare_results.py baseline.json current.json --threshold 15
    python benchmarks/compare_results.py baseline.json current.json --fail-on-regression
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComparisonResult:
    """Result of comparing a single operation."""

    operation: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    current_std: float
    delta_percent: float
    is_regression: bool
    # Throughput comparison (optional)
    baseline_throughput: float | None = None
    current_throughput: float | None = None
    baseline_throughput_actual: float | None = None
    current_throughput_actual: float | None = None
    baseline_throughput_network: float | None = None
    current_throughput_network: float | None = None
    baseline_bytes_requested: int | None = None
    current_bytes_requested: int | None = None
    baseline_bytes_transferred: int | None = None
    current_bytes_transferred: int | None = None
    baseline_network_bytes_recv: int | None = None
    current_network_bytes_recv: int | None = None
    baseline_network_overhead: float | None = None
    current_network_overhead: float | None = None
    baseline_prefetch_ratio: float | None = None
    current_prefetch_ratio: float | None = None
    baseline_cat_ranges: int | None = None
    current_cat_ranges: int | None = None
    # Legacy aliases (for backward compatibility with old JSON files)
    baseline_bytes: int | None = None
    current_bytes: int | None = None


def load_results(path: str | Path) -> dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_results(
    baseline: dict[str, Any],
    current: dict[str, Any],
    threshold: float,
) -> list[ComparisonResult]:
    """Compare two benchmark reports.

    Args:
        baseline: Baseline benchmark results
        current: Current benchmark results
        threshold: Percentage threshold for regression detection

    Returns:
        List of comparison results for each operation
    """
    comparisons: list[ComparisonResult] = []

    # Create lookup for baseline results
    baseline_ops = {r["operation"]: r for r in baseline["results"]}
    current_ops = {r["operation"]: r for r in current["results"]}

    # Compare common operations
    for op_name in baseline_ops:
        if op_name not in current_ops:
            continue

        base = baseline_ops[op_name]
        curr = current_ops[op_name]

        delta = ((curr["mean"] - base["mean"]) / base["mean"]) * 100 if base["mean"] > 0 else 0
        is_regression = delta > threshold

        # Extract throughput data if available
        base_tp = base.get("throughput") or {}
        curr_tp = curr.get("throughput") or {}

        # Handle both old format (bytes_read) and new format (bytes_requested/bytes_transferred)
        baseline_bytes_req = base_tp.get("bytes_requested") or base_tp.get("bytes_read")
        current_bytes_req = curr_tp.get("bytes_requested") or curr_tp.get("bytes_read")
        baseline_bytes_xfer = base_tp.get("bytes_transferred") or baseline_bytes_req
        current_bytes_xfer = curr_tp.get("bytes_transferred") or current_bytes_req

        # Network-level metrics (from psutil)
        baseline_net_recv = base_tp.get("network_bytes_recv")
        current_net_recv = curr_tp.get("network_bytes_recv")

        comparisons.append(
            ComparisonResult(
                operation=op_name,
                baseline_mean=base["mean"],
                baseline_std=base["std"],
                current_mean=curr["mean"],
                current_std=curr["std"],
                delta_percent=round(delta, 1),
                is_regression=is_regression,
                baseline_throughput=base_tp.get("throughput_mbps"),
                current_throughput=curr_tp.get("throughput_mbps"),
                baseline_throughput_actual=base_tp.get("throughput_actual_mbps"),
                current_throughput_actual=curr_tp.get("throughput_actual_mbps"),
                baseline_throughput_network=base_tp.get("throughput_network_mbps"),
                current_throughput_network=curr_tp.get("throughput_network_mbps"),
                baseline_bytes_requested=baseline_bytes_req,
                current_bytes_requested=current_bytes_req,
                baseline_bytes_transferred=baseline_bytes_xfer,
                current_bytes_transferred=current_bytes_xfer,
                baseline_network_bytes_recv=baseline_net_recv,
                current_network_bytes_recv=current_net_recv,
                baseline_network_overhead=base_tp.get("network_overhead"),
                current_network_overhead=curr_tp.get("network_overhead"),
                baseline_prefetch_ratio=base_tp.get("prefetch_ratio"),
                current_prefetch_ratio=curr_tp.get("prefetch_ratio"),
                baseline_cat_ranges=base_tp.get("cat_ranges_calls"),
                current_cat_ranges=curr_tp.get("cat_ranges_calls"),
                # Legacy aliases for backward compatibility
                baseline_bytes=baseline_bytes_req,
                current_bytes=current_bytes_req,
            )
        )

    return comparisons


def format_time(mean: float, std: float) -> str:
    """Format time as mean ± std in appropriate units."""
    if mean >= 1.0:
        return f"{mean * 1000:.0f} ± {std * 1000:.0f} ms"
    else:
        return f"{mean * 1000:.1f} ± {std * 1000:.1f} ms"


def print_comparison(
    baseline: dict[str, Any],
    current: dict[str, Any],
    comparisons: list[ComparisonResult],
    threshold: float,
) -> bool:
    """Print comparison results in a formatted table.

    Returns:
        True if any regressions were found
    """
    print()
    print("=" * 80)
    print("fsspec Benchmark Comparison")
    print("=" * 80)
    print(f"Baseline: fsspec {baseline['fsspec_version']} | Current: fsspec {current['fsspec_version']}")
    print(f"Regression threshold: {threshold}%")
    print()

    # Timing results table
    print("Timing Results:")
    print("-" * 80)
    header = f"{'Operation':<30} | {'Baseline':<15} | {'Current':<15} | {'Delta':<10} | Status"
    print(header)
    print("-" * 80)

    has_regression = False
    for comp in comparisons:
        status = "REGRESSION" if comp.is_regression else "OK"
        if comp.is_regression:
            has_regression = True
            status = f"\033[91m{status}\033[0m"  # Red

        delta_str = f"{comp.delta_percent:+.1f}%"
        if comp.delta_percent > threshold:
            delta_str = f"\033[91m{delta_str}\033[0m"  # Red
        elif comp.delta_percent < -5:  # Improvement
            delta_str = f"\033[92m{delta_str}\033[0m"  # Green

        print(
            f"{comp.operation:<30} | "
            f"{format_time(comp.baseline_mean, comp.baseline_std):<15} | "
            f"{format_time(comp.current_mean, comp.current_std):<15} | "
            f"{delta_str:<10} | {status}"
        )

    # Throughput analysis (if available)
    has_throughput = any(c.baseline_throughput is not None for c in comparisons)
    if has_throughput:
        print()
        print("Throughput Analysis (Logical):")
        print("-" * 80)
        header = f"{'Operation':<30} | {'Base MB/s':<12} | {'Curr MB/s':<12} | {'Bytes Δ':<12} | Requests Δ"
        print(header)
        print("-" * 80)

        for comp in comparisons:
            if comp.baseline_throughput is None:
                continue

            bytes_delta = ""
            if comp.baseline_bytes and comp.current_bytes:
                delta = ((comp.current_bytes - comp.baseline_bytes) / comp.baseline_bytes) * 100
                bytes_delta = f"{delta:+.0f}%"
                if abs(delta) > 10:
                    bytes_delta = f"\033[93m{bytes_delta}\033[0m"  # Yellow warning

            requests_delta = ""
            if comp.baseline_cat_ranges and comp.current_cat_ranges:
                if comp.baseline_cat_ranges > 0:
                    delta = ((comp.current_cat_ranges - comp.baseline_cat_ranges) / comp.baseline_cat_ranges) * 100
                    requests_delta = f"{delta:+.0f}%"
                    if delta > 50:
                        requests_delta = f"\033[91m{requests_delta} <<<\033[0m"  # Red warning

            print(
                f"{comp.operation:<30} | "
                f"{comp.baseline_throughput or 0:<12.2f} | "
                f"{comp.current_throughput or 0:<12.2f} | "
                f"{bytes_delta:<12} | {requests_delta}"
            )

        # Prefetch analysis (if available)
        has_prefetch = any(c.baseline_prefetch_ratio is not None for c in comparisons)
        if has_prefetch:
            print()
            print("API Prefetch Analysis:")
            print("-" * 80)
            header = f"{'Operation':<30} | {'Base Ratio':<12} | {'Curr Ratio':<12} | {'Base Actual':<12} | Curr Actual"
            print(header)
            print("-" * 80)

            for comp in comparisons:
                if comp.baseline_prefetch_ratio is None and comp.current_prefetch_ratio is None:
                    continue

                base_ratio = comp.baseline_prefetch_ratio or 1.0
                curr_ratio = comp.current_prefetch_ratio or 1.0

                # Format prefetch ratio with warning colors
                base_ratio_str = f"{base_ratio:.3f}x"
                curr_ratio_str = f"{curr_ratio:.3f}x"

                # Highlight significant prefetch differences
                if abs(curr_ratio - base_ratio) > 0.1:
                    if curr_ratio > base_ratio:
                        curr_ratio_str = f"\033[93m{curr_ratio_str}\033[0m"  # Yellow - more prefetching
                    else:
                        curr_ratio_str = f"\033[92m{curr_ratio_str}\033[0m"  # Green - less prefetching

                # Format actual throughput
                base_actual = f"{comp.baseline_throughput_actual or 0:.2f} MB/s" if comp.baseline_throughput_actual else "N/A"
                curr_actual = f"{comp.current_throughput_actual or 0:.2f} MB/s" if comp.current_throughput_actual else "N/A"

                print(
                    f"{comp.operation:<30} | "
                    f"{base_ratio_str:<12} | "
                    f"{curr_ratio_str:<12} | "
                    f"{base_actual:<12} | {curr_actual}"
                )

        # Network overhead analysis (if available - from psutil)
        has_network = any(c.baseline_network_overhead is not None and c.baseline_network_overhead > 0 for c in comparisons)
        if has_network:
            print()
            print("Network Overhead Analysis (OS-level, via psutil):")
            print("-" * 80)
            header = f"{'Operation':<30} | {'Base Overhead':<13} | {'Curr Overhead':<13} | {'Base Net MB/s':<13} | Curr Net MB/s"
            print(header)
            print("-" * 80)

            for comp in comparisons:
                if comp.baseline_network_overhead is None or comp.baseline_network_overhead == 0:
                    continue

                base_overhead = comp.baseline_network_overhead or 1.0
                curr_overhead = comp.current_network_overhead or 1.0

                # Format overhead with warning colors
                base_overhead_str = f"{base_overhead:.3f}x"
                curr_overhead_str = f"{curr_overhead:.3f}x"

                # Highlight significant overhead differences
                if curr_overhead > 0 and base_overhead > 0:
                    overhead_change = (curr_overhead - base_overhead) / base_overhead * 100
                    if abs(overhead_change) > 10:
                        if overhead_change > 0:
                            curr_overhead_str = f"\033[91m{curr_overhead_str} (+{overhead_change:.0f}%)\033[0m"  # Red - more overhead
                        else:
                            curr_overhead_str = f"\033[92m{curr_overhead_str} ({overhead_change:.0f}%)\033[0m"  # Green - less overhead

                # Format network throughput
                base_net = f"{comp.baseline_throughput_network:.2f}" if comp.baseline_throughput_network else "N/A"
                curr_net = f"{comp.current_throughput_network:.2f}" if comp.current_throughput_network else "N/A"

                print(
                    f"{comp.operation:<30} | "
                    f"{base_overhead_str:<13} | "
                    f"{curr_overhead_str:<13} | "
                    f"{base_net:<13} | {curr_net}"
                )

    # Potential insights
    print()
    anomalies = []
    for comp in comparisons:
        if comp.baseline_cat_ranges and comp.current_cat_ranges:
            if comp.current_cat_ranges > comp.baseline_cat_ranges * 2:
                ratio = comp.current_cat_ranges / comp.baseline_cat_ranges
                anomalies.append(
                    f">>> {comp.operation}: {ratio:.1f}x more cat_ranges calls "
                    f"({comp.baseline_cat_ranges} → {comp.current_cat_ranges})"
                )

        if comp.baseline_bytes and comp.current_bytes:
            if comp.current_bytes > comp.baseline_bytes * 1.1:
                ratio = comp.current_bytes / comp.baseline_bytes
                anomalies.append(
                    f">>> {comp.operation}: {ratio:.1f}x more bytes requested "
                    f"({comp.baseline_bytes:,} → {comp.current_bytes:,})"
                )

        # Detect prefetch ratio changes
        if comp.baseline_prefetch_ratio and comp.current_prefetch_ratio:
            if abs(comp.current_prefetch_ratio - comp.baseline_prefetch_ratio) > 0.2:
                if comp.current_prefetch_ratio > comp.baseline_prefetch_ratio:
                    anomalies.append(
                        f">>> {comp.operation}: Prefetch ratio increased "
                        f"({comp.baseline_prefetch_ratio:.3f}x → {comp.current_prefetch_ratio:.3f}x) "
                        "- more data being transferred than requested"
                    )
                else:
                    anomalies.append(
                        f">>> {comp.operation}: Prefetch ratio decreased "
                        f"({comp.baseline_prefetch_ratio:.3f}x → {comp.current_prefetch_ratio:.3f}x) "
                        "- less over-reading (could be cache hit or config change)"
                    )

        # Detect significant bytes transferred changes (actual network traffic)
        if comp.baseline_bytes_transferred and comp.current_bytes_transferred:
            if comp.current_bytes_transferred > comp.baseline_bytes_transferred * 1.5:
                ratio = comp.current_bytes_transferred / comp.baseline_bytes_transferred
                anomalies.append(
                    f">>> {comp.operation}: {ratio:.1f}x more bytes transferred over network "
                    f"({comp.baseline_bytes_transferred:,} → {comp.current_bytes_transferred:,})"
                )

        # Detect network overhead changes (from psutil - OS level)
        if comp.baseline_network_overhead and comp.current_network_overhead:
            if comp.baseline_network_overhead > 0 and comp.current_network_overhead > 0:
                overhead_change = (comp.current_network_overhead - comp.baseline_network_overhead) / comp.baseline_network_overhead
                if abs(overhead_change) > 0.15:  # >15% change in network overhead
                    if overhead_change > 0:
                        anomalies.append(
                            f">>> {comp.operation}: Network overhead increased by {overhead_change*100:.0f}% "
                            f"({comp.baseline_network_overhead:.3f}x → {comp.current_network_overhead:.3f}x) "
                            "- MORE prefetching at OS level"
                        )
                    else:
                        anomalies.append(
                            f">>> {comp.operation}: Network overhead decreased by {abs(overhead_change)*100:.0f}% "
                            f"({comp.baseline_network_overhead:.3f}x → {comp.current_network_overhead:.3f}x) "
                            "- LESS prefetching at OS level"
                        )

    if anomalies:
        print("Potential Issues Detected:")
        print("-" * 80)
        for anomaly in anomalies:
            print(f"\033[93m{anomaly}\033[0m")
        print()
        print("Possible causes: Change in merge_offset_ranges batching, prefetch behavior,")
        print("cache configuration, or request coalescing logic in fsspec.")

    print()
    print("=" * 80)

    if has_regression:
        print(f"\033[91mREGRESSION DETECTED: One or more operations exceeded {threshold}% threshold\033[0m")
    else:
        print(f"\033[92mPASS: No regressions detected (threshold: {threshold}%)\033[0m")

    print("=" * 80)
    print()

    return has_regression


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result files and highlight regressions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python benchmarks/compare_results.py baseline.json current.json

  # Custom threshold
  python benchmarks/compare_results.py baseline.json current.json --threshold 15

  # Fail on regression (for scripts/CI)
  python benchmarks/compare_results.py baseline.json current.json --fail-on-regression
""",
    )

    parser.add_argument(
        "baseline",
        help="Path to baseline results JSON file",
    )
    parser.add_argument(
        "current",
        help="Path to current results JSON file",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=20.0,
        help="Percentage threshold for regression detection (default: 20)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if regression detected",
    )
    parser.add_argument(
        "--json",
        help="Output comparison results to JSON file",
    )

    args = parser.parse_args()

    try:
        baseline = load_results(args.baseline)
        current = load_results(args.current)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

    comparisons = compare_results(baseline, current, args.threshold)

    if not comparisons:
        print("No common operations found to compare")
        sys.exit(1)

    has_regression = print_comparison(baseline, current, comparisons, args.threshold)

    # Save JSON output if requested
    if args.json:
        output = {
            "baseline_version": baseline["fsspec_version"],
            "current_version": current["fsspec_version"],
            "threshold": args.threshold,
            "has_regression": has_regression,
            "comparisons": [
                {
                    "operation": c.operation,
                    "baseline_mean": c.baseline_mean,
                    "current_mean": c.current_mean,
                    "delta_percent": c.delta_percent,
                    "is_regression": c.is_regression,
                    "baseline_throughput": c.baseline_throughput,
                    "current_throughput": c.current_throughput,
                    "baseline_throughput_actual": c.baseline_throughput_actual,
                    "current_throughput_actual": c.current_throughput_actual,
                    "baseline_throughput_network": c.baseline_throughput_network,
                    "current_throughput_network": c.current_throughput_network,
                    "baseline_prefetch_ratio": c.baseline_prefetch_ratio,
                    "current_prefetch_ratio": c.current_prefetch_ratio,
                    "baseline_network_overhead": c.baseline_network_overhead,
                    "current_network_overhead": c.current_network_overhead,
                    "baseline_bytes_requested": c.baseline_bytes_requested,
                    "current_bytes_requested": c.current_bytes_requested,
                    "baseline_bytes_transferred": c.baseline_bytes_transferred,
                    "current_bytes_transferred": c.current_bytes_transferred,
                    "baseline_network_bytes_recv": c.baseline_network_bytes_recv,
                    "current_network_bytes_recv": c.current_network_bytes_recv,
                }
                for c in comparisons
            ],
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Comparison saved to: {args.json}")

    if args.fail_on_regression and has_regression:
        sys.exit(1)


if __name__ == "__main__":
    main()
