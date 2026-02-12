# Performance Benchmarks

This directory contains performance regression tests for the segy library,
specifically designed to detect regressions in fsspec I/O operations.

## Overview

The benchmarks measure:

- **File opening** - Spec inference, text/binary header reads via `read_block()`
- **Sequential reads** - Contiguous trace slices (tests range merging optimization)
- **Random reads** - Non-contiguous trace indices (stresses `cat_ranges()`)
- **Header-only reads** - Smaller payloads, more I/O overhead visible
- **Data-only reads** - Larger payloads per trace
- **Concurrent I/O** - Multi-threaded/multi-process reads

Each benchmark captures:
- Timing statistics (mean, std, min, max, median)
- Throughput metrics (bytes read, MB/s)
- I/O operation counts (read_block calls, cat_ranges calls, ranges requested)

## Quick Start

### Basic Benchmark Run

```bash
# Run benchmarks with current environment
nox -s benchmark

# Run with verbose output
nox -s benchmark -- -v

# Run with specific fsspec version
nox -s benchmark -- --fsspec 2026.1.0 --output baseline.json
nox -s benchmark -- --fsspec 2026.2.0 --output current.json
```

### Compare Results

```bash
# Basic comparison
nox -s benchmark-compare -- baseline.json current.json

# With custom threshold (default is 20%)
nox -s benchmark-compare -- baseline.json current.json --threshold 15

# Fail on regression (useful for scripts)
nox -s benchmark-compare -- baseline.json current.json --fail-on-regression
```

## Full Regression Test Workflow

When preparing a release or investigating a reported regression:

```bash
# 1. Create baseline with known-good fsspec version
nox -s benchmark -- --fsspec 2026.1.0 --output baseline.json --concurrency-test -v

# 2. Test with suspected problematic version
nox -s benchmark -- --fsspec 2026.2.0 --output current.json --concurrency-test -v

# 3. Compare results
nox -s benchmark-compare -- baseline.json current.json --threshold 20
```

## CLI Reference

### fsspec_regression.py

```
Usage: python benchmarks/fsspec_regression.py [OPTIONS]

Options:
  -p, --data-path PATH       Path to SEG-Y file (S3, HTTP, or local)
                             Default: Parihaka dataset on S3
  -o, --storage-options JSON JSON string of fsspec storage options
  -w, --warmup N             Number of warmup iterations (default: 2)
  -n, --iterations N         Number of measured iterations (default: 5)
  --trace-count N            Number of traces per operation (default: 1000)
  --random-seed N            Seed for reproducible indices (default: 42)
  --concurrency-test         Enable concurrent I/O benchmarks
  --workers LIST             Comma-separated worker counts (default: 1,2,4)
  --executor TYPE            thread or process (default: thread)
  -O, --output PATH          Output file path for JSON results
  -v, --verbose              Print detailed progress
```

### compare_results.py

```
Usage: python benchmarks/compare_results.py BASELINE CURRENT [OPTIONS]

Arguments:
  BASELINE                   Path to baseline results JSON
  CURRENT                    Path to current results JSON

Options:
  -t, --threshold PERCENT    Regression threshold (default: 20)
  --fail-on-regression       Exit with code 1 if regression detected
  --json PATH                Save comparison results to JSON
```

## Using Custom Data

By default, benchmarks use the Parihaka 3D dataset on S3. You can use a local
copy or any other SEG-Y file:

```bash
# Local file
nox -s benchmark -- --data-path /data/seismic/my_file.sgy

# HTTP endpoint (often more stable than S3)
nox -s benchmark -- --data-path "http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/Taranaiki_Basin/PARIHAKA-3D/Parihaka_PSTM_full_angle.sgy"

# With custom storage options
nox -s benchmark -- --data-path s3://my-bucket/file.sgy --storage-options '{"key": "...", "secret": "..."}'
```

## Understanding Results

### Timing Output

```
sequential_read_1000:
  Time: 0.893s ± 0.005s (min=0.887, max=0.901)
  Throughput: 6.67 MB/s
  Bytes read: 6,244,000
```

### Comparison Output

```
Timing Results:
--------------------------------------------------------------------------------
Operation                      | Baseline        | Current         | Delta      | Status
--------------------------------------------------------------------------------
sequential_read_1000           | 892 ± 23 ms     | 1456 ± 45 ms    | +63.2%     | REGRESSION
random_read_1000               | 2341 ± 89 ms    | 4521 ± 156 ms   | +93.1%     | REGRESSION

Throughput Analysis:
--------------------------------------------------------------------------------
Operation                      | Base MB/s    | Curr MB/s    | Bytes Δ      | Requests Δ
--------------------------------------------------------------------------------
random_read_1000               | 2.50         | 1.32         | +0%          | +340% <<<

Potential Issues Detected:
--------------------------------------------------------------------------------
>>> random_read_1000: 4.4x more cat_ranges calls (12 → 53)
```

### Interpreting Results

- **Timing regression** (>20% slower): Indicates overall performance degradation
- **Increased requests** (Requests Δ): May indicate change in request batching/merging
- **Increased bytes** (Bytes Δ): May indicate change in prefetching behavior
- **Lower throughput**: Combined effect of timing and I/O changes

## Tips for Consistent Results

1. **Use local data when possible** - Eliminates network variability
2. **Run on idle machine** - Minimize background process interference
3. **Use more iterations** - `--iterations 10` for better statistical confidence
4. **Same machine** - Always compare baseline and current on the same hardware
5. **Same Python version** - Ensure consistent Python environment

## Files

- `fsspec_regression.py` - Main benchmark script
- `compare_results.py` - Results comparison utility
- `instrumented_fs.py` - Filesystem wrapper for capturing I/O metrics
- `README.md` - This documentation
