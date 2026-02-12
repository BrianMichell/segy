"""Instrumented filesystem wrapper for capturing I/O metrics."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


@dataclass
class IOMetrics:
    """Container for I/O operation metrics.

    Tracks both logical bytes (what the application requested) and actual bytes
    (what was returned by the filesystem). The difference between these values
    can indicate prefetching or caching behavior.

    Attributes:
        bytes_requested: Total bytes requested by the application (logical bytes).
        bytes_transferred: Total bytes actually returned (may differ due to
            prefetching, block alignment, or caching).
        read_block_calls: Number of read_block() calls made.
        cat_ranges_calls: Number of cat_ranges() calls made.
        ranges_requested: Total number of individual ranges requested.
        info_calls: Number of info() calls made.
    """

    bytes_requested: int = 0
    bytes_transferred: int = 0
    read_block_calls: int = 0
    cat_ranges_calls: int = 0
    ranges_requested: int = 0
    info_calls: int = 0

    @property
    def bytes_read(self) -> int:
        """Alias for bytes_requested for backward compatibility."""
        return self.bytes_requested

    @property
    def prefetch_ratio(self) -> float:
        """Ratio of bytes transferred to bytes requested.

        A value > 1.0 indicates prefetching or over-reading.
        A value of 1.0 indicates exact reads with no prefetching.
        A value < 1.0 would indicate caching (some data served from cache).
        """
        if self.bytes_requested == 0:
            return 1.0
        return self.bytes_transferred / self.bytes_requested

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.bytes_requested = 0
        self.bytes_transferred = 0
        self.read_block_calls = 0
        self.cat_ranges_calls = 0
        self.ranges_requested = 0
        self.info_calls = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "bytes_requested": self.bytes_requested,
            "bytes_transferred": self.bytes_transferred,
            "prefetch_ratio": round(self.prefetch_ratio, 3),
            "read_block_calls": self.read_block_calls,
            "cat_ranges_calls": self.cat_ranges_calls,
            "ranges_requested": self.ranges_requested,
            "info_calls": self.info_calls,
        }


@dataclass
class InstrumentedFileSystem:
    """Wraps an fsspec filesystem to capture I/O metrics.

    This wrapper intercepts calls to read_block, cat_ranges, and info
    to track bytes transferred and number of operations. Useful for
    identifying changes in prefetching, batching, or request patterns
    across fsspec versions.

    The wrapper tracks both:
    - bytes_requested: What the application asked for (logical bytes)
    - bytes_transferred: What was actually returned (actual bytes)

    The difference between these values can reveal prefetching behavior.
    A prefetch_ratio > 1.0 indicates the filesystem returned more data
    than requested (common with block-aligned caching).

    Example:
        >>> from fsspec.core import url_to_fs
        >>> fs, path = url_to_fs("s3://bucket/file.sgy", anon=True)
        >>> instrumented = InstrumentedFileSystem(fs)
        >>> # Use instrumented.read_block, instrumented.cat_ranges, etc.
        >>> print(instrumented.metrics.bytes_requested)
        >>> print(instrumented.metrics.bytes_transferred)
        >>> print(instrumented.metrics.prefetch_ratio)
    """

    _fs: AbstractFileSystem
    metrics: IOMetrics = field(default_factory=IOMetrics)

    def read_block(
        self,
        fn: str,
        offset: int,
        length: int,
        delimiter: bytes | None = None,
    ) -> bytes:
        """Read a block of bytes from a file, tracking metrics.

        Tracks both requested bytes (length parameter) and actual bytes
        returned (len of result).
        """
        self.metrics.read_block_calls += 1
        self.metrics.bytes_requested += length

        result = self._fs.read_block(fn, offset, length, delimiter)

        # Track actual bytes transferred
        self.metrics.bytes_transferred += len(result)

        return result

    def cat_ranges(
        self,
        paths: list[str],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> list[bytes]:
        """Read byte ranges from files, tracking metrics.

        Tracks both requested bytes (sum of ranges) and actual bytes
        returned (sum of result lengths). These may differ due to
        prefetching, block alignment, or caching.
        """
        self.metrics.cat_ranges_calls += 1
        self.metrics.ranges_requested += len(paths)

        # Calculate total bytes requested (logical bytes)
        total_requested = sum(end - start for start, end in zip(starts, ends))
        self.metrics.bytes_requested += total_requested

        # Execute the actual read
        results = self._fs.cat_ranges(paths, starts, ends, **kwargs)

        # Track actual bytes transferred
        total_transferred = sum(len(chunk) for chunk in results)
        self.metrics.bytes_transferred += total_transferred

        return results

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Get file info, tracking calls."""
        self.metrics.info_calls += 1
        return self._fs.info(path, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the wrapped filesystem."""
        return getattr(self._fs, name)


@dataclass
class NetworkBytesTracker:
    """Track actual network bytes using OS-level counters via psutil.

    This provides ground-truth network I/O measurements that capture
    all bytes transferred over the network, including prefetching by
    fsspec/s3fs/botocore that is invisible at the API level.

    Unlike IOMetrics which tracks what the application requests, this
    tracks what actually goes over the wire by reading OS network counters.

    Attributes:
        bytes_sent: Total bytes sent over the network.
        bytes_recv: Total bytes received from the network.

    Example:
        >>> tracker = NetworkBytesTracker()
        >>> tracker.start()
        >>> # ... perform network operations ...
        >>> tracker.stop()
        >>> print(f"Bytes received: {tracker.bytes_recv}")

    Note:
        This requires the 'psutil' package and measures ALL network traffic
        on the interface, so benchmarks should run in isolation for accurate
        measurements.
    """

    bytes_sent: int = 0
    bytes_recv: int = 0
    _start_bytes_sent: int = field(default=0, repr=False)
    _start_bytes_recv: int = field(default=0, repr=False)
    _is_tracking: bool = field(default=False, repr=False)

    def start(self) -> None:
        """Start tracking - record current network counters.

        Call this before the operation you want to measure.
        """
        try:
            import psutil
        except ImportError as e:
            msg = "psutil package required for network byte tracking"
            raise ImportError(msg) from e

        counters = psutil.net_io_counters()
        self._start_bytes_sent = counters.bytes_sent
        self._start_bytes_recv = counters.bytes_recv
        self._is_tracking = True

    def stop(self) -> None:
        """Stop tracking - calculate delta from start.

        Call this after the operation completes. The delta is added
        to the accumulated totals (bytes_sent, bytes_recv).
        """
        if not self._is_tracking:
            return

        try:
            import psutil
        except ImportError:
            return

        counters = psutil.net_io_counters()
        self.bytes_sent += counters.bytes_sent - self._start_bytes_sent
        self.bytes_recv += counters.bytes_recv - self._start_bytes_recv
        self._is_tracking = False

    def reset(self) -> None:
        """Reset accumulated bytes to zero."""
        self.bytes_sent = 0
        self.bytes_recv = 0
        self._is_tracking = False

    @property
    def network_overhead(self) -> float:
        """Calculate network overhead ratio given bytes_requested.

        This is computed externally since we don't track requested bytes here.
        """
        return 0.0  # Computed in ThroughputMetrics

    def to_dict(self) -> dict[str, int]:
        """Convert metrics to dictionary."""
        return {
            "bytes_sent": self.bytes_sent,
            "bytes_recv": self.bytes_recv,
        }

    @staticmethod
    def is_available() -> bool:
        """Check if psutil is available for network tracking."""
        try:
            import psutil  # noqa: F401
            return True
        except ImportError:
            return False


@dataclass
class HTTPMetrics:
    """Container for HTTP-level network metrics.

    This provides deeper network-level tracking than IOMetrics by capturing
    actual HTTP requests and response sizes. Useful for understanding the
    true network cost of operations, especially when fsspec or the underlying
    HTTP client performs additional requests (retries, redirects, etc.).

    Attributes:
        http_requests: Total number of HTTP requests made.
        http_bytes_sent: Total bytes sent in HTTP request bodies.
        http_bytes_received: Total bytes received in HTTP response bodies.
        http_errors: Number of HTTP errors encountered.
    """

    http_requests: int = 0
    http_bytes_sent: int = 0
    http_bytes_received: int = 0
    http_errors: int = 0

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.http_requests = 0
        self.http_bytes_sent = 0
        self.http_bytes_received = 0
        self.http_errors = 0

    def to_dict(self) -> dict[str, int]:
        """Convert metrics to dictionary."""
        return {
            "http_requests": self.http_requests,
            "http_bytes_sent": self.http_bytes_sent,
            "http_bytes_received": self.http_bytes_received,
            "http_errors": self.http_errors,
        }


def create_instrumented_session(
    metrics: HTTPMetrics,
) -> Any:
    """Create a requests Session with HTTP-level instrumentation.

    This wraps a requests.Session to capture HTTP-level metrics including
    actual bytes sent and received over the network. This is useful for
    understanding true network costs beyond what fsspec's cat_ranges reports.

    Args:
        metrics: HTTPMetrics instance to record metrics into.

    Returns:
        An instrumented requests.Session object.

    Example:
        >>> metrics = HTTPMetrics()
        >>> session = create_instrumented_session(metrics)
        >>> # Use session for HTTP requests
        >>> response = session.get("https://example.com/data")
        >>> print(f"Bytes received: {metrics.http_bytes_received}")

    Note:
        This requires the 'requests' package to be installed.
        For S3/aiobotocore, a different approach is needed.
    """
    try:
        import requests
        from requests.adapters import HTTPAdapter
    except ImportError as e:
        msg = "requests package required for HTTP instrumentation"
        raise ImportError(msg) from e

    class InstrumentedHTTPAdapter(HTTPAdapter):
        """HTTP adapter that tracks request/response metrics."""

        def send(
            self,
            request: requests.PreparedRequest,
            *args: Any,
            **kwargs: Any,
        ) -> requests.Response:
            """Send request and track metrics."""
            metrics.http_requests += 1

            # Track bytes sent (request body)
            if request.body:
                if isinstance(request.body, bytes):
                    metrics.http_bytes_sent += len(request.body)
                elif isinstance(request.body, str):
                    metrics.http_bytes_sent += len(request.body.encode())

            try:
                response = super().send(request, *args, **kwargs)

                # Track bytes received from Content-Length or actual content
                content_length = response.headers.get("Content-Length")
                if content_length:
                    metrics.http_bytes_received += int(content_length)
                elif response.content:
                    metrics.http_bytes_received += len(response.content)

                if not response.ok:
                    metrics.http_errors += 1

                return response
            except Exception:
                metrics.http_errors += 1
                raise

    session = requests.Session()
    adapter = InstrumentedHTTPAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session
