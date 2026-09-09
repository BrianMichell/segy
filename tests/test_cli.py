"""Tests for the CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from segy import SegyFactory
from segy.cli.segy import app
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards import get_segy_standard

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def local_segy_path(tmp_path: Path) -> str:
    """Write a tiny local SEG-Y so CLI tests do not depend on public S3."""
    spec = get_segy_standard(SegyStandard.REV1)
    spec.endianness = Endianness.BIG
    spec.trace.data.format = ScalarType.IBM32
    factory = SegyFactory(spec, sample_interval=2000, samples_per_trace=8)

    headers = factory.create_trace_header_template(3)
    samples = factory.create_trace_sample_template(3)
    headers["source_coord_x"] = [111, 222, 333]
    headers["coordinate_scalar"] = [-100, -100, -100]
    samples[:] = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [0.5] * 8,
        [-1.0] * 8,
    ]

    path = tmp_path / "sample.sgy"
    path.write_bytes(
        factory.create_textual_header()
        + factory.create_binary_header()
        + factory.create_traces(headers, samples)
    )
    return str(path)


class TestDump:
    """Test class for CLI's dump options."""

    def test_info_dump(self, local_segy_path: str) -> None:
        """Test generic info dump."""
        result = runner.invoke(app, ["dump", "info", local_segy_path])
        assert result.exit_code == 0
        assert "numTraces" in result.stdout
        assert "fileSize" in result.stdout

    def test_text_dump(self, local_segy_path: str) -> None:
        """Test text header dump."""
        result = runner.invoke(app, ["dump", "text-header", local_segy_path])
        assert result.exit_code == 0
        assert "open-source segy library" in result.stdout

    def test_binary_header_dump(self, local_segy_path: str) -> None:
        """Test binary header dump."""
        result = runner.invoke(app, ["dump", "binary-header", local_segy_path])
        assert result.exit_code == 0
        assert "sample_interval" in result.stdout
        assert "samples_per_trace" in result.stdout

    def test_trace_header_dump(self, local_segy_path: str) -> None:
        """Test trace header dump for one and many traces."""
        single = ["dump", "trace-header", local_segy_path]
        single += ["--index", "0", "--field", "source_coord_x"]
        single_result = runner.invoke(app, single)
        assert single_result.exit_code == 0
        assert "source_coord_x" in single_result.stdout
        assert "111" in single_result.stdout

        many = ["dump", "trace-header", local_segy_path]
        many += ["--index", "0", "--index", "1"]
        many += ["--field", "source_coord_x"]
        many += ["--field", "coordinate_scalar"]
        many_result = runner.invoke(app, many)
        assert many_result.exit_code == 0
        assert "source_coord_x" in many_result.stdout
        assert "coordinate_scalar" in many_result.stdout
        assert "222" in many_result.stdout
        assert "-100" in many_result.stdout

    def test_trace_data_dump(self, local_segy_path: str) -> None:
        """Test trace data dump."""
        args = ["dump", "trace-data", local_segy_path]
        args += ["--index", "0", "--index", "2"]

        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert "1." in result.stdout
        assert "-1." in result.stdout
