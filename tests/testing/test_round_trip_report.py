"""Unit tests for ``gnn.testing.round_trip_report``.

Covers the ``RoundTripReportMixin.generate_report`` paths not exercised by
``test_round_trip_support.py``: the 100%-success recommendation, per-result
differences and warnings sections, and semantic-checksum verdict lines.
"""

from __future__ import annotations

from gnn.parsers.common import GNNFormat
from gnn.testing.round_trip_report import RoundTripReportMixin
from gnn.testing.round_trip_results import ComprehensiveTestReport, RoundTripResult


def _report(
    *results: RoundTripResult, critical: str | None = None
) -> ComprehensiveTestReport:
    report = ComprehensiveTestReport(reference_file="ref.md")
    for result in results:
        report.add_result(result)
    if critical is not None:
        report.critical_errors.append(critical)
    return report


class TestSuccessRecommendation:
    def test_all_pass_report_recommends_celebration(self) -> None:
        report = _report(
            RoundTripResult(
                source_format=GNNFormat.MARKDOWN, target_format=GNNFormat.JSON
            )
        )

        text = RoundTripReportMixin().generate_report(report)

        assert "🎉 **All tests passed!**" in text
        assert "⚠️ **Some tests failed.**" not in text


class TestDetailedResultSections:
    def test_differences_are_listed_per_result(self) -> None:
        result = RoundTripResult(
            source_format=GNNFormat.MARKDOWN, target_format=GNNFormat.JSON
        )
        result.add_difference("Model name mismatch: 'a' vs 'b'")

        text = RoundTripReportMixin().generate_report(_report(result))

        assert "- **Differences:**" in text
        assert "  - Model name mismatch: 'a' vs 'b'" in text

    def test_warnings_are_listed_per_result(self) -> None:
        report = _report(
            RoundTripResult(
                source_format=GNNFormat.MARKDOWN, target_format=GNNFormat.JSON
            )
        )
        report.round_trip_results[0].add_warning("cosmetic whitespace")

        text = RoundTripReportMixin().generate_report(report)

        assert "- **Warnings:**" in text
        assert "  - cosmetic whitespace" in text


class TestChecksumVerdictLines:
    def test_matching_checksums_render_pass_mark(self) -> None:
        report = _report(
            RoundTripResult(
                source_format=GNNFormat.MARKDOWN,
                target_format=GNNFormat.JSON,
                checksum_original="abc123",
                checksum_converted="abc123",
            )
        )

        text = RoundTripReportMixin().generate_report(report)

        assert "- **Semantic Checksum:** ✅" in text

    def test_differing_checksums_render_fail_mark(self) -> None:
        report = _report(
            RoundTripResult(
                source_format=GNNFormat.MARKDOWN,
                target_format=GNNFormat.JSON,
                checksum_original="abc123",
                checksum_converted="def456",
            )
        )

        text = RoundTripReportMixin().generate_report(report)

        assert "- **Semantic Checksum:** ❌" in text
        assert "⚠️ **Some tests failed.**" not in text
