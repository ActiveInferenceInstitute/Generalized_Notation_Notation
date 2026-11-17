#!/usr/bin/env python3
"""
Availability flags for the GNN round-trip test suite.

Extracted from the former ``testing.test_round_trip`` monolith. The parsers,
serializers and schema validator are in-package dependencies of ``gnn``, so
both flags are constant ``True``; they remain so callers (and tests that
monkeypatch them) can keep gating the cross-format path. Import serializers
and parser types from their canonical ``gnn.parsers`` / ``gnn.schema_validator``
modules, not from here.
"""

GNN_AVAILABLE = True
CROSS_FORMAT_AVAILABLE = True
