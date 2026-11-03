# tests/test_data/runner — agent rules

## Overview

Runner-selection fixture for the test infrastructure.

## Purpose

Pin the runner's reporting for a selection that collects and then skips
everything.

## Directory Identity

Fixture directory; no test code.

## Rules

- Keep the `.txt` suffix: renamed `all_skipped.py`, pytest would collect the fixture.
- Update the fixture and `tests/utils/test_testing_runner_execution.py`
  together.
