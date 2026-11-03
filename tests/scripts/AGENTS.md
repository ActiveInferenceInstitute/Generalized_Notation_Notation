# tests/scripts — agent rules

## Overview

Gate-script contracts. These tests fail when a `scripts/` gate's rule changes
without its contract changing with it.

## Purpose

Keep the audits honest: a gate that stops failing on a real violation must be
caught here, with fixtures.

## Directory Identity

Test package over `scripts/`; fixtures build temporary trees.

## Rules

- Change a gate's rule and this contract in the same PR.
- Prefer `tmp_path` fixtures that exercise the violation path; a test that only
  re-encodes the script's own output table proves nothing.
- Keep the live-tree assertions: the real repository must stay compliant.
