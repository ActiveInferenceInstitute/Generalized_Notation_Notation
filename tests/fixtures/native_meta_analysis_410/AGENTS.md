# tests/fixtures/native_meta_analysis_410 — agent rules

## Overview

Exact-source fixture set for the 4.1.0 native consumers.

## Purpose

`provenance.json` owns these files' identity (`scope`, `fixtures`,
`signed_producer`, `static_producer`).

## Directory Identity

Data directory; consumed by the native-artifact consumer tests.

## Rules

- Never edit the results in place, and never rewrite `provenance.json` to match
  a new run.
- A new epoch gets its own directory with its own provenance record.
