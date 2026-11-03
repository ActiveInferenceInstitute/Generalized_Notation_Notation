# tests/export/cascadia — agent rules

## Overview

GEO-INFER interchange fixture for the export tests.

## Purpose

Pin the exported artifacts for one reviewed spec.

## Directory Identity

Fixture directory; no test code.

## Rules

- The spec is the input; the two JSON files are expected outputs. Regenerating
  them requires the paired GEO-INFER interchange evidence, not just a local run.
- Never edit a fixture JSON by hand to match new exporter output.
