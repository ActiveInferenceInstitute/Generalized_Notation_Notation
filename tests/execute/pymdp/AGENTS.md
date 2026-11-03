# tests/execute/pymdp — agent rules

## Overview

PyMDP executor regressions under Step 12.

## Purpose

Pin the recovery behavior and the numeric receipt this framework emits.

## Directory Identity

Test package under `tests/execute/`.

## Rules

- Keep numerical witnesses independent of the implementation under test.
- A receipt-shape change updates the consumer tests in the same PR.
