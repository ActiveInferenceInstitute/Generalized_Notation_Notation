# tests/schema_validator — agent rules

## Overview

Regression suite for the maintained validation surface in
`src/gnn/schema_validator/`.

## Purpose

Keep the validation ladder and its diagnostics pinned: level resolution,
parser failures and cross-format consistency.

## Directory Identity

Test package; cases are named for the behavior they pin.

## Rules

- The validation ladder (BASIC, STANDARD, STRICT, RESEARCH, ROUND_TRIP) is
  normative: assert the level, not just a boolean.
- A new validator diagnostic needs a case here in the same PR.
- Do not relax an assertion to accept a degradation the validator should
  reject; the degradation tests pin exactly that boundary.
