# tests/fixtures — agent rules

## Overview

Committed fixture data shared by more than one test package.

## Purpose

Keep the exact bytes that receipts and assertions cite.

## Directory Identity

Data directory; test code belongs in the owning `tests/<module>/` package.

## Rules

- Fixtures are immutable inputs: a new data set gets a new directory, never an
  in-place edit.
- Do not put generated run output here; pipeline output belongs under `output/`.
