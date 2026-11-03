# tests/types — agent rules

## Overview

Dataclass contract for the shared domain model in `src/gnn/types/`.

## Purpose

Those types are public surface for every consumer; changes here ripple.

## Directory Identity

Test package over `src/gnn/types/`.

## Rules

- Keep constructor defaults, field order and serialization stable; a deliberate
  change updates this contract in the same PR.
- Do not add validation here that belongs to `gnn/schema_validator/`.
