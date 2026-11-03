# template_assets — agent rules

## Overview

Template sources for the CLI scaffolding command.

## Purpose

Give a new project a valid, minimal GNN spec to start from.

## Directory Identity

Asset directory under `src/gnn/cli/`; consumed by `src/gnn/cli/templates.py`.

## Rules

- Template sources must stay under `template_assets/*.md`; the loader refuses
  paths outside it.
- A new asset is a valid GNN spec — it is what new projects initialize from.
