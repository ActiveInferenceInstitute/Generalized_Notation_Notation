# GUI 1 — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Purpose

Form-based GNN model constructor GUI served as a Gradio web application.

## Features

- Model structure editor (states, observations, actions)
- Component management with live state-space validation
- Synchronized plaintext GNN markdown editing
- Headless artifact generation when Gradio is unavailable

## Technology

- `gradio` (from the `gui` extra in `pyproject.toml`)
- Cross-platform (Windows, macOS, Linux)

## Input/Output

- Input: Existing GNN files or blank canvas
- Output: GNN model files in markdown format (`constructed_model_gui1.md` by default)

## State declaration insertion

The public `add_state_space_entry` helper inserts into an existing canonical
`## StateSpaceBlock` before its next level-two section. Existing editor heading
aliases, including the existing spaced `## State Space`, remain supported by the
editor; these aliases do not extend the native GNN readers' section grammar.
Insertion retains all existing text and line endings outside the new declaration.
It preserves physical duplicate declarations and the existing linear delimiter
parser and 8,388,608-character input refusal.

Headless export saves the supplied edited Markdown with a headless artifact
marker. It does not invoke Gradio callbacks or validate scientific equivalence.
