# examples/ — agent rules

## Overview

Worked examples outside the pipeline corpus and the test suite.

## Purpose

Show one integration end to end with the commands that reproduce it.

## Directory Identity

Example tree; each subdirectory carries its own README.

## Rules

- Generated examples stay generated: `examples/rxinfer/asia_model.jl` is emitter
  output pinned by a golden test — change the emitter and regenerate, never the
  committed artifact by hand.
- Do not move example data into `input/gnn_files/`: that corpus is indexed and
  its counts are cited.
