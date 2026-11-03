# input/gnn_files/

The exemplar corpus the pipeline runs: each entry is a runnable Active Inference
generative model spec, grouped by task shape (`discrete/`, `continuous/`,
`multiagent/`, `hierarchical/`, …).

[`INDEX.md`](INDEX.md) is the cold-start index and owns the measured counts: it
lists every spec, its task folder and the commands that run it. Read it before
adding, renaming or moving a spec. Backend admission depends on each model's
declared contract, not on the folder it sits in.
