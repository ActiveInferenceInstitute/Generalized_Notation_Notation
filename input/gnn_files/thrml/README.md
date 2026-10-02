# THRML exemplars

[categorical_smoothing.md](categorical_smoothing.md) is a positive-support
categorical model with asymmetric transitions and unequal state/observation
dimensions. Select the experimental `thrml` backend explicitly. The model
supports genuine released THRML categorical factor/block Gibbs inference;
it makes no action-control, continuous-state, exact-inference or hardware claim.

For a fixed-data witness use backend options `observations: [0, 1, 0]` and
`transition_actions: [1, 0]`. The options condition the entire three-step
observation sequence and associate each action with the following state edge.
The folder README is documentation and is excluded from runnable selection.
