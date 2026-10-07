# Released THRML structural-zero verification

Checked 2026-10-07 against the official
[PyPI release metadata](https://pypi.org/pypi/thrml/json),
[upstream releases](https://github.com/extropic-ai/thrml/releases) and
[inactive categorical padding issue 72](https://github.com/extropic-ai/thrml/issues/72).
The issue was closed on 2026-10-05 with a maintainer statement that it was fixed.
The newest published distribution remains **THRML 0.1.4**, released 2026-08-04.
A closed issue and a fix on upstream main do not identify a released fixed wheel.

The ordinary `thrml-0.1.4-py3-none-any.whl` has SHA-256
`6e2f38cecb562589d230ca063b5fcb5d2a6533201e37bb70c1f2dac4a63a0858`.
Fresh frozen hashed installations outside the GNN checkout reproduced the bug:

| Python | JAX/JAXlib | Numeric policies | Result |
| --- | --- | --- | --- |
| 3.11.15 | 0.9.2 | float32 and float64 | Structural-zero padding introduces a NaN into an unrelated node's logits. |
| 3.12.13 | 0.11.2 | float32 and float64 | The same NaN contaminates the independent conditional distribution. |

The independent two-node witness declares `P(x)=[0,1]` and `P(y)=[0.4,0.6]`.
Stock sampling parameters preserve active `-inf` for the zero in x but produce
a NaN for y's first logit. Thus y does not retain its required independent
finite distribution. Positive controls with `P(x)=[0.2,0.8]` have no NaNs and
10,000 native draws per dtype retain y frequencies within 0.02 of `[0.4,0.6]`.
The corresponding GNN structural-zero model still refuses explicitly.

**S3 is blocked by released upstream compatibility.** GNN keeps its exact
0.1.4 pin and strictly positive A/B/D admission. No clipping, probability repair,
sampler substitution, source rewrite or structural-zero capability is introduced.
See the [THRML scientific contract](../gnn/implementations/thrml.md).

Before enabling structural zeros, identify a new ordinary released wheel and
bind its provider digest, installed source bytes and dependency versions. On
both supported JAX splits, require correct logits, retained exact zeros,
independent probabilities, finite native results, seed/sample replay and the
existing generated-run smoothing/composition witnesses. Then review any bounded
adapter change separately. New coupled/control/continuous/hardware semantics
remain outside this verification workstream.
