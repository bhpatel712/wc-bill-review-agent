# Model framing decision (WK2.9)

**Choice: multi-label**, implemented as four independent binary
classifiers (one per violation type: `unbundling`, `fragmented_billing`,
`causality_mismatch`, `upcoding`), via scikit-learn's
`MultiOutputClassifier`. A single "any flag" binary decision is derived
from the four (`1 - prod(1 - p_i)` — the probability at least one fires)
rather than trained separately, so the binary view is always consistent
with the multi-label view.

## Why multi-label over plain binary

A reviewer who opens a flagged bill needs to know *what to check*, not
just *that something's off*. "Flagged" alone sends them back through the
whole bill; "flagged: fragmented_billing, units on 97140 exceed the MUE
cap" sends them straight to the line that matters. The four violation
types in this dataset are also genuinely different review workflows —
unbundling is a modifier/coding question, causality_mismatch is a
clinical-relevance question, upcoding is a documentation-audit question —
so collapsing them into one flag throws away the information that
determines who should review the bill and how.

## Why four binary classifiers over one 5-way multiclass model

In this first-pass synthetic corpus every anomalous bill has *exactly
one* violation type by construction (see `DATA_CARD.md`), which would
make a single 5-class softmax classifier (none / unbundling /
fragmented_billing / causality_mismatch / upcoding) a valid — and
slightly simpler — choice today. Four independent binary classifiers
were chosen anyway because real bills are not guaranteed to have at most
one problem: a bill can plausibly both unbundle a procedure pair *and*
upcode the E/M visit. A multiclass model can't represent that; four
binary classifiers can predict all four labels simultaneously with no
architecture change, so the model doesn't need to be redesigned the day
multi-violation bills are added to the generator (noted as a stated,
easy extension in `DATA_CARD.md`). The cost is small: each of the four
sub-problems is a clean, well-separated binary split with strong
engineered features behind it (see `features.py` / WK2.10 evaluation),
so training four classifiers instead of one is not adding meaningful
complexity or overfitting risk here.

## What this changes downstream

- `train_baseline.py` / `train_xgboost.py` both fit
  `MultiOutputClassifier` and save one model artifact that scores all
  four labels at once.
- `evaluate.py` reports precision/recall per label *and* for the derived
  "any flag" view, so both the operational (queue-volume) and clinical
  (what's actually wrong) questions are answered.
- The `/review-bill` API (WK6) returns the full label vector with
  per-label confidence, not a single score — the RAG/fine-tuned
  explanation agent (WK4-5) explains whichever labels fired.
