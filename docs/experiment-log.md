# Experiment log

## 2026-10-16 — Initial pilot

Ran all three configurations with seeds 7, 19, and 42 for 600 optimizer steps each. The raw metrics
and reports are retained in `results/pilot/`. All models improved next-byte prediction over random
initialization. The 32-context model had the lowest matched-context test perplexity, but the
128-context seed-7 demo still produced many malformed words at temperature 0.8.

## 2026-10-16 — Longer training budget

Chose a 2,400-step budget for every configuration and all three seeds after inspecting pilot
generation quality. Kept architecture, dataset, optimizer settings, checkpoint-selection rule,
and the predetermined demo seed unchanged. The cosine learning-rate schedule spans the new budget,
so this is a fresh training study, not a resumed continuation of the pilot.

These studies share the same held-out split. Because pilot test measurements were already visible,
the follow-up is exploratory. A stronger future study should preregister a comparison and reserve
a new untouched test set. No model, seed, or checkpoint is selected by test score for the demo.
