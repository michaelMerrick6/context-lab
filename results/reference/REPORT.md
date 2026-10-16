# 2,400-step experiment

These measurements were produced locally by the included suite. They describe a small synthetic corpus, not general language competence.
See the [experiment log](../../docs/experiment-log.md) for the pilot and follow-up history. The studies share a test split, so the follow-up is exploratory.

## Protocol

- Corpus fingerprint: `a476f7deb6a3570ef9043362d1388a4a71ffa7483e16867426265ee652ec89ba`.
- Document counts: `{'train': 960, 'validation': 120, 'test': 120}`.
- Seeds: `[7, 19, 42]`; 2400 optimizer steps per run.
- 1,024 training targets per step; identical width, depth, and optimizer schedule.
- Select the checkpoint with lowest full validation loss; score the test set afterward.
- Evaluation resets context at chunk boundaries and scores every target exactly once.
- Mean ± sample standard deviation across seeds; no confidence interval is claimed.

## Held-out results

| Model | Parameters | Test loss (nats/token) | Test perplexity | Train tokens/sec |
|---|---:|---:|---:|---:|
| Count bigram | 66,564 counts | 1.902 | 6.703 | — |
| context-32 | 124,288 | 0.402 ± 0.002 | 1.494 ± 0.003 | 109,103 |
| context-128 | 124,288 | 0.533 ± 0.005 | 1.703 ± 0.009 | 73,926 |
| no-position-128 | 116,096 | 0.978 ± 0.054 | 2.662 ± 0.144 | 71,396 |

![Validation learning curves](learning-curves.svg)

## Context at inference

Test perplexity at different evaluation chunk lengths, averaged across seeds. A 32-token-trained model evaluated at 64 or 128 encounters untrained absolute position embeddings; interpret those cells as distribution shift.

| Training variant | Eval 32 | Eval 64 | Eval 128 |
|---|---:|---:|---:|
| context-32 | 1.494 | 9.703 | 26.026 |
| context-128 | 5.011 | 4.084 | 1.703 |
| no-position-128 | 5.618 | 3.695 | 2.662 |

## Interpretation limits

- Template overlap remains even though exact documents are disjoint. The test set measures new combinations of familiar templates.
- Changing context also changes chunk boundaries and batch size. The comparison measures this complete training/evaluation policy.
- The no-position variant removes 8,192 learned parameters. Causal masking can still convey ordering information; this is not an orderless model.
- Equal token exposure is not equal compute. CPU throughput is descriptive and includes batch construction, forward/backward passes, and optimizer work.
- Perplexity is tokenization-dependent and does not establish reasoning, factuality, or useful text generation.
- Three seeds expose some variability; stronger claims need more runs and a separate natural-language corpus.

## Environment

```json
{
  "python": "3.13.5",
  "torch": "2.14.1",
  "platform": "macOS-15.6.1-arm64-arm-64bit-Mach-O",
  "device": "cpu",
  "threads": 4
}
```

Raw per-seed metrics and learning histories are in [summary.json](summary.json).
