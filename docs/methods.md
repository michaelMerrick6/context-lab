# Experimental design

## Question and interventions

The primary question is whether increasing training and evaluation context improves prediction on
a small procedural-fiction corpus under a fixed target-token budget. This is a systems experiment
on a complete policy, not an isolated proof about language-model memory.

| Variant | Train context | Position embeddings | Batch size | Target tokens/step |
|---|---:|---|---:|---:|
| context-32 | 32 | learned | 32 | 1,024 |
| context-128 | 128 | learned | 8 | 1,024 |
| no-position-128 | 128 | absent | 8 | 1,024 |

The two context variants allocate the same 128-position table, which holds parameter count constant.
Only positions 0–31 are trained in the short-context model. The no-position model removes 8,192
parameters. Causal masking still makes the architecture sensitive to aspects of ordering.

All runs use width 64, two layers, four heads, dropout 0.1, tied embedding/output weights, AdamW
with learning rate 0.001 and weight decay 0.1, 30 warmup steps, cosine decay to 10% of the peak
learning rate, and gradient clipping at norm 1.0. Default seeds are 7, 19, and 42.

## Corpus and leakage controls

The default generator creates 1,200 unique short stories from three templates and combinations
of names, objects, colors, places, weather, and actions. This is original procedural data, not a
downloaded collection. Exact duplicates are removed before a seeded document-level 80/10/10 split.
The fingerprint covers ordered split contents, and is checked on load.

Byte tokenization has 258 entries: UTF-8 byte values 0–255, BOS=256, EOS=257. It requires no fitting
on any split. Unicode characters can occupy several tokens. A context of 128 tokens is 128 byte or
boundary tokens, not 128 words. Each document receives its own BOS and EOS.

All templates appear across splits. The test distribution asks for new combinations of familiar
patterns. Document separation prevents exact-document leakage, but does not make this a test of
natural-language generalization. A next study should use source-grouped splits of a diverse corpus.

## Training policy

Each batch samples training documents uniformly, then samples a contiguous window within each
document. No example crosses a document boundary. Short examples are right padded; padded targets
use -100 and contribute no loss. All generated documents exceed the reference context lengths, so
reference runs receive exactly 1,024 scored targets per update. Custom corpora with shorter documents
may receive fewer; `trained_tokens` records actual exposure.

Sampling RNG is separate from model/dropout RNG. Seeds and runtime configuration are recorded.
Exact bitwise reproducibility across different hardware and PyTorch versions is not promised.

## Evaluation and checkpoint selection

The entire validation split is scored at step 0, every 100 steps, and the final step. The lowest
validation loss selects the checkpoint. The test split is then scored using that checkpoint.
The 2,400-step reference followed a 600-step pilot whose samples had malformed words. Both use
the same test split, which was already scored during the pilot. This is an exploratory benchmark,
not a confirmatory study with a previously unseen final holdout. The longer budget was applied
to every configuration and seed, without selecting the variant or demo seed on test performance.

Each document's prediction pairs are partitioned into nonoverlapping target chunks. A chunk contains
inputs `d[i:i+C]` and targets `d[i+1:i+C+1]`. Every document transition is scored exactly once,
including BOS→first byte and last byte→EOS. No targets cross documents. A shared denominator counts
actual targets rather than averaging batch losses, which could overweight short final batches.

Context resets at each chunk boundary. Thus a longer context reduces resets as well as allowing
more history within a chunk. The bigram is invariant to this partition because it only uses the
immediately preceding token. The report also evaluates every checkpoint under a shared set of
chunk lengths. Inference at positions 32–127 in the short-context model uses untrained positional
embeddings, so those entries are explicitly a distribution-shift diagnostic.

Cross entropy is reported in nats/target and perplexity is its exponential. Bits per token is also
stored. These include boundary tokens and should not be described as a pure bits-per-byte metric.
Perplexities are comparable here because the vocabulary and scored targets are shared.

## Baseline and uncertainty

The bigram is fitted once using all training transitions, with additive smoothing alpha=0.1. Its
66,564 entries are counts, not gradient-trained parameters. It gets the entire training corpus once;
the neural models get repeated sampled exposure. This is a simple predictive baseline, not a
compute-matched comparison.

The report shows mean and sample standard deviation across three independent seeds. This is a
small sample and is not a confidence interval or a significance test. Differences should be treated
as descriptive results for this corpus and training budget.

## Runtime accounting

Training time covers batch construction, forward and backward passes, clipping, and optimizer updates.
Validation and checkpoint writes are excluded from training throughput; total wall time is stored
separately. The reference backend is CPU with four PyTorch threads. No GPU cost estimate is inferred.
