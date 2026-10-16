# Context Lab model card

## Model and training

Small decoder-only byte language model implemented in PyTorch. The default architecture uses two
transformer layers, width 64, four heads, a learned 128-position table, and tied input/output
embeddings. The bundled checkpoint is `context-128-seed-7`, selected by validation loss within that
run. Seed 7 is the predetermined demo seed; it is not selected for best test performance.

The original procedural-fiction corpus has 1,200 unique stories and an 80/10/10 document split.
The corpus generator and fingerprint are included. No pretrained weights or hosted inference API
are used. Training configuration, runtime, and per-seed metrics are recorded in
`results/reference/summary.json`.

## Intended use

Learning transformer mechanics, inspecting attention, reproducing small experiments, and exploring
the difference between training loss, held-out prediction, and generation quality. Suitable for
local educational demonstrations and portfolio discussion with the limitations stated.

## Limitations

The corpus is synthetic, small, English, and template-based. The model can reproduce syntax without
tracking the right person, object, or color. It may repeat text, generate invalid UTF-8 sequences,
or fail outside the training templates. Byte decoding replaces invalid sequences for display.
It has no retrieval, tools, instruction tuning, factual grounding, or safety training. It should not
be used to answer consequential questions or described as a general-purpose chatbot.

Document splits share templates. Held-out perplexity measures prediction of new combinations of
familiar patterns, not broad language understanding. The reference report includes uncertainty
across three seeds and explains the context-reset and compute confounders.

## Artifacts and license

MIT license for the implementation, generator, and bundled model weights. Checkpoint loading uses
PyTorch's `weights_only=True`; use artifacts from sources you trust. The checkpoint contains weights
and configuration, not personal data, user prompts, optimizer state, or credentials.
