# Understanding and extending Context Lab

This project was developed with AI coding assistance. The best way to make it your own is to trace
the computation, reproduce the measurements, and add an experiment you can defend.

## Read in this order

1. `tokenizer.py`: a string becomes UTF-8 bytes, each mapped directly to a vocabulary index.
2. `data.py`: build input/target pairs shifted by one token; inspect a padded batch.
3. `baseline.py`: predict the next byte using only the preceding byte's transition counts.
4. `model.py`: follow tensor shapes through embeddings, attention, and the vocabulary projection.
5. `experiment.py`: follow one optimizer step, validation checkpoint selection, and test evaluation.

Start by printing one batch and explaining each target. Then run the overfitting test: if a model
cannot memorize a tiny sequence, a larger experiment is hard to interpret.

## The math you should be able to explain

For a token embedding matrix `X` with shape `[batch, time, width]`, each head computes
`Q = XWq`, `K = XWk`, and `V = XWv`. Its output is

```text
softmax(Q Kᵀ / sqrt(head_width) + causal_mask) V
```

The causal mask is zero on/below the diagonal and negative infinity above it. Applying it before
softmax assigns zero probability to future positions. Dividing by the square root of head width
keeps dot-product magnitudes from growing unchecked as head width grows.

The network predicts a probability distribution over next tokens. Cross entropy is the average
negative log probability assigned to the observed targets. Backpropagation computes derivatives;
AdamW uses those derivatives to update parameters. PyTorch implements differentiation and the
optimizer, while this project defines the architecture and training policy.

Residual connections add the input to a sublayer's output. Layer normalization stabilizes the
activations. Weight tying uses the same matrix for input embeddings and output vocabulary scores,
reducing parameter count. `model.parameter_count()` avoids counting shared parameters twice.

## Questions to answer without looking at the code

- Why would omitting the causal mask make validation misleading?
- Why does the target at time t correspond to the input token at time t+1?
- Why can a character occupy more than one byte token?
- Why is the bigram's result unchanged by evaluation context length?
- What does low training loss with high validation loss suggest?
- Why select a checkpoint on validation rather than test loss?
- Why is averaging per-batch losses wrong when batches contain different target counts?
- Why is 32 versus 128 context not an equal-compute comparison?
- What do three seeds tell you, and what do they leave uncertain?
- Why does removing position embeddings not make a causal transformer entirely orderless?
- Why doesn't a plausible-looking continuation prove the model preserves facts?
- What changes when a short-context-trained model sees untrained absolute positions?

## A useful personal extension

Add a **controlled color-recall test**. Introduce a color early in a story, insert a controlled
number of distractor bytes, and query the color at the end. Split complete generated examples
before training and reserve unseen combinations for testing. Measure exact-match color accuracy
against distractor length, alongside perplexity.

Predefine the dataset, metric, and comparison before running the experiment. Keep the query suffix
identical across colors so a bigram cannot solve it from the last byte alone. Confirm label balance
and report chance accuracy. This measures a targeted dependency more directly than aggregate loss.

Other extensions include a recurrent-network baseline, source-grouped natural-text evaluation,
relative position embeddings, and comparison of explicit versus fused attention. Change one main
factor at a time and record both accuracy and resource use.

## Writing about the project

Use only results in the report and describe your actual contribution. Once you understand and
reproduce the work, a factual project summary can describe the architecture, experiment controls,
and measured outcomes. Do not claim production deployment, novel research, or general intelligence
from these toy experiments. Keep a short log of experiments you personally ran and what they taught you.
