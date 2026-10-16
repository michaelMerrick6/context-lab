import json
import math

import pytest
import torch

from context_lab.baseline import BigramLM
from context_lab.data import evaluation_batches, load_corpus, prepare_corpus, training_batch
from context_lab.experiment import TrainConfig, evaluate, load_checkpoint, train_run
from context_lab.model import ModelConfig


def test_splits_are_deterministic_disjoint_and_fingerprinted(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert prepare_corpus(a) == prepare_corpus(b)
    corpus = load_corpus(a)
    assert len(corpus.train) == 960
    sets = [set(map(tuple, split)) for split in [corpus.train, corpus.validation, corpus.test]]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    (a / "test.json").write_text(json.dumps(["tampered"]))
    with pytest.raises(ValueError, match="fingerprint"):
        load_corpus(a)


def test_evaluation_scores_every_transition_once_without_crossing_documents():
    documents = [[256, 10, 11, 12, 13, 257], [256, 20, 257]]
    expected = [10, 11, 12, 13, 257, 20, 257]
    for context in [1, 2, 4, 8]:
        actual = []
        for _, y in evaluation_batches(documents, context, batch_size=2):
            actual.extend(y[y != -100].tolist())
        assert actual == expected


def test_padding_is_ignored_and_training_sampling_reproducible():
    documents = [[256, 65, 257]]
    a = training_batch(documents, 8, 2, torch.Generator().manual_seed(5))
    b = training_batch(documents, 8, 2, torch.Generator().manual_seed(5))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[1][0].tolist() == [65, 257, -100, -100, -100, -100, -100, -100]


def test_bigram_normalized_and_heldout_metrics_token_weighted():
    docs = [[0, 1, 2, 1], [0, 2]]
    model = BigramLM(vocab_size=3).fit(docs)
    torch.testing.assert_close(model.log_probabilities.exp().sum(-1), torch.ones(3).double())
    expected = (
        -sum(
            model.log_probabilities[a, b].item()
            for doc in docs
            for a, b in zip(doc[:-1], doc[1:], strict=True)
        )
        / 4
    )
    for context in [1, 2, 8]:
        metrics = evaluate(model, docs, context)
        assert metrics["tokens"] == 4
        assert metrics["loss_nats"] == pytest.approx(expected)
        assert metrics["perplexity"] == pytest.approx(math.exp(expected))


def test_end_to_end_training_and_checkpoint_reload(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    prepare_corpus(data)
    metrics = train_run(
        data,
        run,
        ModelConfig(width=16, heads=2, layers=1, max_context=16),
        TrainConfig(steps=3, context_length=16, batch_tokens=32, eval_every=3, threads=1),
    )
    model, checkpoint = load_checkpoint(run / "best.pt")
    corpus = load_corpus(data)
    assert checkpoint["corpus_sha256"] == corpus.manifest["sha256"]
    assert evaluate(model, corpus.test, 16) == metrics["test"]
    assert metrics["trained_tokens"] == 96
    assert (run / "metrics.json").exists()
    with pytest.raises(FileExistsError):
        train_run(data, run, model.config, TrainConfig(context_length=16))
