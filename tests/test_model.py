import pytest
import torch

from context_lab.model import ModelConfig, TransformerLM
from context_lab.tokenizer import ByteTokenizer


@pytest.fixture
def model():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    return TransformerLM(ModelConfig(width=16, heads=2, layers=2, max_context=16, dropout=0))


def test_utf8_round_trip():
    tokenizer = ByteTokenizer()
    text = "hello, café 🌱\n"
    assert tokenizer.decode(tokenizer.encode(text, boundaries=True)) == text


def test_future_tokens_cannot_change_past_predictions(model):
    model.eval()
    tokens = torch.randint(0, 256, (2, 12))
    changed = tokens.clone()
    changed[:, 7:] = torch.randint(0, 256, (2, 5))
    original, _ = model(tokens)
    altered, _ = model(changed)
    torch.testing.assert_close(original[:, :7], altered[:, :7], atol=1e-6, rtol=1e-6)


def test_attention_mask_and_probability_normalization(model):
    model.eval()
    weights = model.attention_weights(torch.tensor([[256, 65, 66, 67]]))
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))


def test_tied_weights_and_backward(model):
    assert model.output.weight is model.token_embedding.weight
    x = torch.randint(0, 256, (2, 8))
    _, loss = model(x, x.roll(-1, dims=1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_can_overfit_a_tiny_sequence(model):
    x = torch.tensor([[256, 65, 66, 67, 65, 66, 67]])
    y = torch.tensor([[65, 66, 67, 65, 66, 67, 257]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = model(x, y)[1].item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.1


def test_generation_restores_mode_and_supports_context_crop(model):
    prompt = torch.tensor([[256, 65]])
    model.train()
    a = model.generate(prompt, max_new_tokens=8, top_k=1, context_length=4, eos_id=-1)
    b = model.generate(prompt, max_new_tokens=8, top_k=1, context_length=4, eos_id=-1)
    assert model.training
    assert a.shape == (1, 10)
    assert torch.equal(a, b)
    assert not (a[:, 2:] == 256).any()


def test_invalid_model_configuration():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(width=15, heads=2)
