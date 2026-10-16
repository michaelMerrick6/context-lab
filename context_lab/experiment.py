"""Training, held-out evaluation, safe inference checkpoints, and controlled experiments."""

import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from context_lab.baseline import BigramLM
from context_lab.data import evaluation_batches, load_corpus, training_batch
from context_lab.model import ModelConfig, TransformerLM


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 600
    context_length: int = 128
    batch_tokens: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 0.1
    warmup_steps: int = 30
    eval_every: int = 100
    seed: int = 7
    threads: int = 4
    device: str = "cpu"

    def __post_init__(self):
        if (
            min(self.steps, self.context_length, self.batch_tokens, self.eval_every, self.threads)
            < 1
        ):
            raise ValueError("Training dimensions and intervals must be positive")
        if self.batch_tokens % self.context_length:
            raise ValueError("batch_tokens must be divisible by context_length")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("Invalid optimizer configuration")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configure_runtime(seed=7, threads=4):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    # CPU is the reference backend. Floating-point results can still differ across platforms.
    torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def evaluate(model, documents, context_length, device="cpu"):
    was_training = getattr(model, "training", False)
    if isinstance(model, torch.nn.Module):
        model.eval()
    nll, count = 0.0, 0
    try:
        for x, y in evaluation_batches(documents, context_length, device=device):
            logits, _ = model(x)
            nll += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            count += (y != -100).sum().item()
    finally:
        if isinstance(model, torch.nn.Module):
            model.train(was_training)
    if count == 0:
        raise ValueError("Cannot evaluate an empty corpus")
    loss = nll / count
    return {
        "loss_nats": loss,
        "perplexity": math.exp(loss),
        "bits_per_token": loss / math.log(2),
        "tokens": count,
    }


def load_checkpoint(path, device="cpu"):
    # Only load your own or otherwise trusted artifacts. weights_only narrows the loader surface.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TransformerLM(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), payload


def train_run(data_path, output_path, model_config: ModelConfig, config: TrainConfig):
    if config.context_length > model_config.max_context:
        raise ValueError("Training context exceeds model capacity")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    corpus = load_corpus(data_path)
    configure_runtime(config.seed, config.threads)
    model = TransformerLM(model_config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batch_rng = torch.Generator().manual_seed(config.seed + 1000)
    history, best_loss, best_step = [], float("inf"), 0
    batch_size = config.batch_tokens // config.context_length
    train_seconds, trained_tokens = 0.0, 0
    wall_start = time.perf_counter()

    def validate(step, train_loss=None, grad_norm=None):
        nonlocal best_loss, best_step
        metrics = evaluate(model, corpus.validation, config.context_length, config.device)
        row = {
            "step": step,
            "train_loss": train_loss,
            "validation_loss": metrics["loss_nats"],
            "validation_perplexity": metrics["perplexity"],
            "gradient_norm": grad_norm,
            "trained_tokens": trained_tokens,
        }
        history.append(row)
        if metrics["loss_nats"] < best_loss:
            best_loss, best_step = metrics["loss_nats"], step
            torch.save(
                {
                    "format_version": 1,
                    "model_config": model_config.to_dict(),
                    "train_config": asdict(config),
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "corpus_sha256": corpus.manifest["sha256"],
                    "best_step": step,
                },
                output / "best.pt",
            )
        print(
            f"step {step:4d} | val loss {metrics['loss_nats']:.4f} | "
            f"val perplexity {metrics['perplexity']:.2f}",
            flush=True,
        )

    validate(0)
    for step in range(1, config.steps + 1):
        started = time.perf_counter()
        if config.warmup_steps and step <= config.warmup_steps:
            multiplier = step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
            multiplier = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * multiplier
        model.train()
        x, y = training_batch(
            corpus.train, config.context_length, batch_size, batch_rng, config.device
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # item() synchronizes accelerators, so throughput includes completed optimizer work.
        last_loss, last_norm = loss.item(), grad_norm.item()
        trained_tokens += (y != -100).sum().item()
        train_seconds += time.perf_counter() - started
        if step % config.eval_every == 0 or step == config.steps:
            validate(step, last_loss, last_norm)
    best_model, _ = load_checkpoint(output / "best.pt", config.device)
    validation = evaluate(best_model, corpus.validation, config.context_length, config.device)
    test = evaluate(best_model, corpus.test, config.context_length, config.device)
    result = {
        "model_config": model_config.to_dict(),
        "train_config": asdict(config),
        "parameters": model.parameter_count(),
        "best_step": best_step,
        "corpus_sha256": corpus.manifest["sha256"],
        "validation": validation,
        "test": test,
        "test_context_sweep": {
            str(length): evaluate(best_model, corpus.test, length, config.device)
            for length in (32, 64, 128)
            if length <= model_config.max_context
        },
        "trained_tokens": trained_tokens,
        "training_seconds": train_seconds,
        "training_tokens_per_second": trained_tokens / train_seconds,
        "wall_seconds": time.perf_counter() - wall_start,
        "history": history,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": config.device,
            "threads": config.threads,
        },
    }
    write_json(output / "metrics.json", result)
    return result


def run_suite(data_path, output_path, *, steps=600, seeds=(7, 19, 42), threads=4):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Supply distinct seeds")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    corpus = load_corpus(data_path)
    configure_runtime(threads=threads)
    baseline = BigramLM().fit(corpus.train)
    summary = {
        "corpus": corpus.manifest,
        "protocol": {
            "steps": steps,
            "batch_tokens": 1024,
            "seeds": list(seeds),
            "selection": "lowest full validation loss; test is measured after training",
            "evaluation": "document-local nonoverlapping chunks; every target counted once",
            "comparison": "equal training target budget, same width/depth/optimizer schedule",
        },
        "bigram": {"alpha": baseline.alpha, "test": evaluate(baseline, corpus.test, 128)},
        "runs": [],
    }
    variants = [
        ("context-32", 32, True),
        ("context-128", 128, True),
        ("no-position-128", 128, False),
    ]
    for name, context, positions in variants:
        for seed in seeds:
            print(f"\n{name} / seed {seed}", flush=True)
            result = train_run(
                data_path,
                output / f"{name}-seed-{seed}",
                ModelConfig(position_embeddings=positions),
                TrainConfig(steps=steps, context_length=context, seed=seed, threads=threads),
            )
            summary["runs"].append({"variant": name, **result})
            write_json(output / "summary.json", summary)
    return summary
