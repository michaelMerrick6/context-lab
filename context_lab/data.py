"""Document-level splits, deterministic sampling, and exactly-once evaluation targets."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from context_lab.tokenizer import ByteTokenizer


@dataclass
class Corpus:
    train: list[list[int]]
    validation: list[list[int]]
    test: list[list[int]]
    manifest: dict


def toy_documents(count: int = 1200, seed: int = 31415) -> list[str]:
    """Original template-generated fiction, for cheap mechanics experiments only."""
    if count < 20 or count > 100_000:
        raise ValueError("count must be between 20 and 100000")
    rng = random.Random(seed)
    names = ["Mira", "Jonah", "Ada", "Leo", "Nora", "Eli", "Sam", "Iris", "Omar", "June"]
    colors = ["red", "blue", "green", "gold", "silver", "white", "black", "violet"]
    objects = ["key", "book", "box", "stone", "bell", "map", "cup", "coin"]
    places = ["river", "garden", "tower", "bridge", "forest", "station", "market", "harbor"]
    weather = ["rain", "fog", "snow", "wind", "sunshine", "thunder"]
    actions = [
        "waited quietly",
        "walked home",
        "called a friend",
        "opened the gate",
        "sat by the fire",
        "looked at the stars",
        "wrote a letter",
        "lit a candle",
    ]
    documents, seen = [], set()
    while len(documents) < count:
        name, color, obj, place, sky, action = (
            rng.choice(items) for items in (names, colors, objects, places, weather, actions)
        )
        other = rng.choice([n for n in names if n != name])
        template = rng.randrange(3)
        if template == 0:
            text = (
                f"{name} found a {color} {obj} near the {place}. "
                f"Outside, there was {sky}. {other} {action}. "
                f"Later, {name} returned to the {place}. The {obj} was still {color}.\n"
            )
        elif template == 1:
            text = (
                f"At the {place}, {name} gave {other} a {color} {obj}. "
                f"They talked about the {sky} and {action}. "
                f"Before leaving, {other} thanked {name} for the {obj}.\n"
            )
        else:
            text = (
                f"The {sky} had stopped when {name} reached the {place}. "
                f"{other} carried a {color} {obj} and {action}. "
                f"Together they took the {obj} home from the {place}.\n"
            )
        if text not in seen:
            documents.append(text)
            seen.add(text)
    return documents


def prepare_corpus(path: str | Path, *, source: str | Path | None = None, seed: int = 31415):
    """External corpora use blank lines as document separators; exact duplicates are removed."""
    if source is None:
        documents = toy_documents(seed=seed)
        provenance = "Original procedural fiction; generator in context_lab/data.py"
    else:
        documents = [part.strip() for part in Path(source).read_text().split("\n\n")]
        provenance = f"User-provided corpus: {Path(source).name}; user must verify usage rights"
    documents = list(dict.fromkeys(d for d in documents if d.strip()))
    if len(documents) < 20:
        raise ValueError("Need at least 20 distinct documents, separated by blank lines")
    random.Random(seed).shuffle(documents)
    train_end, val_end = int(len(documents) * 0.8), int(len(documents) * 0.9)
    splits = {
        "train": documents[:train_end],
        "validation": documents[train_end:val_end],
        "test": documents[val_end:],
    }
    digest = hashlib.sha256(json.dumps(splits, ensure_ascii=False).encode()).hexdigest()
    manifest = {
        "sha256": digest,
        "seed": seed,
        "provenance": provenance,
        "documents": {name: len(docs) for name, docs in splits.items()},
        "split_policy": "deduplicate exact text, seeded shuffle, 80/10/10 by document",
        "tokenizer": "UTF-8 bytes plus BOS=256 and EOS=257",
    }
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    for name, docs in splits.items():
        (output / f"{name}.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_corpus(path: str | Path) -> Corpus:
    root, tokenizer = Path(path), ByteTokenizer()
    splits = {
        name: json.loads((root / f"{name}.json").read_text())
        for name in ("train", "validation", "test")
    }
    manifest = json.loads((root / "manifest.json").read_text())
    digest = hashlib.sha256(json.dumps(splits, ensure_ascii=False).encode()).hexdigest()
    if manifest["sha256"] != digest:
        raise ValueError("Corpus fingerprint mismatch; rerun prepare after changing data")
    sets = [set(docs) for docs in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Documents overlap between splits")
    return Corpus(
        **{k: [tokenizer.encode(d, boundaries=True) for d in docs] for k, docs in splits.items()},
        manifest=manifest,
    )


def training_batch(documents, context_length, batch_size, generator, device="cpu"):
    # Sample documents uniformly. Short documents use padding; targets never cross documents.
    x = torch.zeros((batch_size, context_length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for row in range(batch_size):
        doc = documents[torch.randint(len(documents), (1,), generator=generator).item()]
        start = torch.randint(max(1, len(doc) - context_length), (1,), generator=generator).item()
        chunk = doc[start : start + context_length + 1]
        length = len(chunk) - 1
        x[row, :length] = torch.tensor(chunk[:-1])
        y[row, :length] = torch.tensor(chunk[1:])
    return x.to(device), y.to(device)


def evaluation_batches(documents, context_length, batch_size=32, device="cpu"):
    # Non-overlapping target chunks: every document transition is counted exactly once.
    # Context resets at each chunk. This policy is intentionally shared by every model.
    examples = []
    for doc in documents:
        for start in range(0, len(doc) - 1, context_length):
            examples.append(doc[start : start + context_length + 1])
    for offset in range(0, len(examples), batch_size):
        chunks = examples[offset : offset + batch_size]
        x = torch.zeros((len(chunks), context_length), dtype=torch.long)
        y = torch.full_like(x, -100)
        for row, chunk in enumerate(chunks):
            length = len(chunk) - 1
            x[row, :length] = torch.tensor(chunk[:-1])
            y[row, :length] = torch.tensor(chunk[1:])
        yield x.to(device), y.to(device)
