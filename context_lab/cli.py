import argparse
import json
from pathlib import Path

import torch

from context_lab.data import load_corpus, prepare_corpus
from context_lab.experiment import (
    TrainConfig,
    configure_runtime,
    evaluate,
    load_checkpoint,
    run_suite,
    train_run,
)
from context_lab.model import ModelConfig
from context_lab.tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train, measure, and inspect a tiny transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create deterministic document-level splits")
    prepare.add_argument("--out", default="artifacts/corpus")
    prepare.add_argument("--source", help="UTF-8 text with documents separated by blank lines")
    prepare.add_argument("--seed", type=int, default=31415)
    train = commands.add_parser("train", help="Train one model from random initialization")
    train.add_argument("--data", default="artifacts/corpus")
    train.add_argument("--out", default="runs/single")
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--context", type=int, default=128)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    train.add_argument("--no-positions", action="store_true")
    suite = commands.add_parser("suite", help="Run the three variants with repeated seeds")
    suite.add_argument("--data", default="artifacts/corpus")
    suite.add_argument("--out", default="runs/suite")
    suite.add_argument("--steps", type=int, default=600)
    suite.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 42])
    suite.add_argument("--threads", type=int, default=4)
    sample = commands.add_parser("sample", help="Generate from a locally trained checkpoint")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--prompt", default="Mira found a")
    sample.add_argument("--tokens", type=int, default=220)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--top-k", type=int, default=40)
    sample.add_argument("--seed", type=int, default=7)
    score = commands.add_parser("evaluate", help="Score a checkpoint on its held-out corpus")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--data", default="artifacts/corpus")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_corpus(args.out, source=args.source, seed=args.seed), indent=2))
    elif args.command == "train":
        result = train_run(
            args.data,
            args.out,
            ModelConfig(
                max_context=max(128, args.context), position_embeddings=not args.no_positions
            ),
            TrainConfig(
                steps=args.steps,
                context_length=args.context,
                seed=args.seed,
                threads=args.threads,
                device=args.device,
            ),
        )
        print(json.dumps({"test": result["test"], "best_step": result["best_step"]}, indent=2))
    elif args.command == "suite":
        run_suite(args.data, args.out, steps=args.steps, seeds=args.seeds, threads=args.threads)
        print(f"Results saved to {Path(args.out) / 'summary.json'}")
    elif args.command == "sample":
        configure_runtime(args.seed)
        model, checkpoint = load_checkpoint(args.checkpoint)
        tokenizer = ByteTokenizer()
        prompt = torch.tensor([[tokenizer.bos_id, *tokenizer.encode(args.prompt)]])
        tokens = model.generate(
            prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            context_length=checkpoint["train_config"]["context_length"],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(tokenizer.decode(tokens[0].tolist()))
    else:
        configure_runtime()
        model, checkpoint = load_checkpoint(args.checkpoint)
        corpus = load_corpus(args.data)
        if checkpoint["corpus_sha256"] != corpus.manifest["sha256"]:
            parser.error("Checkpoint and corpus fingerprints differ")
        print(
            json.dumps(
                evaluate(model, corpus.test, checkpoint["train_config"]["context_length"]), indent=2
            )
        )


if __name__ == "__main__":
    main()
