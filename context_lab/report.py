"""Create a compact, auditable report from actual experiment artifacts."""

import hashlib
import json
import shutil
import statistics
from pathlib import Path

COLORS = {"context-32": "#236b75", "context-128": "#d15531", "no-position-128": "#765da1"}


def grouped(summary):
    return {name: [r for r in summary["runs"] if r["variant"] == name] for name in COLORS}


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def loss_chart(summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.labelcolor": "#263c3c",
            "text.color": "#263c3c",
            "xtick.color": "#263c3c",
            "ytick.color": "#263c3c",
            "svg.hashsalt": "context-lab",
        }
    ):
        fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
        fig.set_facecolor("#faf8f3")
        ax.set_facecolor("#faf8f3")
        for name, runs in grouped(summary).items():
            series = []
            for row in zip(*(r["history"] for r in runs), strict=True):
                mean, sd = mean_sd([h["validation_loss"] for h in row])
                series.append((row[0]["step"], mean, sd))
            steps, means, sds = zip(*series, strict=True)
            ax.plot(steps, means, label=name, color=COLORS[name], linewidth=2.3)
            ax.fill_between(
                steps,
                [m - s for m, s in zip(means, sds, strict=True)],
                [m + s for m, s in zip(means, sds, strict=True)],
                color=COLORS[name],
                alpha=0.13,
                linewidth=0,
            )
        ax.set_title("Learning to predict the next byte", loc="left", fontsize=17, pad=20)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Validation loss (nats/token)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#dedfd8", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right", title="Mean ± 1 sample SD")
        fig.savefig(destination, format="svg", metadata={"Date": None})
        plt.close(fig)
        svg = Path(destination)
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def export_report(suite_path, output_path="results/reference", checkpoint_path=None):
    suite, output = Path(suite_path), Path(output_path)
    summary = json.loads((suite / "summary.json").read_text())
    expected = 3 * len(summary["protocol"]["seeds"])
    if len(summary["runs"]) != expected:
        raise ValueError("Suite is incomplete; refusing to export partial benchmark results")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(suite / "summary.json", output / "summary.json")
    loss_chart(summary, output / "learning-curves.svg")
    lines = [
        f"# {summary['protocol']['steps']:,}-step experiment",
        "",
        "These measurements were produced locally by the included suite. "
        "They describe a small synthetic corpus, not general language competence.",
        "See the [experiment log](../../docs/experiment-log.md) "
        "for the pilot and follow-up history. "
        "The studies share a test split, so the follow-up is exploratory.",
        "",
        "## Protocol",
        "",
        f"- Corpus fingerprint: `{summary['corpus']['sha256']}`.",
        f"- Document counts: `{summary['corpus']['documents']}`.",
        f"- Seeds: `{summary['protocol']['seeds']}`; "
        f"{summary['protocol']['steps']} optimizer steps per run.",
        "- 1,024 training targets per step; identical width, depth, and optimizer schedule.",
        "- Select the checkpoint with lowest full validation loss; score the test set afterward.",
        "- Evaluation resets context at chunk boundaries and scores every target exactly once.",
        "- Mean ± sample standard deviation across seeds; no confidence interval is claimed.",
        "",
        "## Held-out results",
        "",
        "| Model | Parameters | Test loss (nats/token) | Test perplexity | Train tokens/sec |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = summary["bigram"]["test"]
    lines.append(
        f"| Count bigram | 66,564 counts | {baseline['loss_nats']:.3f} | "
        f"{baseline['perplexity']:.3f} | — |"
    )
    for name, runs in grouped(summary).items():
        loss, loss_sd = mean_sd([r["test"]["loss_nats"] for r in runs])
        ppl, ppl_sd = mean_sd([r["test"]["perplexity"] for r in runs])
        speed = statistics.mean(r["training_tokens_per_second"] for r in runs)
        lines.append(
            f"| {name} | {runs[0]['parameters']:,} | {loss:.3f} ± {loss_sd:.3f} | "
            f"{ppl:.3f} ± {ppl_sd:.3f} | {speed:,.0f} |"
        )
    lines += [
        "",
        "![Validation learning curves](learning-curves.svg)",
        "",
        "## Context at inference",
        "",
        "Test perplexity at different evaluation chunk lengths, averaged across seeds. "
        "A 32-token-trained model evaluated at 64 or 128 encounters untrained "
        "absolute position embeddings; interpret those cells as distribution shift.",
        "",
        "| Training variant | Eval 32 | Eval 64 | Eval 128 |",
        "|---|---:|---:|---:|",
    ]
    for name, runs in grouped(summary).items():
        values = [
            statistics.mean(r["test_context_sweep"][str(c)]["perplexity"] for r in runs)
            for c in (32, 64, 128)
        ]
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Template overlap remains even though exact documents are disjoint. "
        "The test set measures new combinations of familiar templates.",
        "- Changing context also changes chunk boundaries and batch size. "
        "The comparison measures this complete training/evaluation policy.",
        "- The no-position variant removes 8,192 learned parameters. "
        "Causal masking can still convey ordering information; this is not an orderless model.",
        "- Equal token exposure is not equal compute. CPU throughput is descriptive "
        "and includes batch construction, forward/backward passes, and optimizer work.",
        "- Perplexity is tokenization-dependent and does not establish reasoning, "
        "factuality, or useful text generation.",
        "- Three seeds expose some variability; stronger claims need more runs and "
        "a separate natural-language corpus.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(summary["runs"][0]["environment"], indent=2),
        "```",
        "",
        "Raw per-seed metrics and learning histories are in [summary.json](summary.json).",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    if checkpoint_path:
        # Fix the demo seed in advance rather than selecting the best test result.
        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(suite / "context-128-seed-7" / "best.pt", target)
        export_samples(target, output / "samples.json")


def export_samples(checkpoint_path, output_path):
    import torch

    from context_lab.experiment import configure_runtime, load_checkpoint
    from context_lab.tokenizer import ByteTokenizer

    configure_runtime()
    model, payload = load_checkpoint(checkpoint_path)
    tokenizer = ByteTokenizer()
    examples = []
    for prompt in ["Mira found a", "At the harbor,", "The rain had stopped"]:
        tokens = torch.tensor([[tokenizer.bos_id, *tokenizer.encode(prompt)]])
        generated = model.generate(
            tokens,
            max_new_tokens=220,
            temperature=0.8,
            top_k=40,
            context_length=payload["train_config"]["context_length"],
            generator=torch.Generator().manual_seed(7),
        )
        examples.append({"prompt": prompt, "output": tokenizer.decode(generated[0].tolist())})
    record = {
        "checkpoint": str(checkpoint_path),
        "sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "sampling": {"seed": 7, "temperature": 0.8, "top_k": 40, "max_new_tokens": 220},
        "examples": examples,
    }
    Path(output_path).write_text(json.dumps(record, indent=2) + "\n")
