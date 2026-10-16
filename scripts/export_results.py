"""Usage: python scripts/export_results.py runs/reference"""

import argparse

from context_lab.report import export_report

parser = argparse.ArgumentParser()
parser.add_argument("suite")
parser.add_argument("--out", default="results/reference")
parser.add_argument("--checkpoint", default="checkpoints/context-128-seed-7.pt")
args = parser.parse_args()
export_report(args.suite, args.out, args.checkpoint)
