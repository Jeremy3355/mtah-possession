"""Score the organisers' four baseline variants on the degraded DFL benchmark.

The point is a realism check of the degradation: on the real match the published full-match scores
are raw 0.0486 / smooth25 0.1996 / smooth50 0.2433 / smooth125 0.2242 (spellF1) and macroF1
0.65-0.68. A benchmark that reproduces that ordering and rough level is one we can tune on.

    python scripts/bench_baselines.py [--seeds 0 1]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import baseline, bench  # noqa: E402
from mtah.match import X  # noqa: E402


def clean_oracle(match):
    """Upper bound sanity check: the ground truth itself must score 1.0."""
    return match.labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--mode", default="random", choices=["random", "offpitch"])
    args = parser.parse_args()
    predictors = {"gt_oracle": clean_oracle}
    for w in (0, 25, 50, 125):
        predictors["baseline_" + ("smooth{}".format(w) if w else "raw")] = \
            (lambda m, w=w: baseline.predict(m, w))
    bench.print_table(bench.run(predictors, seeds=args.seeds, mode=args.mode))
