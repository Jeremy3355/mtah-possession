"""Leave-one-match-out evaluation on the degraded DFL benchmark, for any feature version.

For each held-out match: train on the other six (both degradation modes, seed 0), predict the
held-out match in each mode, decode over a grid of costs, score with the official metric.
Probabilities are cached per --tag, so re-running the decoder grid is cheap.

    python scripts/cv.py --tag v2 --version 2
"""
import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import bench, config, model  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
MODES = ("random", "offpitch")
KEYS = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "n_pred_spells", "n_gt_spells")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--version", type=int, default=model.FEATURE_VERSION)
    parser.add_argument("--switch", type=float, nargs="+", default=[2.0, 4.0, 8.0, 12.0])
    parser.add_argument("--dead", type=float, nargs="+", default=[8.0, 16.0, 24.0, 32.0])
    parser.add_argument("--ball-motion", action="store_true",
                        help="re-admit ball_speed and ball_travelled features")
    parser.add_argument("--arch", default=model.ARCH, choices=["joint", "factor"])
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0],
                        help="degradation seeds of the training matches (test always uses seed 0)")
    parser.add_argument("--test-suffix", default="",
                        help="e.g. +anim: score on held-out matches whose dead ball is animated")
    parser.add_argument("--compare", help="tag of an earlier run: print paired per-match differences")
    args = parser.parse_args()
    model.FEATURE_VERSION = args.version
    model.ARCH = args.arch
    if args.ball_motion:
        model.EXCLUDED_PREFIXES = tuple(p for p in model.EXCLUDED_PREFIXES
                                        if p not in ("ball_speed", "ball_travelled"))

    pm = bench._metrics()
    ids = bench.io_dfl.match_ids()
    cache = os.path.join(config.CACHE_DIR, "cv_" + args.tag)
    os.makedirs(cache, exist_ok=True)
    res = {}  # (mode, s, d) -> {key: [..]}
    for held in ids:
        paths = {m: os.path.join(cache, "{}_{}.npy".format(held, m)) for m in MODES}
        if not all(os.path.isfile(p) for p in paths.values()):
            clf = model.train([bench.degraded_match(i, seed, mode=m)
                               for i in ids if i != held for m in MODES for seed in args.train_seeds],
                              arch=args.arch)
        for mode in MODES:
            test = bench.degraded_match(held, 0, mode=mode + args.test_suffix)
            if os.path.isfile(paths[mode]):
                proba = np.load(paths[mode])
            else:
                proba = model.predict_proba(clf, test)
                np.save(paths[mode], proba)
            for s in args.switch:
                for d in args.dead:
                    r = pm.evaluate(test.labels, model.decode(proba, test.half, s, d))
                    slot = res.setdefault((mode, s, d), {k: [] for k in KEYS})
                    for k in KEYS:
                        slot[k].append(r[k])
        print("held out", held, flush=True)

    for mode in MODES:
        print("\n[{}] mode {}  spellF1 (switch rows x dead cols)".format(args.tag, mode))
        print("  " + " " * 10 + " ".join("{:>7}".format(d) for d in args.dead))
        for s in args.switch:
            print("  {:>10}".format(s) + " ".join(
                "{:7.4f}".format(np.mean(res[(mode, s, d)]["spell_f1_mean"])) for d in args.dead))
    grid = [(s, d) for s in args.switch for d in args.dead]
    both = {k: np.mean([np.mean(res[(m,) + k]["spell_f1_mean"]) for m in MODES]) for k in grid}
    best = max(both, key=both.get)
    print("\n[{}] best: switch {} dead {}  mean spellF1 {:.4f}".format(args.tag, best[0], best[1], both[best]))
    for k in sorted({best, (4.0, 16.0)} & set(grid)):
        for mode in MODES:
            r = res[(mode,) + k]
            print("  s{:<4} d{:<4} {:<9} spellF1 {:.4f} (sd {:.4f})  macroF1 {:.4f}  aliveDeadF1 {:.4f}  poss {:.0f}/{:.0f}".format(
                k[0], k[1], mode, np.mean(r["spell_f1_mean"]), np.std(r["spell_f1_mean"]),
                np.mean(r["macro_f1"]), np.mean(r["alive_dead_f1"]),
                np.mean(r["n_pred_spells"]), np.mean(r["n_gt_spells"])))

    import json
    store = {"{}|{}|{}".format(*k): v for k, v in res.items()}
    with open(os.path.join(cache, "results.json"), "w") as handle:
        json.dump(store, handle)
    if args.compare:
        with open(os.path.join(config.CACHE_DIR, "cv_" + args.compare, "results.json")) as handle:
            ref = json.load(handle)
        print("\npaired per-match spellF1, {} minus {} (switch 4, dead 16):".format(args.tag, args.compare))
        for mode in MODES:
            key = "{}|4.0|16.0".format(mode)
            if key not in ref or key not in store:
                continue
            diff = np.array(store[key]["spell_f1_mean"]) - np.array(ref[key]["spell_f1_mean"])
            print("  {:<9} mean {:+.4f}  per match {}  wins {}/{}".format(
                mode, diff.mean(), " ".join("{:+.3f}".format(d) for d in diff),
                int((diff > 0).sum()), len(diff)))
