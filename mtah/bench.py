"""The DFL benchmark: degraded DFL matches, scored with the organisers' own metric code.

    from mtah import bench
    bench.run({"baseline_smooth50": lambda m: baseline.predict(m, 50)})

Each predictor maps a Match to per-frame codes (0..3). Scores come from the challenge repository's
evaluation/possession_metrics.py, imported rather than copied, so they are the official numbers.
"""
import os
import sys

import numpy as np

from . import config, io_dfl
from .degrade import DEGRADE_VERSION, degrade, load_calibration
from .match import Match


def _metrics():
    sys.path.insert(0, config.evaluation_dir())
    import possession_metrics as pm  # noqa: E402
    return pm


def clean_match(match_id):
    path = os.path.join(config.CACHE_DIR, "dfl", match_id + ".npz")
    if os.path.isfile(path):
        return Match.load(path)
    match = io_dfl.load(match_id)
    match.save(path)
    return match


def degraded_match(match_id, seed=0, calibration=None, mode="random"):
    """mode: 'offpitch' (ball mostly lost off the pitch, plus random gaps) or 'random'."""
    path = os.path.join(config.CACHE_DIR, "dfl_degraded",
                        "{}_{}_s{}_d{}.npz".format(match_id, mode, seed, DEGRADE_VERSION))
    if os.path.isfile(path):
        return Match.load(path)
    base = mode.replace("+anim", "")
    match = degrade(clean_match(match_id), calibration or load_calibration(), seed=seed,
                    off_pitch=(base == "offpitch"), animate_dead=mode.endswith("+anim"))
    match.save(path)
    return match


def run(predictors, match_ids=None, seeds=(0,), verbose=True, mode="random"):
    """{predictor: {"spell_f1": [...], "macro_f1": [...], ...}} over matches x seeds."""
    pm = _metrics()
    match_ids = match_ids or io_dfl.match_ids()
    calibration = load_calibration()
    keys = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "alive_dead_f1",
            "team_acc_both_alive")
    out = {name: {k: [] for k in keys} for name in predictors}
    for match_id in match_ids:
        for seed in seeds:
            match = degraded_match(match_id, seed, calibration, mode)
            gt = match.labels
            for name, predict in predictors.items():
                pred = np.asarray(predict(match), np.int8)
                result = pm.evaluate(gt, pred)
                for k in keys:
                    out[name][k].append(result[k])
            if verbose:
                print("  scored {} seed {}".format(match_id, seed), flush=True)
    return out


def print_table(results):
    print("\n{:<24} {:>8} {:>8} {:>9} {:>8} {:>8}  {:>12}".format(
        "predictor", "spellF1", "  sd", "macroF1", "aliveF1", "teamAcc", "poss/gt"))
    rows = sorted(results.items(), key=lambda kv: -np.mean(kv[1]["spell_f1_mean"]))
    for name, r in rows:
        print("{:<24} {:>8.4f} {:>8.4f} {:>9.4f} {:>8.4f} {:>8.4f}  {:>12}".format(
            name, np.mean(r["spell_f1_mean"]), np.std(r["spell_f1_mean"]),
            np.mean(r["macro_f1"]), np.mean(r["alive_dead_f1"]),
            np.mean(r["team_acc_both_alive"]),
            "{:.0f}/{:.0f}".format(np.mean(r["n_pred_spells"]), np.mean(r["n_gt_spells"]))))
