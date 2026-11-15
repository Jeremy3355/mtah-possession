"""Where are possessions lost? Uses the cached CV probabilities of a tag (scripts/cv.py).

For every ground-truth possession of every held-out DFL match, at IoU 0.5:
  * recall by possession duration;
  * recall by how much of it the ball was visible;
  * frame-level confusions: true D called a team, a true team called D, H/A swapped.

    python scripts/diagnose.py --tag f3
"""
import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import bench, config, model  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
BUCKETS_S = (0, 2, 4, 8, 16, 32, 1e9)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--mode", default="random")
    parser.add_argument("--switch", type=float, default=4.0)
    parser.add_argument("--dead", type=float, default=16.0)
    args = parser.parse_args()
    pm = bench._metrics()

    rows = []  # (duration_s, ball_visible_frac, matched)
    conf = np.zeros((3, 3), np.int64)
    n_pred = n_gt = 0
    for mid in bench.io_dfl.match_ids():
        path = os.path.join(config.CACHE_DIR, "cv_" + args.tag, "{}_{}.npy".format(mid, args.mode))
        m = bench.degraded_match(mid, 0, mode=args.mode)
        pred = model.decode(np.load(path), m.half, args.switch, args.dead)
        runs = pm.valid_runs(m.labels)
        gt_sp = pm.possession_spells(m.labels, runs)
        pr_sp = pm.possession_spells(pred, runs)
        flags = pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), 0.5)
        vis = np.isfinite(m.ball[:, 0])
        for (a, b, _), hit in zip(gt_sp, flags):
            rows.append(((b - a) / m.fps, vis[a:b].mean(), bool(hit)))
        g = m.graded
        for t in range(3):
            for p in range(3):
                conf[t, p] += int(np.sum((m.labels[g] == t) & (pred[g] == p)))
        n_pred += len(pr_sp)
        n_gt += len(gt_sp)

    rows = np.array(rows)
    print("[{} / {}] possessions predicted {} vs true {}; recall@IoU0.5 {:.3f}".format(
        args.tag, args.mode, n_pred, n_gt, rows[:, 2].mean()))
    print("\nrecall by true possession duration")
    for lo, hi in zip(BUCKETS_S[:-1], BUCKETS_S[1:]):
        sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        print("  {:>4.0f}-{:<4} s  n {:>5} ({:>4.1f}% of possessions)  recall {:.3f}".format(
            lo, "inf" if hi > 1e8 else int(hi), int(sel.sum()), 100 * sel.mean(),
            rows[sel, 2].mean() if sel.any() else float("nan")))
    print("\nrecall by share of the possession with the ball visible")
    for lo, hi in ((0, 0.01), (0.01, 0.5), (0.5, 0.99), (0.99, 1.01)):
        sel = (rows[:, 1] >= lo) & (rows[:, 1] < hi)
        print("  {:.2f}-{:.2f}  n {:>5}  recall {:.3f}".format(
            lo, min(hi, 1), int(sel.sum()), rows[sel, 2].mean() if sel.any() else float("nan")))
    names = "HAD"
    tot = conf.sum()
    print("\nframe confusion (rows true, cols predicted), % of graded frames")
    print("        " + "".join("{:>8}".format(n) for n in names))
    for t in range(3):
        print("  {:<5} ".format(names[t]) + "".join("{:8.2f}".format(100 * conf[t, p] / tot) for p in range(3)))
