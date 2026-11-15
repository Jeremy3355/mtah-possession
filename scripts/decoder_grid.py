"""Decoder-only experiments on cached CV probabilities: no retraining.

Compares the uniform team-change cost against a visibility-dependent one (cheaper where the ball
was seen), paired per held-out match.

    python scripts/decoder_grid.py --tag j4_anim --suffix +anim
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--dead", type=float, default=32.0)
    parser.add_argument("--base-switch", type=float, default=8.0)
    parser.add_argument("--visible", type=float, nargs="+", default=[3.0, 4.0, 6.0, 8.0])
    parser.add_argument("--hidden", type=float, nargs="+", default=[8.0, 12.0, 16.0])
    args = parser.parse_args()
    pm = bench._metrics()
    ids = bench.io_dfl.match_ids()
    res = {}
    for mode in MODES:
        for mid in ids:
            m = bench.degraded_match(mid, 0, mode=mode + args.suffix)
            proba = np.load(os.path.join(config.CACHE_DIR, "cv_" + args.tag, "{}_{}.npy".format(mid, mode)))
            base = pm.evaluate(m.labels, model.decode(proba, m.half, args.base_switch, args.dead))
            res.setdefault((mode, "base"), []).append(base["spell_f1_mean"])
            for sv in args.visible:
                for sh in args.hidden:
                    st = model.visibility_switch_cost(m.ball, sv, sh)
                    r = pm.evaluate(m.labels, model.decode(proba, m.half, args.base_switch, args.dead, st))
                    res.setdefault((mode, sv, sh), []).append(r["spell_f1_mean"])
        print("done", mode, flush=True)
    for mode in MODES:
        b = np.array(res[(mode, "base")])
        print("\n[{}{}] {}  uniform switch {:.0f}: spellF1 {:.4f}".format(args.tag, args.suffix, mode, args.base_switch, b.mean()))
        print("  visible\\hidden " + " ".join("{:>16}".format(h) for h in args.hidden))
        for sv in args.visible:
            cells = []
            for sh in args.hidden:
                d = np.array(res[(mode, sv, sh)]) - b
                cells.append("{:.4f} {:+.3f} {}/7".format(np.mean(res[(mode, sv, sh)]), d.mean(), int((d > 0).sum())))
            print("  {:>14} ".format(sv) + " ".join("{:>16}".format(c) for c in cells))
