"""Regenerate the submission file from the provided data. This is the command the README names.

    python scripts/make_submission.py --out submissions/<team>.csv

Steps: read the BAR-ATH tactical ATD; train v1 on all seven degraded DFL matches (both
degradation modes); predict and decode BAR-ATH; write the 199,319-row CSV; run the organisers'
validator on it. Deterministic end to end: fixed seeds, fixed model random_state.
"""
import argparse
import os
import subprocess
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import bench, config, io_atd, kickoff, model  # noqa: E402
from mtah.match import D, LETTERS, Match, write_labels_csv  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Decoder costs, chosen by leave-one-match-out CV on the DFL benchmark (scripts/cv.py) with feature
# version 4. Best in both degradation modes and in the animated-dead-ball check (tag j4_anim).
SWITCH_COST = 8.0
DEAD_COST = 32.0
MODES = ("random", "offpitch")


def load_bar_ath():
    path = os.path.join(config.CACHE_DIR, "BAR-ATH.npz")
    if os.path.isfile(path):
        return Match.load(path)
    match = io_atd.load_default()
    match.save(path)
    return match


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(config.ROOT, "submissions", "latest.csv"))
    parser.add_argument("--switch", type=float, default=SWITCH_COST)
    parser.add_argument("--dead", type=float, default=DEAD_COST)
    args = parser.parse_args()

    train = [bench.degraded_match(i, 0, mode=m) for i in bench.io_dfl.match_ids() for m in MODES]
    clf = model.train(train)
    match = load_bar_ath()
    proba = model.predict_proba(clf, match)
    # Before each half's published kick-off the ball is dead: pin it before decoding, so the
    # decoder places the first possession from the kick-off rather than smearing it backwards.
    pre = kickoff.apply(np.full(match.n_frames, -1, np.int8), match) == D
    proba[pre] = (1e-6, 1e-6, 1.0)
    labels = model.decode(proba, match.half, args.switch, args.dead)
    write_labels_csv(args.out, labels)

    counts = {c: int(np.sum(LETTERS[labels] == c)) for c in "HADX"}
    print("wrote {}  {}".format(os.path.relpath(args.out, config.ROOT), counts))
    validator = os.path.join(config.evaluation_dir(), "validate_submission.py")
    sys.exit(subprocess.call([sys.executable, validator, args.out]))
