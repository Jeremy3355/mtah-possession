"""Sanity checks of submission files on BAR-ATH that need no ground truth. Aggregates only.

  * possession and stoppage counts and durations, home share;
  * agreement with Smart Tagging: inside attacking phases tagged for one team (Build Up,
    Progression, Creating Chances, Ball in Final Third / in the Box, Attacking Transition), how
    often the file names that team. Smart Tagging is automatic and noisy, so this is a consistency
    check between two independent systems, not a score;
  * dead-ball share inside SET PIECES windows (which open a few seconds before the restart).

    python scripts/check_bar_ath.py submissions/v4.csv submissions/v3k.csv ...
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import config  # noqa: E402

ATTACK = {"build up", "progression", "creating chances", "ball in final third", "ball in the box",
          "attacking transition"}
TEAM = {"barcelona": 0, "atletic club": 1, "athletic club": 1}
CODES = {"H": 0, "A": 1, "D": 2, "X": 3}


def load(path):
    col = pd.read_csv(path, dtype={"label": str})["label"].str.strip().str.upper()
    return col.map(CODES).values.astype(np.int8)


def tagging():
    root = ET.parse(os.path.join(config.DATA_DIR, "tactical", "BAR-ATH_tactical_smart_tagging.xml")).getroot()
    team = np.full(config.ATD_N_FRAMES, -1, np.int8)
    setp = np.zeros(config.ATD_N_FRAMES, bool)
    for inst in root.iter("instance"):
        code = (inst.findtext("code") or "").strip().lower()
        a = int(round(float(inst.findtext("start")) * config.FPS))
        b = int(round(float(inst.findtext("end")) * config.FPS))
        tags = {l.findtext("group"): (l.findtext("text") or "").strip().lower() for l in inst.findall("label")}
        if code in ATTACK and tags.get("Team") in TEAM:
            team[a:b] = TEAM[tags["Team"]]
        if code == "set pieces":
            setp[a:b] = True
    return team, setp


def runs(codes, graded):
    out = []
    for first, last in config.ATD_HALVES:
        seg = codes[first:last + 1]
        edges = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], edges, [len(seg)]))
        out += [(bounds[i + 1] - bounds[i], seg[bounds[i]]) for i in range(len(bounds) - 1)]
    return out


if __name__ == "__main__":
    graded = np.zeros(config.ATD_N_FRAMES, bool)
    for first, last in config.ATD_HALVES:
        graded[first:last + 1] = True
    team, setp = tagging()
    print("{:<22} {:>6} {:>7} {:>6} {:>7} {:>7} {:>7} {:>8} {:>8}".format(
        "file", "poss", "mean s", "<4s %", "stops", "mean s", "H share", "ST team", "D in SP"))
    for path in sys.argv[1:]:
        c = load(path)
        r = runs(c, graded)
        poss = [n / config.FPS for n, k in r if k in (0, 1)]
        stops = [n / config.FPS for n, k in r if k == 2]
        alive = graded & (c < 2)
        tagged = graded & (team >= 0) & (c < 2)
        print("{:<22} {:>6} {:>7.1f} {:>6.0f} {:>7} {:>7.1f} {:>7.3f} {:>8.3f} {:>8.3f}".format(
            os.path.basename(path)[:22], len(poss), np.mean(poss), 100 * np.mean(np.array(poss) < 4),
            len(stops), np.mean(stops), np.mean(c[alive] == 0),
            np.mean(c[tagged] == team[tagged]), np.mean(c[graded & setp] == 2)))
