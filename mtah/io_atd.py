"""Read the Metrica FIFA/EPTS ATD (metadata XML + raw text) into a Match.

Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>" with frame_id from 1, NaN for missing. Every
separator is turned into a comma so the whole file parses in one numpy call.
"""
import xml.etree.ElementTree as ET

import numpy as np

from . import config
from .match import Match


def team_of_slots(meta_path):
    """0 for home slots, 1 for away; the first Team in the metadata is home (FIFA/EPTS)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}".format(meta_path))
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def load(meta_path, raw_path, n_frames=config.ATD_N_FRAMES, halves=config.ATD_HALVES,
         pitch=config.ATD_PITCH, name="BAR-ATH"):
    team = team_of_slots(meta_path)
    n_slots = len(team)
    with open(raw_path) as handle:
        text = handle.read()
    lines = text.replace(":", ",").replace(";", ",").split()
    width = 1 + 3 * n_slots + 3
    flat = np.array(",".join(lines).split(","), dtype=np.float64)
    if flat.size != width * len(lines):
        raise ValueError("expected {} fields on each of {} lines".format(width, len(lines)))
    table = flat.reshape(len(lines), width)

    frame = table[:, 0].astype(np.int64) - 1  # frame_id counts from 1, video frames from 0
    players = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    groups = table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    players[frame] = groups[:, :, :2]
    ball[frame] = table[:, -3:-1]
    # Note: the third ball channel ('s') is not a ball speed; compute speed from positions.

    half = np.zeros(n_frames, np.int8)
    for index, (first, last) in enumerate(halves, start=1):
        half[first:last + 1] = index
    return Match(name=name, ball=ball, players=players, team=team, half=half, pitch=pitch,
                 fps=config.FPS)


def load_default():
    import os
    folder = os.path.join(config.DATA_DIR, "tactical")
    return load(os.path.join(folder, "BAR-ATH_tactical_atd.xml"),
                os.path.join(folder, "BAR-ATH_tactical_atd.txt"))
