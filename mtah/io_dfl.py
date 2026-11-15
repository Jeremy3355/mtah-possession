"""Read one match of the DFL open data (Bassek et al., 2025, CC-BY 4.0) into a labelled Match.

Positions XML: one FrameSet per (GameSection, person); the BALL FrameSet carries per-frame
BallPossession (1 = home, 2 = away) and BallStatus (0 = dead, 1 = alive). Coordinates are metres
with the origin at the centre spot; they are normalised here to the ATD's 0-1 convention.

The two halves are laid end to end with DFL_HALF_GAP frames of X between them, so the scorer sees
two separate runs exactly as it does for BAR-ATH.
"""
import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from . import config
from .match import A, D, H, X, Match

SECTIONS = ("firstHalf", "secondHalf")
_FRAME = re.compile(rb'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"([^>]*)/>')
_POSS = re.compile(rb'BallPossession="(\d)"')
_STATUS = re.compile(rb'BallStatus="(\d)"')
_FRAMESET = re.compile(rb'<FrameSet ([^>]*)>')
_ATTR = re.compile(rb'(\w+)="([^"]*)"')


def home_team_id(info_path):
    for team in ET.parse(info_path).getroot().iter("Team"):
        if team.get("Role") == "home":
            return team.get("TeamId")
    raise ValueError("no home team in {}".format(info_path))


def match_files(match_id, folder=config.DFL_DIR):
    def one(pattern):
        found = glob.glob(os.path.join(folder, pattern.format(match_id)))
        if len(found) != 1:
            raise IOError("expected one file for {} in {}".format(pattern, folder))
        return found[0]
    return (one("DFL_02_01_matchinformation_*_{}.xml"),
            one("DFL_04_03_positions_raw_observed_*_{}.xml"))


def match_ids(folder=config.DFL_DIR):
    paths = glob.glob(os.path.join(folder, "DFL_04_03_positions_raw_observed_*.xml"))
    return sorted(os.path.basename(p)[:-4].split("_")[-1] for p in paths)


def _framesets(path):
    """Yield (attrs, body bytes) per FrameSet. Regex over bytes: ~10x faster than ElementTree."""
    with open(path, "rb") as handle:
        data = handle.read()
    starts = [m for m in _FRAMESET.finditer(data)]
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(data)
        attrs = {k.decode(): v.decode() for k, v in _ATTR.findall(m.group(1))}
        yield attrs, data[m.end():end]


def load(match_id, folder=config.DFL_DIR):
    info_path, pos_path = match_files(match_id, folder)
    home = home_team_id(info_path)

    ball = {}    # section -> (N, x, y, poss, status)
    people = {}  # (team, person) -> {section: (N, x, y)}
    pitch = (105.0, 68.0)
    for attrs, body in _framesets(pos_path):
        section, team = attrs.get("GameSection"), attrs.get("TeamId", "")
        if section not in SECTIONS or team.lower() == "referee":
            continue
        rows = _FRAME.findall(body)
        n = np.array([int(r[0]) for r in rows], np.int64)
        xy = np.array([(float(r[1]), float(r[2])) for r in rows], np.float64).reshape(-1, 2)
        if team.upper() == "BALL":
            poss = np.array([int(_POSS.search(r[3]).group(1)) for r in rows], np.int8)
            status = np.array([int(_STATUS.search(r[3]).group(1)) for r in rows], np.int8)
            ball[section] = (n, xy, poss, status)
        else:
            people.setdefault((0 if team == home else 1, attrs["PersonId"]), {})[section] = (n, xy)

    # Timeline: half 1, a gap of X, half 2. Frame numbers inside a half come from the BALL set.
    offsets, total = {}, 0
    for index, section in enumerate(SECTIONS):
        n = ball[section][0]
        if not np.array_equal(n, np.arange(n[0], n[0] + len(n))):
            raise ValueError("{} {}: ball frames are not contiguous".format(match_id, section))
        offsets[section] = total - n[0]
        total += len(n) + (config.DFL_HALF_GAP if index == 0 else 0)

    slots = sorted(people)
    team = np.array([t for t, _ in slots], np.int8)
    players = np.full((total, len(slots), 2), np.nan, np.float32)
    ball_xy = np.full((total, 2), np.nan, np.float32)
    labels = np.full(total, X, np.int8)
    half = np.zeros(total, np.int8)
    length, width = pitch

    def norm(xy):
        return np.column_stack((xy[:, 0] / length + 0.5, xy[:, 1] / width + 0.5))

    for index, section in enumerate(SECTIONS, start=1):
        n, xy, poss, status = ball[section]
        rows = n + offsets[section]
        ball_xy[rows] = norm(xy)
        half[rows] = index
        labels[rows] = np.where(status == 0, D, np.where(poss == 1, H, A))
    for slot, key in enumerate(slots):
        for section, (n, xy) in people[key].items():
            rows = n + offsets[section]
            ok = (rows >= 0) & (rows < total)
            players[rows[ok], slot] = norm(xy[ok])

    return Match(name=match_id, ball=ball_xy, players=players, team=team, half=half,
                 pitch=pitch, fps=config.FPS, labels=labels)
