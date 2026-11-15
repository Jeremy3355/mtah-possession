"""Export matches, labels and homographies to compact binaries for the local viewer.

Everything goes to viewer/data/ (gitignored): the BAR-ATH tracking is competition data and must stay
on this machine. Prints only file names and sizes.

    python scripts/export_viewer.py

Binary layout (little-endian):
  players.bin  uint16 (T, S, 2)  normalised coord q = round((v + PAD) / SPAN * 65534), 65535 = missing
  ball.bin     uint16 (T, 2)     same encoding
  <label>.bin  uint8  (T,)       0 = H, 1 = A, 2 = D, 3 = X
  homography.bin float32 (T, 9)  pitch(0-1) -> pixels, NaN where invalid  (BAR-ATH only)
"""
import glob
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import bench, config, model  # noqa: E402
from mtah.match import Match  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = os.path.join(config.ROOT, "viewer", "data")
PAD, SPAN = 0.15, 1.3
DFL_MATCHES = ("DFL-MAT-J03WMX", "DFL-MAT-J03WN1")


def quantise(a):
    q = np.round((a + PAD) / SPAN * 65534)
    q = np.where(np.isfinite(q), np.clip(q, 0, 65534), 65535)
    return q.astype("<u2")


def read_labels(path):
    import pandas as pd
    codes = {"H": 0, "A": 1, "D": 2, "X": 3}
    col = pd.read_csv(path, dtype={"label": str})["label"].str.strip().str.upper()
    return col.map(codes).fillna(3).astype(np.uint8).values


def write(folder, match, labels, extra_meta=None, homography=None):
    os.makedirs(folder, exist_ok=True)
    quantise(match.players).tofile(os.path.join(folder, "players.bin"))
    quantise(match.ball).tofile(os.path.join(folder, "ball.bin"))
    tracks = []
    for key, (title, codes) in labels.items():
        np.asarray(codes, np.uint8).tofile(os.path.join(folder, key + ".bin"))
        tracks.append({"key": key, "title": title})
    if homography is not None:
        homography.astype("<f4").tofile(os.path.join(folder, "homography.bin"))
    halves = []
    for h in (1, 2):
        idx = np.flatnonzero(match.half == h)
        if len(idx):
            halves.append([int(idx[0]), int(idx[-1])])
    meta = {"folder": os.path.basename(folder), "name": match.name, "n_frames": match.n_frames, "n_slots": int(match.players.shape[1]),
            "team": match.team.tolist(), "pitch": list(match.pitch), "fps": match.fps,
            "pad": PAD, "span": SPAN, "halves": halves, "labels": tracks,
            "has_homography": homography is not None}
    meta.update(extra_meta or {})
    with open(os.path.join(folder, "meta.json"), "w") as handle:
        json.dump(meta, handle)
    size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder))
    print("  {:<28} {:>7.1f} MB".format(os.path.basename(folder), size / 1e6))
    return meta


def bar_ath():
    match = Match.load(os.path.join(config.CACHE_DIR, "BAR-ATH.npz"))
    labels = {}
    for path in sorted(glob.glob(os.path.join(config.ROOT, "submissions", "*.csv"))):
        key = "pred_" + os.path.basename(path)[:-4]
        labels[key] = ("Predicción " + os.path.basename(path)[:-4], read_labels(path))
    gt = os.path.join(config.evaluation_dir(), "gt", "BAR-ATH_possession_gt_public.csv")
    labels["gt_public"] = ("GT público (5 min)", read_labels(gt))
    labels["baseline"] = ("Baseline smooth50",
                          read_labels(os.path.join(config.evaluation_dir(), "results", "baseline_smooth50.csv")))

    import pandas as pd
    table = pd.read_csv(os.path.join(config.DATA_DIR, "tactical", "BAR-ATH_tactical_homography.csv"))
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    hom = table[cols].values.astype(np.float32)
    video = os.path.relpath(os.path.join(config.DATA_DIR, "tactical", "BAR-ATH_tactical.mp4"),
                            config.WORKSPACE)
    return write(os.path.join(OUT, "BAR-ATH"), match, labels,
                 {"video": "/" + video.replace(os.sep, "/"), "title": "BAR-ATH (ATD táctico)"}, hom)


def dfl(match_id):
    degraded = bench.degraded_match(match_id, 0, mode="random")
    clean = bench.clean_match(match_id)
    labels = {"gt": ("Verdad DFL", degraded.labels)}
    probs = os.path.join(config.CACHE_DIR, "cv_j4", "{}_random.npy".format(match_id))
    if os.path.isfile(probs):
        labels["pred_cv_j4"] = ("Predicción v4 (CV, partido no visto)",
                                model.decode(np.load(probs), degraded.half, 8.0, 32.0))
    metas = [write(os.path.join(OUT, match_id), degraded, labels,
                   {"title": "{} · DFL degradado (como ATD)".format(match_id)})]
    metas.append(write(os.path.join(OUT, match_id + "_clean"), clean, labels,
                       {"title": "{} · DFL limpio".format(match_id)}))
    return metas


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("exporting to", os.path.relpath(OUT, config.ROOT))
    index = [bar_ath()]
    for match_id in DFL_MATCHES:
        index.extend(dfl(match_id))
    with open(os.path.join(OUT, "index.json"), "w") as handle:
        json.dump([{"folder": m["folder"], "title": m["title"]} for m in index], handle)
    print("done")
