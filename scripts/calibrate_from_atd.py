"""Measure the BAR-ATH ATD's noise as aggregate distributions and write calibration/atd_noise.json.

Prints a short summary only: quantiles and means, no frames or positions.

    python scripts/calibrate_from_atd.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import config, io_atd  # noqa: E402
from mtah.stats import noise_stats  # noqa: E402

if __name__ == "__main__":
    cache = os.path.join(config.CACHE_DIR, "BAR-ATH.npz")
    if os.path.isfile(cache):
        from mtah.match import Match
        match = Match.load(cache)
    else:
        match = io_atd.load_default()
        match.save(cache)
    stats = noise_stats(match)
    os.makedirs(os.path.dirname(config.CALIBRATION_PATH), exist_ok=True)
    with open(config.CALIBRATION_PATH, "w") as handle:
        json.dump(stats, handle, indent=1)

    def pct(key):
        qs = stats[key]
        return "p10 {:.2f}  p50 {:.2f}  p90 {:.2f}  p99 {:.2f}".format(qs[10], qs[50], qs[90], qs[99])

    print("ball visible frac      %.3f" % stats["ball_visible_frac"])
    print("ball visible runs (s)  " + pct("ball_visible_run_s"))
    print("ball missing runs (s)  " + pct("ball_missing_run_s"))
    print("slot visible runs (s)  " + pct("slot_visible_run_s"))
    if stats["slot_gap_s"]:
        print("slot interior gaps (s) " + pct("slot_gap_s"))
    print("players visible mean   %.2f  per team %s" % (
        stats["players_visible_mean"], [round(v, 2) for v in stats["players_visible_per_team_mean"]]))
    print("jitter (m)  ball %.3f  players %.3f" % (stats["ball_jitter_m"], stats["player_jitter_m"]))
    print("wrote", os.path.relpath(config.CALIBRATION_PATH, config.ROOT))
