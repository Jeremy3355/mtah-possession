# mtah-possession

Our entry for **More than a Hack 2027, Phase 1**: a frame-level ball-possession label (H / A / D)
for every frame of BAR-ATH, built from the automatic tracking (ATD).

The model is trained **only on open data from other matches**: the DFL/Bundesliga open dataset
(Bassek et al. 2025, CC-BY 4.0). That data is degraded until it looks like the BAR-ATH ATD. No
label of the competition match is used for training, tuning or model selection.

## Reproduce the submission

Requirements: Python 3.14 (see `.python-version`), about 3 GB of disk for the DFL files, and about
10 minutes.

```bash
# 1. environment
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. data
#    - the organisers' repository next to this one: ../more-than-a-hack-2027
#    - the match bundle unpacked into ../more-than-a-hack-2027/data   (gated, see their DATA.md)
#    - the DFL open data (21 files, MD5-checked):
bash scripts/download_dfl.sh                      # -> ../external_data/dfl

# 3. noise calibration from the ATD (aggregate quantiles only)
.venv/bin/python scripts/calibrate_from_atd.py    # -> calibration/atd_noise.json

# 4. train on degraded DFL, predict BAR-ATH, write and validate the CSV
.venv/bin/python scripts/make_submission.py --out submissions/trackless.csv
```

Everything is deterministic: fixed seeds for the degradation and a fixed `random_state` for the
model. Run from an empty cache, step 4 reproduces the same file byte for byte.

Paths can be overridden with `MTAH_REPO`, `MTAH_DATA_DIR`, `DFL_DIR` and `MTAH_CACHE`.

## How it works

1. **One common format** (`mtah/match.py`). Both the BAR-ATH ATD and the DFL data become a `Match`:
   ball and slots in normalised 0–1 coordinates, the team of each slot, and the half.
2. **Degrading the DFL into ATD look-alikes** (`mtah/degrade.py`), calibrated on aggregate ATD
   statistics (`mtah/stats.py`):
   - ball gaps with the ATD's own visible and missing run lengths;
   - each player's track cut into ATD-length tracklets on anonymous slots;
   - player visibility matched to the ATD.
   
   There are two modes: random gaps, and gaps biased to an off-pitch ball.
3. **Identity-free features** (`mtah/features.py`):
   - nearest-player distance per team;
   - collective speed, stillness and own-half shares;
   - team centroids and how far each team is advancing;
   - time since and until the ball was last seen, and who was nearest then.
   
   Each is taken over centred windows of 0.25–20 s and one-sided before/after windows. Signals
   that behave differently in the ATD and in the DFL are excluded: ball speed and ball
   displacement, because the DFL freezes the ball during stoppages and the ATD does not.
4. **Classifier.** Gradient boosting with H/A/D per frame, trained team-symmetrically (every match
   also with home and away swapped).
5. **Decoder** (`mtah/model.py`). A three-state Viterbi with costs for changing team and for
   entering or leaving a stoppage. It turns the per-frame probabilities into whole possessions
   without lagging the boundaries.
6. **Kick-off rule** (`mtah/kickoff.py`). Graded frames before a half's published kick-off are
   dead ball. That is frames 110,808–113,741 in BAR-ATH (FAQ.md).

## Evaluation (DFL benchmark, leave-one-match-out, official metric code)

```bash
.venv/bin/python scripts/bench_baselines.py              # organisers' baseline on the benchmark
.venv/bin/python scripts/cv.py --tag v4 --version 4       # our model, 7 folds, both modes
.venv/bin/python scripts/diagnose.py --tag v4             # where possessions are lost
.venv/bin/python scripts/check_bar_ath.py submissions/*.csv   # BAR-ATH sanity checks (no GT)
```

## Local viewer

```bash
.venv/bin/python scripts/export_viewer.py                 # -> viewer/data (gitignored)
VIEWER_PORT=8790 .venv/bin/python scripts/serve_viewer.py # http://127.0.0.1:8790/mtah-possession/viewer/
```

The viewer shows the tactical video with the tracking projected through the homography, a 2D
recreation, and the label timelines (for BAR-ATH: the public 5 minutes and the organisers'
baseline only, never our predictions). It is for looking at the data; nothing is tuned on it. It
runs on 127.0.0.1 only.

## Data policy

No competition data is committed: `cache/`, `bench/`, `viewer/data/` and every `*.csv` are
gitignored. `calibration/atd_noise.json` holds distribution quantiles only, which cannot
reconstruct the tracking.

Data credit: DFL Deutsche Fußball Liga; Bassek, Rein, Weber & Memmert (2025), "An integrated
dataset of spatiotemporal and event data in elite soccer", *Scientific Data*.
