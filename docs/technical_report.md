# More than a Hack 2027 — Phase 1: Ball Possession
## Technical report — team `<TEAM_NAME>`

Repository: `<PUBLIC_REPO_URL>` · Submission: `<TEAM_NAME>.csv` (199,319 rows, validated)

---

## 1. Approach in one paragraph

We label every frame H / A / D with a gradient-boosting classifier on identity-free features of the
automatic tracking, followed by a three-state Viterbi decoder that turns per-frame probabilities into
whole possessions. The challenge gives no labels for training, and its public 5 minutes may not be
used for tuning. So the model is trained and every choice is made on **another, labelled source**:
the DFL open Bundesliga dataset (7 matches). We **degrade** that data until it looks like the BAR-ATH
automatic tracking, using only aggregate statistics measured on the ATD. All model selection is
leave-one-match-out cross-validation on that benchmark, scored with the organisers' own
`possession_metrics.py`.

## 2. Why this design

- **No labels on the target match, and a public window too small to choose with.** The organisers
  show that tuning one window on the public 5 minutes already picks the wrong value. A labelled
  external source was the only way to make decisions measurable.
- **The ranking metric rewards whole possessions** (spellF1, IoU 0.3–0.7). A per-frame classifier
  flickers, and a centred majority vote lags every boundary. A Viterbi with explicit costs for a
  team change and for entering or leaving a stoppage produces clean segments with no lag.
- **The ball is missing on 41% of graded frames, in long stretches** (median gap 10.7 s; gaps over
  10 s cover 35% of graded frames). Possession during gaps has to come from what the two teams are
  doing, not from the ball.

## 3. Data used, and data dropped

| asset | used? | how |
|---|---|---|
| Tactical ATD (players + ball) | **yes** | the only per-frame input of the model |
| Tactical homography | viewer only | projects the tracking onto the video for visual debugging |
| Smart Tagging | checks only | agreement check (sec. 7); not a model input, because the DFL has no equivalent to train it on |
| Tactical video | viewer only | inspection; not decoded by the model |
| Broadcast video / ATD / homography | **dropped** | not synchronised with the tactical timeline; mapping it was out of scope for the time available |
| Public 5-minute ground truth | pipeline check only | scored once per finished version to confirm the format and the label meaning; never used to train, tune or choose |
| ATD speed channel `s` | **dropped** | not a ball speed (median 1.2 vs 5.5 m/s computed from positions); speeds are computed from positions |

### External training data

- **What:** the DFL open data, "An integrated dataset of spatiotemporal and event data in elite
  soccer" (Bassek, Rein, Weber & Memmert, *Scientific Data* 2025). It is licensed CC-BY 4.0 and
  hosted on figshare (doi 10.6084/m9.figshare.28196177).
- **Size:** 7 full matches of the 1st and 2nd Bundesliga, tracked by TRACAB at 25 Hz.
- **Labels:** `BallPossession` (home/away) and `BallStatus` (alive/dead) on every frame, as supplied
  by the DFL. We did not label anything ourselves. Nothing comes from the competition match.
- **Download:** `scripts/download_dfl.sh` fetches the files and checks each MD5 against figshare.

## 4. Making the DFL look like the ATD (`mtah/degrade.py`)

Calibration (`scripts/calibrate_from_atd.py`) measures only distribution quantiles on the ATD and
stores them in `calibration/atd_noise.json`:

- the ball's visible and missing run lengths;
- slot tracklet lengths;
- the number of visible players;
- how often the visible ball is off the pitch.

The degradation reproduces each of these:

- **Ball gaps:** the ball alternates visible and missing, with run lengths drawn from the ATD's own
  distributions. The overall visible share matches (0.593).
- **No identities:** every player's track is cut into ATD-length tracklets and spread over anonymous
  slots of the right team, exactly as ATD slots behave.
- **Visibility:** players are hidden often enough to match the ATD's 20.5 visible players on average.
- **Two modes**, to test robustness: gaps independent of the game state (`random`), and gaps biased
  towards an off-pitch ball (`offpitch`). Every version had to win in both.

**A leak we found and removed.** The DFL holds the ball perfectly still during every stoppage: 100%
of dead frames have zero ball displacement, against about 1% of alive frames. The ATD ball keeps
moving through stoppages. On the benchmark, "the ball is still" was therefore a near-perfect
dead-ball detector that does not exist on BAR-ATH.

- **Removed:** every ball-motion feature (ball speed, ball displacement).
- **Measured:** to see how much each model still relied on the leak, we re-scored every model on
  test matches whose stoppage ball is animated (carried and handed on by nearby players). Those are
  the "realistic" numbers in sec. 6.

## 5. Model

**Features** (`mtah/features.py`) are computed from positions only, in each team's attacking
direction (inferred per half from team centroids):

- distance from the ball to the nearest home and nearest away player, and the difference;
- the number of players within 3 m of the ball;
- the ball's distance to the lines (with an off-pitch ball, more than 2 m out, treated as unseen);
- the mean collective speed per team, the share of slow and fast players, and own-half shares;
- team centroid, spread, and how fast each team advances;
- time since and until the ball was last seen, and which team was nearest then.

Each is averaged over centred windows of ±0.25, 1, 3, 10 and 20 s. Seven key signals also get
one-sided windows: the previous and next 0.5 s and 2 s, plus their difference. These locate a
hand-over that centred windows blur.

Tracking outside play (pre-match, halftime, before a kick-off) is blanked before features are
computed, and no gap fill crosses a half.

**Classifier:**

- scikit-learn `HistGradientBoostingClassifier`, with 250 iterations, learning rate 0.08, 31 leaves,
  at least 200 samples per leaf, L2 1.0, and `random_state=0`;
- trained on all 7 DFL matches in both degradation modes, using every third frame;
- **team-symmetric**: every match is also used with home and away swapped, and predictions average
  the two views. Nothing is learned about "the home side".

**Decoder.** A three-state Viterbi per half. A team change costs 8 and entering or leaving D costs 32,
in log-probability units. Both costs were chosen by cross-validation (sec. 6).

**Kick-off rule** (`mtah/kickoff.py`). BAR-ATH's second half is graded from frame 110,808 but kicks
off at 113,742, both published by the organisers in FAQ.md and DATA.md. The DFL data starts every
half at its kick-off, so the classifier has never seen "teams walking out". Frames before each
half's published kick-off are fixed to D before decoding. We tried detecting the kick-off from the
tracking instead and dropped it: ATD own-half shares never reach 1.0 around the halfway line.

## 6. Possession rules, and results

**Rules as implemented.**

- **Dead ball:** whatever the classifier and decoder call D, plus the pre-kick-off frames.
- **Loose and contested balls:** possession stays with the current team unless the evidence for the
  other team outweighs the switch cost. A deflection or an unconverted aerial duel is too short to
  pay for two switches, so it does not change possession. A real interception does.
- **Restarts:** leaving D costs the same for either team, so the first team clearly in control after
  a stoppage gets the ball.

The granularity this produces follows the DFL's definition. The DFL's possessions are finer than the
public BAR-ATH window suggests (19–26 per 5 minutes against 17), so we did not tune the switch cost
towards more possessions.

**Cross-validation on the DFL benchmark** (spellF1, mean of 7 held-out matches; official metric
code):

| version | random | offpitch | realistic (stoppage ball moving) |
|---|---|---|---|
| organisers' baseline (smooth50) | 0.199 | 0.248 | – |
| v1: features + decoder | 0.482 | 0.517 | – |
| v3: + dead-ball features, ball-motion removed | 0.497 | 0.527 | 0.437 |
| **v4: + one-sided windows, costs 8 / 32** | **0.540** | **0.580** | **0.476** |

v4 against v3 is +0.025 on the realistic test, and v4 wins on **7 of 7** held-out matches in every
mode. We read the realistic column as the honest estimate; the gap to the other columns is what the
frozen-ball leak was worth.

**Tried after v4 and not kept** (realistic test, paired per held-out match, both modes):

| change | random | offpitch | verdict |
|---|---|---|---|
| team-change cost lower where the ball was seen (best of a 4×3 grid) | +0.003 | +0.003 | within noise, not kept |
| finer one-sided windows (0.12 s and 0.25 s added) | −0.002 | +0.001 | no gain |
| two degradation seeds per training match | −0.000 | −0.000 | no gain |

The per-match differences in each row range from −0.02 to +0.01, so none of these is
distinguishable from v4. We kept the simpler model.

**Where the remaining errors are** (v3 diagnostics):

- **Short possessions:** recall at IoU 0.5 is 0.09 for possessions under 2 s and 0.29 for 2–4 s,
  against 0.77–0.88 for those over 8 s. Short possessions are about half of the DFL's.
- **Team confusions:** H↔A errors cover about 7% of frames, against about 4% for confusions
  involving D.

## 7. Checks on BAR-ATH (no ground truth involved)

| | possessions | mean | < 4 s | stoppages | agrees with Smart Tagging team in attacking phases |
|---|---|---|---|---|---|
| organisers' baseline | 594 | 7.2 s | 45% | 133 | 0.70 |
| **v4 (submitted)** | **308** | **11.0 s** | **37%** | **113** | **0.83** |

Smart Tagging is automatic and noisy, so this agreement is a consistency check between two
independent systems, not a score. The public 5 minutes, scored once as a format check after v4 was
fixed, gave spellF1 0.585 and macroF1 0.852.

## 8. Limitations, and what we would do with more time

- **Domain gap.** Everything is learnt on another league's tracking, made to look like the ATD.
  - The degradation does not yet reproduce wrong-team or duplicated slots: the ATD flickers more,
    with 28.7 raw H/A switches per visible-ball minute against 21.2 on the degraded DFL.
  - It does not reproduce the ATD's ball behaviour during stoppages.
- **Definition gap.** The DFL's possession definition is finer than the challenge's. Our granularity
  is set by the decoder costs, not learnt from BAR-ATH.
- **Short possessions** are the largest remaining loss.
- **Broadcast feed and video unused.** A ball re-detected from video would shorten the 41% of frames
  with no ball, which is the single biggest limitation of the input.
- **Only 7 training matches.** More open matches with frame-level possession, or several degradation
  seeds, would reduce variance.

## 9. Reproduction

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.14
bash scripts/download_dfl.sh                                          # DFL open data, MD5-checked
.venv/bin/python scripts/calibrate_from_atd.py                        # ATD noise quantiles
.venv/bin/python scripts/make_submission.py --out submissions/<TEAM_NAME>.csv
```

The organisers' repository goes next to ours, with the match bundle in its `data/`. The pipeline is
deterministic: run from an empty cache, it regenerates the submitted file byte for byte (about
10 minutes on an 8-core laptop).

**No competition data is in the repository.** `calibration/atd_noise.json` holds aggregate quantiles
only.

Use of LLMs: an AI coding assistant (Claude) helped write code and this document. No competition
data was uploaded to it: its scripts printed aggregate statistics and scores only.
