# Brief for the session on the Windows PC

You are continuing work on `mtah-possession`, our entry for More than a Hack 2027, Phase 1 (frame
possession H/A/D for BAR-ATH). Development so far happened on a Mac. This PC (Windows, Ryzen 7
9800X3D, RTX 5080 16 GB) takes over training.

## Hard rules (competition Terms and Conditions, plus our own policy)

- **Never** show, print, cat, head or screenshot competition data: tracking rows, labels, video
  frames or images. Scripts may print **aggregate output only** (counts, percentiles, scores). The
  T&C forbid making the data or any extract of it available to a cloud-hosted LLM.
- **Never** upload competition data anywhere: no cloud drive, no online service.
- No hand labels on BAR-ATH, and no use of the public 5-minute ground truth to train, tune or
  choose. It is a format check only, run once per finished version.
- The submission must be reproducible from the repository: the same file, byte for byte.
- Do not commit or push without the user's OK. Do not commit data, caches or CSVs (`.gitignore`
  already covers them).

## Step 1 — set up and reproduce v4 (do this first)

Follow `docs/SETUP_WINDOWS.md`, sections 1–4, inside WSL2 Ubuntu. The user copies the data bundle
into `more-than-a-hack-2027/data/`.

Done when:

- `git diff --stat calibration/` is empty after calibration;
- `md5sum submissions/v4.csv` gives `83fbddcecc98e7d10ed350e8f8c5683b`.

If the checksum differs, stop. Report which step diverges (calibration JSON or CSV) and how much:
run `scripts/check_bar_ath.py` on the CSV and give the count of frames that differ, as a number
only.

## Step 2 — GPU environment

Follow `docs/SETUP_WINDOWS.md`, section 5. Done when torch reports `cuda.is_available() == True` and
the RTX 5080.

## Step 3 — speed check of the current pipeline

Time `scripts/cv.py --tag pc_check --version 4 --switch 8 --dead 32 --test-suffix +anim`.

It must match `cache/cv_j4_anim` from the Mac: spellF1 random 0.4761 and offpitch 0.4767, using the
switch 8 / dead 32 row. Report the wall time.

## Where things stand (do not redo)

**v4 is the submission.** It is gradient boosting on identity-free features, trained on degraded
open DFL data (7 matches), with a 3-state Viterbi (switch 8, dead 32) and a published kick-off rule.

| test | spellF1 |
|---|---|
| CV on DFL, random | 0.540 |
| CV on DFL, offpitch | 0.580 |
| realistic, stoppage ball animated | 0.476 |
| organisers' baseline | 0.199 |

Tried and rejected, paired on 7 matches:

- visibility-dependent switch cost;
- finer one-sided windows (v5);
- two degradation seeds.

The technical report is `docs/technical_report.md`.

## Phase 2 ideas, after steps 1–3 and only if the user asks

1. **Ball detection on the tactical video** to fill the ATD's ball gaps (41% of graded frames).
   - Use a public detector (for example RT-DETR) running locally on the GPU.
   - Fine-tune it with pseudo-labels from the ATD's own ball, projected through the homography.
   - Validate on hidden ATD-visible frames.
   - Estimate the score effect by simulating the recovered gaps on the DFL benchmark.
2. **A sequence model** (temporal transformer or CNN over the identity-free player set) instead of
   gradient boosting plus Viterbi, compared paired against v4 in the same CV.

Keep GPU inference deterministic, or round its outputs, so the submission stays byte-identical.
