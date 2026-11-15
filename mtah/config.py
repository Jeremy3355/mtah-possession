"""Paths and constants. Everything is overridable by environment variable."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(ROOT)

# The organisers' challenge repository: its scorer is the official one, so we import it, never copy.
CHALLENGE_REPO = os.environ.get("MTAH_REPO") or os.path.join(WORKSPACE, "more-than-a-hack-2027")
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(CHALLENGE_REPO, "data")
DFL_DIR = os.environ.get("DFL_DIR") or os.path.join(WORKSPACE, "external_data", "dfl")
CACHE_DIR = os.environ.get("MTAH_CACHE") or os.path.join(ROOT, "cache")
BENCH_DIR = os.environ.get("MTAH_BENCH") or os.path.join(ROOT, "bench")
CALIBRATION_PATH = os.path.join(ROOT, "calibration", "atd_noise.json")

FPS = 25

# BAR-ATH tactical video, from the challenge repository's games.py / DATA.md.
ATD_N_FRAMES = 199319
ATD_HALVES = ((11185, 90651), (110808, 190197))  # inclusive video frames
ATD_PITCH = (105.3, 68.0)

# Frames of X inserted between the halves of a DFL match, so the scorer treats them as two runs.
DFL_HALF_GAP = 500


def evaluation_dir():
    return os.path.join(CHALLENGE_REPO, "evaluation")
