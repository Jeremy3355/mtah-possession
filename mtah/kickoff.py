"""Everything graded before a half's kick-off is dead ball.

BAR-ATH's second half is graded from video frame 110,808 but does not kick off until 113,742. The
organisers publish both numbers ("One trap worth knowing", FAQ.md; also the starter's
load_smart_tagging.py), so they are match metadata rather than labels. Every frame in between is D.

The DFL training data starts each half at its kick-off, so the classifier never sees the "teams
walking out" state and cannot be expected to call it; this rule covers it.

Detecting the kick-off from the tracking instead was tried and dropped: around the halfway line the
ATD's own-half shares never reach 1.0, so a formation rule was not reliable.
"""
import numpy as np

from .match import D

# Published kick-off video frames for BAR-ATH (DATA.md, FAQ.md).
BAR_ATH_KICKOFFS = {1: 11185, 2: 113742}


def apply(labels, match, kickoffs=BAR_ATH_KICKOFFS):
    """Copy of labels with D on every graded frame of a half before its kick-off."""
    out = labels.copy()
    frames = np.arange(match.n_frames)
    for half, frame in kickoffs.items():
        out[(match.half == half) & (frames < frame)] = D
    return out
