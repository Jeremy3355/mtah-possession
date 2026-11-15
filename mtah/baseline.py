"""The organisers' reference baseline, re-implemented on a Match: nearest tracked player to the
ball, D where the ball is not tracked, then a centred majority vote. Used to check that the
degraded DFL benchmark behaves like the real match (same ranking of the four variants)."""
import numpy as np

from .match import A, D, H, X

NOBODY = -1


def nearest_player_codes(match):
    ball = match.ball_m()
    players = match.players_m()
    d2 = ((players - ball[:, None, :]) ** 2).sum(axis=2)
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    best = d2.argmin(axis=1)
    has = np.isfinite(d2.min(axis=1))
    codes = np.where(has, match.team[best], NOBODY).astype(np.int8)
    return codes


def smooth(codes, window):
    if window <= 0:
        return np.where(codes == 0, H, np.where(codes == 1, A, D)).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    out = np.full(len(codes), D, np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], H, A)
    return out


def predict(match, window):
    labels = smooth(nearest_player_codes(match), window)
    labels[~match.graded] = X
    return labels
