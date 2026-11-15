"""Per-frame features of a Match, identity-free and source-agnostic.

Every feature is computed from positions only: no player identities, no ATD speed channel, nothing
specific to one provider. Coordinates are first rotated so that each team's attacking direction is
known (from where its centroid sits in each half), which makes the features mean the same thing in
both halves and for both sides.

Raw per-frame signals are then averaged over centred windows of several widths, NaN-aware, so the
classifier sees both the instant and its context. Gradient boosting handles the remaining NaN.
"""
import numpy as np

WINDOWS = (6, 25, 75, 250)
WINDOWS_V2 = (6, 25, 75, 250, 500)
OFFPITCH_NAN_M = 2.0
MAX_PLAYER_SPEED = 12.0  # m/s; faster is a tracklet jump, not a sprint
MAX_BALL_SPEED = 40.0
GAP_CAP_S = 30.0


def _attack_sign(match):
    """(T,) +1 where home attacks towards x = L, -1 where towards x = 0. 0 outside play."""
    sign = np.zeros(match.n_frames, np.float32)
    for h in (1, 2):
        rows = match.half == h
        if not rows.any():
            continue
        home = np.nanmean(match.players[rows][:, match.team == 0, 0])
        away = np.nanmean(match.players[rows][:, match.team == 1, 0])
        sign[rows] = 1.0 if home < away else -1.0
    return sign


def _team_mean(values, mask):
    """Nan-aware mean over slots selected by mask, per frame."""
    v = values[:, mask]
    n = np.isfinite(v).sum(axis=1)
    s = np.nansum(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def _speed(xy_m, fps, cap):
    """Per-frame speed (m/s) from a centred two-frame difference; NaN where not computable."""
    v = np.full(xy_m.shape[:-1], np.nan, np.float32)
    d = np.linalg.norm(xy_m[2:] - xy_m[:-2], axis=-1) * fps / 2
    v[1:-1] = np.where(d <= cap, d, np.nan)
    return v


def _fill_index(valid, forward=True):
    """Index of the nearest valid frame at-or-before (forward) / at-or-after (backward), or -1."""
    n = len(valid)
    idx = np.where(valid, np.arange(n), -1 if forward else n)
    if forward:
        out = np.maximum.accumulate(idx)
    else:
        out = np.minimum.accumulate(idx[::-1])[::-1]
        out[out == n] = -1
    return out


def rolling_mean(x, w):
    """Centred NaN-aware mean over +/- w frames."""
    finite = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(finite)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan).astype(np.float32)


def raw_features(match, version=1, play=None):
    """play: optional (T,) bool of frames in play. From version 3, tracking outside it (pre-match,
    halftime, before a kick-off) is blanked before anything is computed, so no window or gap fill
    can reach into states the DFL training data never contains."""
    fps = match.fps
    length, width = match.pitch
    ball = match.ball_m()
    players = match.players_m()
    if version >= 3 and play is not None:
        ball = np.where(play[:, None], ball, np.nan)
        players = np.where(play[:, None, None], players, np.nan)
    sign = _attack_sign(match)
    home, away = match.team == 0, match.team == 1
    T = match.n_frames
    f = {}

    dx = np.maximum(0, np.maximum(-ball[:, 0], ball[:, 0] - length))
    dy = np.maximum(0, np.maximum(-ball[:, 1], ball[:, 1] - width))
    off = np.hypot(dx, dy)
    if version >= 2:
        # Harmonise sources: the ATD almost never reports a ball off the pitch while clean DFL
        # often does, so a ball further out than OFFPITCH_NAN_M counts as not seen, everywhere.
        ball = np.where((off > OFFPITCH_NAN_M)[:, None], np.nan, ball)
    vis = np.isfinite(ball[:, 0])
    f["ball_vis"] = vis.astype(np.float32)
    f["ball_speed"] = _speed(ball, fps, MAX_BALL_SPEED)
    f["ball_off"] = np.where(vis, off, np.nan)
    # Ball x in the home side's attacking frame: 1 = at the goal home attacks.
    bx = ball[:, 0] / length
    f["ball_x_home"] = np.where(sign > 0, bx, 1 - bx)
    f["ball_edge"] = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])

    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d_home = dist[:, home].min(axis=1)
    d_away = dist[:, away].min(axis=1)
    d_home[~np.isfinite(d_home)] = np.nan
    d_away[~np.isfinite(d_away)] = np.nan
    f["d_home"] = np.minimum(d_home, 40)
    f["d_away"] = np.minimum(d_away, 40)
    f["d_diff"] = np.clip(d_away - d_home, -25, 25)  # > 0: a home player is closer
    f["near_home"] = np.where(np.isfinite(f["d_diff"]), (f["d_diff"] > 0).astype(np.float32), np.nan)
    f["n_within_3m"] = np.where(vis, (dist < 3).sum(axis=1), np.nan).astype(np.float32)

    speed = _speed(players, fps, MAX_PLAYER_SPEED)
    f["speed_all"], n_all = _team_mean(speed, np.ones(len(match.team), bool))
    f["speed_home"], _ = _team_mean(speed, home)
    f["speed_away"], _ = _team_mean(speed, away)
    f["speed_max"] = np.where(np.isfinite(speed).any(axis=1), np.nanmax(np.where(np.isfinite(speed), speed, -1), axis=1), np.nan)
    f["n_players"] = np.isfinite(players[:, :, 0]).sum(axis=1).astype(np.float32)

    # Team shape in each team's own attacking frame (1 = the goal it attacks).
    px = players[:, :, 0] / length
    for name, mask, s in (("home", home, sign), ("away", away, -sign)):
        own = np.where(s[:, None] > 0, px, 1 - px)
        cx, _ = _team_mean(own, mask)
        f["cx_" + name] = cx
        f["spread_" + name] = np.nanstd(np.where(mask[None, :], own, np.nan), axis=1)
        vel = np.full(T, np.nan, np.float32)
        vel[12:-12] = (cx[24:] - cx[:-24]) * length * fps / 24
        f["advance_" + name] = vel
    f["cx_diff"] = f["cx_home"] - f["cx_away"]
    f["advance_diff"] = f["advance_home"] - f["advance_away"]

    if version >= 2:
        # Dead-ball signals. Stoppages are collective: most players slow down at once.
        fin = np.isfinite(speed)
        n = np.maximum(fin.sum(axis=1), 1)
        f["frac_slow"] = np.where(fin.any(axis=1), (fin & (speed < 1.0)).sum(axis=1) / n, np.nan)
        f["frac_fast"] = np.where(fin.any(axis=1), (fin & (speed > 4.0)).sum(axis=1) / n, np.nan)
        # Kick-off shape: every player in his own half.
        for name, mask, s in (("home", home, sign), ("away", away, -sign)):
            own = np.where(s[:, None] > 0, px, 1 - px)[:, mask]
            seen = np.isfinite(own)
            f["own_half_" + name] = np.where(seen.any(axis=1),
                                             (seen & (own < 0.5)).sum(axis=1) / np.maximum(seen.sum(axis=1), 1),
                                             np.nan)
        f["own_half_min"] = np.fmin(f["own_half_home"], f["own_half_away"])
        # Did the ball travel? Displacement over 2 s, as an indicator robust to slow drift.
        disp = np.full(T, np.nan, np.float32)
        disp[25:-25] = np.linalg.norm(ball[50:] - ball[:-50], axis=1)
        f["ball_travelled"] = np.where(np.isfinite(disp), (disp > 5.0).astype(np.float32), np.nan)
        f["ball_edge_clip"] = np.where(vis, np.clip(f["ball_edge"], 0, 40), np.nan)

    # Ball gaps: how long since / until the ball was seen, and who was nearest then.
    last = _fill_index(vis, forward=True)
    nxt = _fill_index(vis, forward=False)
    i = np.arange(T)
    if version >= 3:
        # A gap fill never crosses into another half (or the X gap between DFL halves).
        last = np.where((last >= 0) & (match.half[np.maximum(last, 0)] == match.half), last, -1)
        nxt = np.where((nxt >= 0) & (match.half[np.maximum(nxt, 0)] == match.half), nxt, -1)
    f["t_since_ball"] = np.where(last >= 0, np.minimum((i - last) / fps, GAP_CAP_S), GAP_CAP_S).astype(np.float32)
    f["t_until_ball"] = np.where(nxt >= 0, np.minimum((nxt - i) / fps, GAP_CAP_S), GAP_CAP_S).astype(np.float32)
    dd = np.nan_to_num(f["d_diff"], nan=0.0)
    f["last_d_diff"] = np.where(last >= 0, dd[np.maximum(last, 0)], np.nan)
    f["next_d_diff"] = np.where(nxt >= 0, dd[np.maximum(nxt, 0)], np.nan)
    bxh = np.nan_to_num(f["ball_x_home"], nan=0.5)
    f["last_ball_x_home"] = np.where(last >= 0, bxh[np.maximum(last, 0)], np.nan)
    f["next_ball_x_home"] = np.where(nxt >= 0, bxh[np.maximum(nxt, 0)], np.nan)
    return {k: np.asarray(v, np.float32) for k, v in f.items()}


ONE_SIDED = ("d_diff", "near_home", "n_within_3m", "advance_diff", "cx_diff", "frac_slow", "ball_vis")
ONE_SIDED_WINDOWS = (12, 50)
ONE_SIDED_WINDOWS_V5 = (3, 6, 12, 50)  # v5 adds 0.12 s and 0.25 s, for short possessions


def side_mean(x, w, future):
    """NaN-aware mean over the w frames strictly before (future=False) or after t."""
    finite = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(finite)))
    i = np.arange(len(x))
    if future:
        lo, hi = np.minimum(len(x), i + 1), np.minimum(len(x), i + 1 + w)
    else:
        lo, hi = np.maximum(0, i - w), i
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan).astype(np.float32)


def feature_matrix(match, version=1, play=None):
    """(T, F) float32 and the feature names."""
    raw = raw_features(match, version, play)
    cols, names = [], []
    for k, v in raw.items():
        cols.append(v)
        names.append(k)
        for w in (WINDOWS if version < 2 else WINDOWS_V2):
            cols.append(rolling_mean(v, w))
            names.append("{}_w{}".format(k, w))
        if version >= 4 and k in ONE_SIDED:
            # Before / after / change: centred windows blur a hand-over, one-sided ones locate it.
            for w in (ONE_SIDED_WINDOWS if version < 5 else ONE_SIDED_WINDOWS_V5):
                past, fut = side_mean(v, w, False), side_mean(v, w, True)
                cols += [past, fut, fut - past]
                names += ["{}_past{}".format(k, w), "{}_fut{}".format(k, w), "{}_chg{}".format(k, w)]
    return np.column_stack(cols), names


def swap_teams(match):
    """The same match with home and away exchanged, labels included. For symmetric training."""
    from .match import A, H, Match
    labels = None
    if match.labels is not None:
        labels = match.labels.copy()
        labels[match.labels == H] = A
        labels[match.labels == A] = H
    return Match(name=match.name + "_swap", ball=match.ball, players=match.players,
                 team=(1 - match.team).astype(np.int8), half=match.half, pitch=match.pitch,
                 fps=match.fps, labels=labels)
