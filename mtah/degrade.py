"""Make a clean, labelled Match (DFL) look like the BAR-ATH automatic tracking.

What is reproduced, each from the aggregate distributions in calibration/atd_noise.json:

  * ball gaps: the ball alternates visible / missing with run lengths drawn from the ATD's own
    visible-run and missing-run distributions (long stretches, not flicker);
  * no identities: every person's track is cut into tracklets with ATD-like lengths, separated by
    short gaps, and each tracklet lands in a random free slot of its team -- so a slot is a
    tracklet, not a player, exactly as in the ATD;
  * visibility: gaps are scaled so the mean number of visible players matches the ATD.

What is not reproduced (yet): duplicated tracks, wrong-team slots, and any correlation between
ball loss and game state. Deterministic: the RNG is seeded from the match name and `seed`.
"""
import json
import zlib

import numpy as np

from . import config
from .match import Match
from .stats import QUANTILES, graded_blocks

SLOTS_PER_TEAM = 23


def load_calibration(path=config.CALIBRATION_PATH):
    with open(path) as handle:
        return json.load(handle)


def _sampler(quantiles_s, fps, rng):
    """Inverse-CDF sampler (in frames, >= 1) from a 101-point quantile table in seconds."""
    table = np.asarray(quantiles_s, float) * fps

    def draw():
        return max(1, int(round(np.interp(rng.random(), QUANTILES, table))))
    return draw


def _alternating_mask(n, draw_on, draw_off, p_on, rng):
    """Boolean mask of length n alternating runs of True/False with the given run samplers."""
    mask = np.zeros(n, bool)
    state = rng.random() < p_on
    pos = 0
    while pos < n:
        length = draw_on() if state else draw_off()
        if state:
            mask[pos:pos + length] = True
        pos += length
        state = not state
    return mask


DEGRADE_VERSION = 3  # bump to invalidate cached degraded matches


def degrade(match, calibration=None, seed=0, ball=True, players=True, off_pitch=True,
            animate_dead=False):
    cal = calibration or load_calibration()
    key = [zlib.crc32(match.name.encode()), seed]
    rng = np.random.default_rng(key)
    fps = match.fps
    if animate_dead and match.labels is not None:
        match = Match(name=match.name, ball=_animate_dead_ball(match, key), players=match.players,
                      team=match.team, half=match.half, pitch=match.pitch, fps=fps,
                      labels=match.labels)
    out_ball = match.ball.copy()
    if ball:
        out_ball = _degrade_ball(match, cal, key, off_pitch)

    out_players, out_team = match.players, match.team
    if players:
        out_players, out_team = _fragment_players(match, cal, rng)
    return Match(name=match.name, ball=out_ball, players=out_players, team=out_team,
                 half=match.half, pitch=match.pitch, fps=fps, labels=match.labels)


CARRY_HOLD_S = (3.0, 8.0)   # a player keeps the dead ball this long before handing it on
CARRY_PASS_S = 1.0          # a hand-on travels for this long
CARRY_REACH_M = 25.0        # ... to a player within this distance
CARRY_OFFSET_M = 0.4
RESTART_BLEND_S = 1.0       # the ball reaches the real restart spot over the last second


def _animate_dead_ball(match, key):
    """Replace the DFL's frozen dead ball with a ball that players carry and hand on.

    The DFL positional data holds the ball perfectly still during every stoppage (100% of dead
    frames have zero displacement, against ~1% of alive ones), which would make 'the ball is still'
    a perfect dead-ball detector on the benchmark and on nothing else: the ATD's ball keeps moving
    through stoppages (it is visible on ~80% of the published all-dead stretch, and travels).

    Model: the player nearest the frozen spot picks the ball up and carries it; every 3-8 s it is
    handed on over 1 s to a random player within 25 m; over the last second it moves to where play
    actually restarts. A heuristic, documented as such; its only job is to remove the leak.
    """
    rng = np.random.default_rng(key + [3])
    fps = match.fps
    ball = match.ball_m().astype(np.float64)
    players = match.players_m().astype(np.float64)
    pitch = np.asarray(match.pitch, np.float64)
    dead = (match.labels == 2) & (match.half > 0)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], dead.astype(np.int8), [0]))))
    out = ball.copy()
    for a, b in zip(edges[::2], edges[1::2]):
        start = ball[a]
        if not np.isfinite(start).all():
            continue
        pos = start.copy()
        carrier, t = None, a
        while t < b:
            seen = np.flatnonzero(np.isfinite(players[t, :, 0]))
            if not len(seen):
                out[t] = pos
                t += 1
                continue
            if carrier is None or not np.isfinite(players[t, carrier, 0]):
                carrier = seen[np.argmin(np.linalg.norm(players[t, seen] - pos, axis=1))]
            hold = int(rng.uniform(*CARRY_HOLD_S) * fps)
            offset = rng.normal(0, CARRY_OFFSET_M, 2)
            end = min(b, t + hold)
            for u in range(t, end):
                if np.isfinite(players[u, carrier, 0]):
                    pos = players[u, carrier] + offset
                out[u] = pos
            t = end
            if t >= b:
                break
            near = [s for s in np.flatnonzero(np.isfinite(players[t, :, 0]))
                    if s != carrier and np.linalg.norm(players[t, s] - pos) < CARRY_REACH_M]
            if not near:
                continue
            nxt = int(rng.choice(near))
            n = min(b - t, int(CARRY_PASS_S * fps))
            target = players[min(t + n, b - 1), nxt]
            if np.isfinite(target).all():
                w = (np.arange(1, n + 1) / n)[:, None]
                out[t:t + n] = pos + w * (target - pos)
                pos = out[t + n - 1].copy()
                carrier = nxt
            t += n
        # Blend into the real restart position over the last second.
        if b < len(ball) and np.isfinite(ball[b]).all():
            n = min(b - a, int(RESTART_BLEND_S * fps))
            w = (np.arange(1, n + 1) / n)[:, None]
            out[b - n:b] = out[b - n:b] * (1 - w) + ball[b] * w
    return (out / pitch).astype(np.float32)


def _degrade_ball(match, cal, key, drop_off_pitch=True):
    """Drop the ball where the ATD would not have it.

    Two mechanisms. (1) The ATD barely tracks the ball off the pitch, so positions further outside
    than the ATD's own p95 out-of-pitch distance are removed. (2) On top of that, random missing
    stretches with ATD run lengths. The missing-run lengths are scaled by one factor, found by
    bisection, so the overall visible fraction in graded frames matches the ATD's.
    """
    ball_m = match.ball_m()
    length, width = match.pitch
    dx = np.maximum(0, np.maximum(-ball_m[:, 0], ball_m[:, 0] - length))
    dy = np.maximum(0, np.maximum(-ball_m[:, 1], ball_m[:, 1] - width))
    outside = np.hypot(dx, dy) > 0
    graded = match.graded
    target = cal["ball_visible_frac"]
    # Off-pitch stretches survive with probability p, chosen so the share of visible ball
    # positions that are off the pitch matches the ATD's (~4%, against ~30% in clean DFL).
    off_share = outside[graded].mean()
    p_keep = min(1.0, cal.get("ball_out_frac", 0.0) * target / max(off_share, 1e-9))
    rng_off = np.random.default_rng(key + [2])
    off_pitch = np.zeros(match.n_frames, bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], outside.astype(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        if drop_off_pitch and rng_off.random() >= p_keep:
            off_pitch[start:stop] = True

    def masked(scale):
        rng = np.random.default_rng(key + [1])
        draw_vis = _sampler(cal["ball_visible_run_s"], match.fps, rng)
        miss_table = np.asarray(cal["ball_missing_run_s"]) * scale
        draw_miss = _sampler(miss_table, match.fps, rng)
        keep = np.ones(match.n_frames, bool)
        for start, stop in graded_blocks(match):
            keep[start:stop] = _alternating_mask(stop - start, draw_vis, draw_miss, target, rng)
        return keep & ~off_pitch & np.isfinite(ball_m[:, 0])

    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        if masked(mid)[graded].mean() > target:
            lo = mid
        else:
            hi = mid
    keep = masked((lo + hi) / 2)
    out = match.ball.copy()
    out[~keep] = np.nan
    return out


def _fragment_players(match, cal, rng):
    fps = match.fps
    T = match.n_frames
    draw_run = _sampler(cal["slot_visible_run_s"], fps, rng)
    # Gap length: exponential, mean chosen so the visible fraction of on-pitch people matches.
    on_pitch_now = np.isfinite(match.players[match.graded, :, 0]).sum(axis=1).mean()
    target = min(1.0, cal["players_visible_mean"] / max(on_pitch_now, 1e-9))
    mean_run = float(np.mean(np.asarray(cal["slot_visible_run_s"]) * fps))
    mean_gap = mean_run * (1 - target) / max(target, 1e-3)

    players = np.full((T, 2 * SLOTS_PER_TEAM, 2), np.nan, np.float32)
    team = np.repeat(np.array([0, 1], np.int8), SLOTS_PER_TEAM)
    busy_until = np.zeros(2 * SLOTS_PER_TEAM, np.int64)  # first frame a slot is free again

    tracklets = []
    for person in range(match.players.shape[1]):
        seen = np.isfinite(match.players[:, person, 0])
        idx = np.flatnonzero(seen)
        if not len(idx):
            continue
        # Contiguous on-pitch spans of this person (halves, substitutions).
        breaks = np.flatnonzero(np.diff(idx) > 1)
        spans = zip(np.concatenate(([idx[0]], idx[breaks + 1])),
                    np.concatenate((idx[breaks], [idx[-1]])) + 1)
        for start, stop in spans:
            pos = start
            while pos < stop:
                length = draw_run()
                end = min(stop, pos + length)
                tracklets.append((pos, end, person, match.team[person]))
                pos = end + max(1, int(round(rng.exponential(mean_gap)))) if mean_gap > 0 else end

    for start, stop, person, side in sorted(tracklets):
        pool = np.flatnonzero((team == side) & (busy_until <= start))
        if not len(pool):  # every slot of that side is busy: drop this tracklet
            continue
        slot = int(rng.choice(pool))
        players[start:stop, slot] = match.players[start:stop, person]
        busy_until[slot] = stop
    return players, team
