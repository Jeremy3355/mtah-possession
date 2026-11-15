"""Aggregate noise statistics of a Match: what the degradation has to reproduce.

Everything returned is a distribution summary (quantiles, means), never a frame or a position, so
the output is safe to print and to commit.
"""
import numpy as np

QUANTILES = np.linspace(0, 1, 101)


def runs(mask):
    """(lengths of True runs, lengths of False runs) of a boolean array."""
    if not len(mask):
        return np.array([], int), np.array([], int)
    edges = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    bounds = np.concatenate(([0], edges, [len(mask)]))
    lengths = np.diff(bounds)
    values = mask[bounds[:-1]]
    return lengths[values], lengths[~values]


def graded_blocks(match):
    """[(start, stop)] of each half, half-open."""
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(match.half == h)
        if len(idx):
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def robust_jitter(xy_m):
    """Per-axis robust sigma of the second difference of a (T, 2) metre track, NaN-aware."""
    d2 = xy_m[2:] - 2 * xy_m[1:-1] + xy_m[:-2]
    d2 = d2[np.isfinite(d2).all(axis=1)]
    if not len(d2):
        return float("nan")
    mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0)
    return float(np.mean(1.4826 * mad))


def noise_stats(match):
    fps = match.fps
    ball_vis, ball_miss, slot_vis, slot_gap = [], [], [], []
    for start, stop in graded_blocks(match):
        seen = np.isfinite(match.ball[start:stop, 0])
        v, m = runs(seen)
        ball_vis.extend(v)
        ball_miss.extend(m)
        for s in range(match.players.shape[1]):
            seen_s = np.isfinite(match.players[start:stop, s, 0])
            if not seen_s.any():
                continue
            v, m = runs(seen_s)
            slot_vis.extend(v)
            # Only interior gaps: leading/trailing absence is a slot not yet / no longer used.
            first, last = np.flatnonzero(seen_s)[[0, -1]]
            _, interior = runs(seen_s[first:last + 1])
            slot_gap.extend(interior)

    graded = match.graded
    visible = np.isfinite(match.players[graded, :, 0])
    per_team = [visible[:, match.team == t].sum(axis=1) for t in (0, 1)]
    ball_m = match.ball_m()
    players_m = match.players_m()
    outside = outside_distance(ball_m[graded], match.pitch)
    q = lambda a: (np.quantile(np.asarray(a, float) / fps, QUANTILES).round(3).tolist()
                   if len(a) else None)
    return {
        "ball_visible_frac": float(np.isfinite(match.ball[graded, 0]).mean()),
        "ball_visible_run_s": q(ball_vis),
        "ball_missing_run_s": q(ball_miss),
        "n_ball_missing_runs": len(ball_miss),
        "slot_visible_run_s": q(slot_vis),
        "slot_gap_s": q(slot_gap) if slot_gap else None,
        "players_visible_mean": float((per_team[0] + per_team[1]).mean()),
        "players_visible_per_team_mean": [float(p.mean()) for p in per_team],
        "ball_jitter_m": robust_jitter(ball_m),
        "player_jitter_m": float(np.nanmedian([robust_jitter(players_m[:, s])
                                               for s in range(players_m.shape[1])])),
        "ball_out_frac": float(np.mean(outside > 0)) if len(outside) else None,
        "ball_out_distance_m": q(outside[outside > 0] * fps) if (outside > 0).any() else None,
    }


def outside_distance(ball_m, pitch):
    """Metres outside the pitch rectangle (0 inside) for each finite ball position."""
    b = ball_m[np.isfinite(ball_m).all(axis=1)]
    dx = np.maximum(0, np.maximum(-b[:, 0], b[:, 0] - pitch[0]))
    dy = np.maximum(0, np.maximum(-b[:, 1], b[:, 1] - pitch[1]))
    return np.hypot(dx, dy)
