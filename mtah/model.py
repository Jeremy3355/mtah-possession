"""Per-frame H/A/D probabilities from identity-free features, decoded into segments by Viterbi.

Classifiers are gradient boosting (scikit-learn, fixed random_state, so fully deterministic),
trained only on degraded DFL matches -- never on BAR-ATH. Training is team-symmetric: every match
is also used with home and away swapped, so nothing is learned about "the home side".

Two architectures:

  joint   one 3-class H/A/D classifier (v1-v3).
  factor  two heads. An alive/dead head that sees player-only features -- the ones whose
          distributions match between the ATD and the DFL -- so it never depends on how the DFL
          handles a dead ball (it freezes it). And a team head, H vs A, trained on alive frames
          only, where the DFL ball is real. P(D) = p_dead, P(H) = (1 - p_dead) * p_home.

The decoder is a three-state Viterbi with a cost for every change of state, which stops flicker
and, unlike a centred majority vote, adds no lag at boundaries.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import feature_matrix, swap_teams
from .match import A, D, H, X

# Features whose distribution differs clearly between the ATD and degraded DFL (DFL tracks a still
# or off-pitch ball that the ATD does not), so a model trained on DFL would misread them on BAR-ATH.
EXCLUDED_PREFIXES = ("ball_speed", "ball_off", "ball_edge_w", "ball_edge", "ball_travelled")
INCLUDED_PREFIXES = ("ball_edge_clip",)
# Alive/dead head: player-only features.
DEAD_PREFIXES = ("speed_", "frac_", "own_half", "cx_", "spread_", "advance_", "n_players")
SUBSAMPLE = 3
PARAMS = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, early_stopping=False, random_state=0)

FEATURE_VERSION = 4
ARCH = "joint"


def play_mask(match):
    """Frames in play: inside a half and not before its kick-off (published for BAR-ATH)."""
    from .kickoff import BAR_ATH_KICKOFFS
    play = match.half > 0
    if match.name == "BAR-ATH":
        frames = np.arange(match.n_frames)
        for half, frame in BAR_ATH_KICKOFFS.items():
            play &= ~((match.half == half) & (frames < frame))
    return play


def features(match, version=None):
    """(x_all, x_dead) feature matrices for the joint/team head and the alive/dead head."""
    x, names = feature_matrix(match, FEATURE_VERSION if version is None else version,
                              play_mask(match))
    keep = [i for i, n in enumerate(names)
            if n.startswith(INCLUDED_PREFIXES) or not n.startswith(EXCLUDED_PREFIXES)]
    dead = [i for i, n in enumerate(names) if n.startswith(DEAD_PREFIXES)]
    return x[:, keep], x[:, dead]


def _rows(m, subsample):
    return np.flatnonzero(play_mask(m) & (m.labels != X))[::subsample]


def train(matches, arch=None):
    arch = arch or ARCH
    xs, xd, ys = [], [], []
    for match in matches:
        for m in (match, swap_teams(match)):
            x_all, x_dead = features(m)
            rows = _rows(m, SUBSAMPLE)
            xs.append(x_all[rows])
            xd.append(x_dead[rows])
            ys.append(m.labels[rows])
    x_all, x_dead, y = np.vstack(xs), np.vstack(xd), np.concatenate(ys)
    if arch == "joint":
        return ("joint", HistGradientBoostingClassifier(**PARAMS).fit(x_all, y))
    dead_head = HistGradientBoostingClassifier(**PARAMS).fit(x_dead, (y == D).astype(np.int8))
    alive = y != D
    team_head = HistGradientBoostingClassifier(**PARAMS).fit(x_all[alive], (y[alive] == H).astype(np.int8))
    return ("factor", dead_head, team_head)


def _proba_one(model, match):
    x_all, x_dead = features(match)
    if model[0] == "joint":
        clf = model[1]
        p = clf.predict_proba(x_all)
        order = list(clf.classes_)
        return p[:, [order.index(H), order.index(A), order.index(D)]]
    p_dead = model[1].predict_proba(x_dead)[:, 1]
    p_home = model[2].predict_proba(x_all)[:, 1]
    return np.column_stack(((1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead))


def predict_proba(model, match):
    """(T, 3) probabilities for H, A, D, averaged with the team-swapped view for symmetry."""
    p = _proba_one(model, match)
    ps = _proba_one(model, swap_teams(match))[:, [1, 0, 2]]  # swapped view: its H is our A
    return (p + ps) / 2


def viterbi(logp, switch_cost, dead_cost, switch_t=None):
    """Most likely H/A/D path. Changing H<->A costs switch_cost (or switch_t[t] when given, a
    per-frame cost), entering/leaving D costs dead_cost."""
    T = len(logp)
    trans = np.array([[0, switch_cost, dead_cost],
                      [switch_cost, 0, dead_cost],
                      [dead_cost, dead_cost, 0]], np.float64)
    score = logp[0].astype(np.float64).copy()
    back = np.empty((T, 3), np.int8)
    for t in range(1, T):
        if switch_t is not None:
            trans[0, 1] = trans[1, 0] = switch_t[t]
        cand = score[:, None] - trans  # from i to j
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logp[t]
    path = np.empty(T, np.int8)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def visibility_switch_cost(ball, switch_visible, switch_hidden, window=12):
    """Per-frame team-change cost: cheaper where the ball was seen within +/- window frames.

    A hand-over the tracking actually saw is better evidenced than one inferred across a gap, so
    the decoder may follow short possessions there and stay conservative inside ball gaps.
    """
    seen = np.isfinite(ball[:, 0]).astype(np.int32)
    cs = np.concatenate(([0], np.cumsum(seen)))
    i = np.arange(len(seen))
    near = (cs[np.minimum(len(seen), i + window + 1)] - cs[np.maximum(0, i - window)]) > 0
    return np.where(near, switch_visible, switch_hidden).astype(np.float64)


def decode(proba, half, switch_cost=8.0, dead_cost=8.0, switch_t=None):
    """Viterbi inside each half separately; X outside play. switch_t: optional per-frame cost."""
    out = np.full(len(proba), X, np.int8)
    logp = np.log(np.clip(proba, 1e-6, 1.0))
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if len(rows):
            out[rows] = viterbi(logp[rows], switch_cost, dead_cost,
                                None if switch_t is None else switch_t[rows])
    return out
