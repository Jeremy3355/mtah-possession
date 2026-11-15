"""One common in-memory format for every match, whatever its source.

The model only ever sees a Match, so it runs unchanged on the BAR-ATH ATD and on the (degraded) DFL
open data. Coordinates are normalised 0-1 along (length, width), the ATD convention; multiply by
`pitch` for metres.

    ball     (T, 2) float32   NaN where the ball is not tracked
    players  (T, S, 2) float32 NaN where a slot is empty
    team     (S,) int8        0 = home, 1 = away
    half     (T,) int8        0 = outside play, 1, 2
    labels   (T,) int8 or None, 0 = H, 1 = A, 2 = D, 3 = X   (only for labelled sources)
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])


@dataclass
class Match:
    name: str
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    half: np.ndarray
    pitch: tuple
    fps: int = 25
    labels: Optional[np.ndarray] = None

    @property
    def n_frames(self):
        return len(self.ball)

    @property
    def graded(self):
        return self.half > 0

    def ball_m(self):
        return self.ball * np.asarray(self.pitch, np.float32)

    def players_m(self):
        return self.players * np.asarray(self.pitch, np.float32)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        extra = {} if self.labels is None else {"labels": self.labels}
        np.savez_compressed(path, name=self.name, ball=self.ball, players=self.players,
                            team=self.team, half=self.half, pitch=np.asarray(self.pitch),
                            fps=self.fps, **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(name=str(z["name"]), ball=z["ball"], players=z["players"], team=z["team"],
                   half=z["half"], pitch=tuple(float(v) for v in z["pitch"]),
                   fps=int(z["fps"]), labels=z["labels"] if "labels" in z.files else None)


def write_labels_csv(path, codes):
    """A frame,label CSV in the submission format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[codes]))
    return path
