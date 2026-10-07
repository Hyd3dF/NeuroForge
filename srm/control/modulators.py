"""Modulators (07 §3): DA, ACh, NE and 5-HT as scalars computed from measurable quantities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Modulators:
    da: float = 0.0  # TD error (Selector critic)
    ach: float = 0.0  # expected uncertainty: EMA of normalized prediction error
    ne: float = 0.0  # unexpected surprise: contradiction / evidence-against events
    ht5: float = 0.5  # patience: stakes and remaining budget
    ema: float = 0.2
    history: list[dict[str, float]] = field(default_factory=list)

    def update(self, td_error: float, prediction_error: float, surprise_events: int, stakes: float,
               budget_fraction_left: float) -> None:
        self.da = td_error
        self.ach = (1 - self.ema) * self.ach + self.ema * min(1.0, max(0.0, prediction_error))
        self.ne = min(3.0, 0.5 * self.ne + float(surprise_events))
        self.ht5 = min(1.0, max(0.0, 0.5 * stakes + 0.5 * budget_fraction_left))
        self.history.append({"da": self.da, "ach": self.ach, "ne": self.ne, "ht5": self.ht5})

    @property
    def gain_ach(self) -> float:
        """Gate gain: more bottom-up weight when the internal model is unreliable."""
        return 1.0 + self.ach

    @property
    def hebbian_modulator(self) -> float:
        return max(0.0, self.da) + 0.5 * self.ach
