"""Binomial tree model parameters."""

from dataclasses import dataclass

from hesperides.validation import BinomialParameterValidator


@dataclass(frozen=True)
class BinomialModel:

    up: float
    down: float
    rate: float

    def validate(self) -> None:
        BinomialParameterValidator().validate(
            up=self.up,
            down=self.down,
            rate=self.rate,
        )

    # ------------------------------------------------------------------
    # Risk-neutral measure
    # ------------------------------------------------------------------

    def risk_neutral_prob(self) -> float:
        """Risk-neutral probability of an up move.

        q = (1 + R - d) / (u - d)
        """
        return (1.0 + self.rate - self.down) / (self.up - self.down)
