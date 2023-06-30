import math

import numpy as np


class Scorer:
    base_odds = math.exp(3)

    def __init__(
        self,
        min_score=200,
        max_score=1_200,
        pdo=50,
        base_odds=base_odds,
        base=650,
    ):
        self.A = pdo / math.log(2)
        self.B = base - math.log(base_odds) * self.A
        self.min_score = min_score
        self.max_score = max_score
        self.lower_proba = 1 / (math.exp((max_score - self.B) / self.A) + 1)
        self.upper_proba = 1 / (math.exp((min_score - self.B) / self.A) + 1)

    def _to_score(self, proba):
        if proba < self.lower_proba:
            return self.max_score
        if proba > self.upper_proba:
            return self.min_score
        return round(self.A * math.log((1 - proba) / proba) + self.B)

    def to_score(self, proba):
        if np.isscalar(proba):
            return self._to_score(proba)
        return np.vectorize(self._to_score)(proba)

    def _to_proba(self, score):
        return 1 / (1 + math.exp((score - self.B) / self.A))

    def to_proba(self, score):
        if np.isscalar(score):
            return self._to_proba(score)
        return np.vectorize(self._to_proba)(score)
