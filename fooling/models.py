import functools
from abc import ABC
from abc import abstractmethod

import numpy as np

from fooling.utils import one_hot_encode


class SimpleTabularModel(ABC):
    def __init__(self, negative_outcome, positive_outcome, idxs, reduce,
                 name=None):
        self.negative_outcome = negative_outcome
        self.positive_outcome = positive_outcome
        if isinstance(idxs, int):
            idxs = [idxs]
        self.idxs = idxs
        self.reduce = reduce
        if name is None:
            name = str(self)
        self._name = name

    @property
    def name(self):
        if self._name is None:
            return str(self)
        return self._name

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        raise NotImplementedError

    @abstractmethod
    def score(self, X, y):
        raise NotImplementedError

    def __str__(self):
        return (f'{self.__class__.__name__} idxs={self.idxs} '
                f'reduce={self.reduce}')

    def __repr__(self):
        return str(self)


class SimpleClassificationModel(SimpleTabularModel):
    def __init__(self, negative_outcome, positive_outcome, idxs, thresholds=0,
                 reduce=np.logical_or, name=None):
        super().__init__(negative_outcome, positive_outcome, idxs, reduce, name)
        thresholds = np.asarray(thresholds)
        if thresholds.ndim == 0:
            thresholds = np.repeat(thresholds, len(self.idxs))
        elif thresholds.ndim != 1:
            raise ValueError(f'Thresholds must be scalar or have 1 dim, '
                             f'instead data has {thresholds.ndim} dims.')
        elif len(thresholds) != len(self.idxs):
            raise ValueError(f'Thresholds ({len(thresholds)}) and '
                             f'unrelated_idxs ({len(self.idxs)}) must '
                             f'have the same number of elements.')
        self.thresholds = thresholds

    def predict(self, X):
        return np.asarray([
            self.negative_outcome
            if functools.reduce(self.reduce, (
                x[idx] > 0 for idx, thresh in zip(self.idxs,
                                                  self.thresholds)))
            else self.positive_outcome
            for x in X
        ])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(X)


class PrejudicedClassificationModel(SimpleClassificationModel):
    """the biased model"""

    def __init__(self, negative_outcome, positive_outcome, racist_idxs,
                 thresholds=0, reduce=np.logical_or, name=None):
        super().__init__(
            negative_outcome, positive_outcome,
            idxs=racist_idxs, thresholds=thresholds,
            reduce=reduce, name=name,
        )


class InnocuousClassificationModel(SimpleClassificationModel):
    """the display model with unrelated features"""

    def __init__(self, negative_outcome, positive_outcome, unrelated_idxs,
                 thresholds=0, reduce=np.logical_xor, name=None):
        super().__init__(
            negative_outcome, positive_outcome,
            idxs=unrelated_idxs, thresholds=thresholds,
            reduce=reduce, name=name,
        )


class SimpleRegressionModel(SimpleTabularModel):
    def __init__(self, negative_outcome, positive_outcome, idxs,
                 reduce=np.multiply, name=None):
        super().__init__(negative_outcome, positive_outcome, idxs, reduce, name)

    def predict(self, X):
        return np.asarray([
            functools.reduce(self.reduce, (x[idx] for idx in self.idxs))
            for x in X
        ])

    def predict_proba(self, X):
        raise TypeError('predict_proba invalid for regression models')

    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)


class PrejudicedRegressionModel(SimpleRegressionModel):
    """the biased model"""

    def __init__(self, negative_outcome, positive_outcome, racist_idxs,
                 reduce=np.multiply, name=None):
        super().__init__(
            negative_outcome, positive_outcome,
            idxs=racist_idxs, reduce=reduce, name=name,
        )


class InnocuousRegressionModel(SimpleRegressionModel):
    """the display model with unrelated features"""

    def __init__(self, negative_outcome, positive_outcome, unrelated_idxs,
                 reduce=np.multiply, name=None):
        super().__init__(
            negative_outcome, positive_outcome,
            idxs=unrelated_idxs, reduce=reduce, name=name,
        )
