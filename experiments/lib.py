"""
lib.py - A Unfooling-LIME-SHAP file
Copyright (C) 2022  Zach Carmichael
"""
from typing import List
from typing import Tuple
from typing import Sequence

from abc import ABC
from abc import abstractmethod

import numpy as np

from fooling.models import SimpleTabularModel


class Problem(ABC):
    """Base Problem class for experiment definitions"""

    def __init__(self, params):
        self._params = params
        (self._X, self._y, self._features,
         self._categorical_features) = self.load_data()

    @property
    @abstractmethod
    def biased_features(self):
        raise NotImplementedError

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def features(self):
        return self._features

    @property
    def categorical_features(self):
        return self._categorical_features

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray,
                                 Sequence[str], Sequence[str]]:
        pass

    @property
    @abstractmethod
    def prejudiced_model(self) -> SimpleTabularModel:
        pass

    @property
    @abstractmethod
    def innocuous_models(self) -> List[SimpleTabularModel]:
        pass

    @property
    @abstractmethod
    def sensitive_features(self) -> List[str]:
        pass

    @property
    def sensitive_feature_idxs(self) -> List[int]:
        return [self._features.index(name) for name in self.sensitive_features]
