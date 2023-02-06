from functools import wraps

import numpy as np
from sklearn.base import OutlierMixin, BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class NoveltyDetector(OutlierMixin, BaseEstimator):
    def __init__(
            self,
            name='IF',
            strategy=None,
            reduce='min',
            selected_idxs=None,
            **kwargs,
    ):
        kwargs.setdefault('n_jobs', -1)
        if name == 'LOF':
            kwargs.setdefault('novelty', True)
            kwargs.setdefault('n_neighbors', 20)
            estimator_cls = LocalOutlierFactor
        elif name == 'IF':
            kwargs.setdefault('n_estimators', 100)
            kwargs.setdefault('max_samples', 256)
            estimator_cls = IsolationForest
        else:
            raise ValueError(name)
        self._estimator = None
        self._estimators = None
        self._estimator_cls = estimator_cls
        self._estimator_kwargs = kwargs

        if strategy is None:
            if selected_idxs is not None:
                strategy = 'selected_idxs'
            else:
                strategy = 'combined'
        if strategy.startswith('selected_idxs'):
            assert selected_idxs is not None, (
                f'{strategy} requires that selected_idxs be specified')

        self._selected_idxs = selected_idxs

        if isinstance(strategy, str):
            assert strategy in {'combined', 'independent', 'y',
                                'selected_idxs_independent', 'selected_idxs'}
        else:
            raise ValueError(f'Unknown strategy "{strategy}"')
        if reduce is not None and not callable(reduce):
            if reduce == 'mean':
                reduce = np.mean
            elif reduce == 'max':
                reduce = np.max
            elif reduce == 'min':
                reduce = np.min
            elif reduce == 'median':
                reduce = np.median
            else:
                raise ValueError(f'Unknown reduction "{reduce}"')
        self.strategy = strategy
        self.reduce = reduce

    def _init_estimator(self, X):
        kwargs = self._estimator_kwargs.copy()
        if ('max_samples' in kwargs and
                not isinstance(kwargs['max_samples'], str) and
                len(X) < kwargs['max_samples']):
            kwargs['max_samples'] = len(X)
        return self._estimator_cls(**kwargs)

    def _apply_strategy(func):
        @wraps(func)
        def wrapper(self: 'NoveltyDetector', X, y, *args, **kwargs):
            if y.ndim == 1:
                y = y[:, np.newaxis]

            if self.strategy == 'combined':
                Xy = np.concatenate([X, y], axis=1)
                if self._estimator is None:
                    self._estimator = self._init_estimator(X)
                return func(self, Xy, *args, **kwargs)
            elif self.strategy == 'y':
                if self._estimator is None:
                    self._estimator = self._init_estimator(X)
                return func(self, y, *args, **kwargs)
            elif self.strategy == 'independent':
                if self._estimators is None:
                    self._estimators = [self._init_estimator(X)
                                        for _ in range(X.shape[1])]
                results = []
                for i in range(X.shape[1]):
                    Xy = np.concatenate([X[:, i:i + 1], y], axis=1)
                    self._estimator = self._estimators[i]
                    results.append(func(self, Xy, *args, **kwargs))
                self._estimator = None
                return results
            elif self.strategy == 'selected_idxs':
                Xy = np.concatenate([X[:, self._selected_idxs], y], axis=1)
                if self._estimator is None:
                    self._estimator = self._init_estimator(X)
                return func(self, Xy, *args, **kwargs)
            elif self.strategy == 'selected_idxs_independent':
                if self._estimators is None:
                    self._estimators = [
                        self._init_estimator(X)
                        for _ in range(len(self._selected_idxs))
                    ]
                results = []
                for est, i in enumerate(self._selected_idxs):
                    Xy = np.concatenate([X[:, i:i + 1], y], axis=1)
                    self._estimator = self._estimators[est]
                    results.append(func(self, Xy, *args, **kwargs))
                self._estimator = None
                return results
            else:
                raise RuntimeError(f'invalid strategy {self.strategy}')

        return wrapper

    def _apply_reduce(func):
        @wraps(func)
        def wrapper(self: 'NoveltyDetector', *args, **kwargs):
            results = func(self, *args, **kwargs)
            if not isinstance(results, list) or self.reduce is None:
                return results
            # otherwise
            results = np.stack(results, axis=-1)
            return self.reduce(results, axis=-1)

        return wrapper

    @_apply_strategy
    def fit(self, X, y=None, **kwargs) -> 'NoveltyDetector':
        assert y is None
        self._estimator.fit(X, y, **kwargs)
        return self

    @_apply_reduce
    @_apply_strategy
    def fit_predict(self, X, y=None, **kwargs):
        assert y is None
        return self._estimator.fit_predict(X, y, **kwargs)

    @_apply_reduce
    @_apply_strategy
    def predict(self, X=None, **kwargs):
        return self._estimator.predict(X, **kwargs)

    @_apply_reduce
    @_apply_strategy
    def score_samples(self, X, **kwargs):
        return self._estimator.score_samples(X, **kwargs)

    @_apply_reduce
    @_apply_strategy
    def decision_function(self, X, **kwargs):
        return self._estimator.decision_function(X, **kwargs)

    @property
    def threshold_(self):
        if self.strategy in {'combined', 'selected_idxs', 'y'}:
            assert self._estimator is not None, 'must call fit first'
            return self._estimator.offset_
        elif self.strategy in {'independent', 'selected_idxs_independent'}:
            assert self._estimators is not None, 'must call fit first'
            return [estimator.offset_ for estimator in self._estimators]
        else:
            raise RuntimeError(f'invalid strategy {self.strategy}')
