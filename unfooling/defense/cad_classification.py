import warnings

import numpy as np

from sklearn.base import OutlierMixin
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


class CADClassificationDetector(OutlierMixin, BaseEstimator):
    def __init__(
            self,
            name='GMM',
            epsilon=.05,
            with_f_x=True,
            with_p_y=True,
            nan_percentile=None,
            **kwargs,
    ):
        if name == 'GMM':
            kwargs.setdefault('n_components', 10)
            kwargs.setdefault('covariance_type', 'full')
            kwargs.setdefault('max_iter', 100)
            estimator_cls = GaussianMixture
        elif name == 'VAE':
            from unfooling.defense import VAE

            kwargs.setdefault('conditional', False)
            kwargs.setdefault('condition_on_y', True)
            kwargs.setdefault('log_var', False)
            kwargs.setdefault('alt_loss', True)
            estimator_cls = VAE
        elif name == 'LOF':
            kwargs.setdefault('n_jobs', -1)
            kwargs.setdefault('novelty', True)
            kwargs.setdefault('n_neighbors', 20)
            estimator_cls = LocalOutlierFactor
        elif name == 'IF':
            kwargs.setdefault('n_jobs', -1)
            kwargs.setdefault('n_estimators', 100)
            kwargs.setdefault('max_samples', 256)
            estimator_cls = IsolationForest
        elif name == 'KDE':
            estimator_cls = KernelDensity
        else:
            raise ValueError(name)

        self.epsilon = epsilon

        self.name = name
        self._estimator_cls = estimator_cls
        self._estimator_kwargs = kwargs

        self._estimators = None
        self._p_y = None  # probability of y
        self._f_x = None  # likelihood of X

        self.with_f_x = with_f_x
        self.with_p_y = with_p_y
        self.nan_percentile = nan_percentile

        self.threshold_ = None

    def _init_estimator(self, X):
        kwargs = self._estimator_kwargs.copy()
        if ('max_samples' in kwargs and
                not isinstance(kwargs['max_samples'], str) and
                len(X) < kwargs['max_samples']):
            kwargs['max_samples'] = len(X)
        return self._estimator_cls(**kwargs)

    @staticmethod
    def _validate_input(X, y):
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            warnings.warn(f'Casting X from {X.dtype} to {np.float32}')
            X = X.astype(np.float32)
        y = np.asarray(y)
        assert len(X) == len(y), 'come on man'
        if not np.issubdtype(y.dtype, np.integer):
            warnings.warn(f'Casting y from {y.dtype} to {np.int32}')
            y_orig = y
            y = y.astype(np.int32)
            if not (y == y_orig).all():
                raise ValueError(f'In casting y from {y_orig.dtype} to '
                                 f'{y.dtype} there was a loss of precision. '
                                 f'This could happen if your labels are not '
                                 f'integer-coded classes.')
        if y.ndim != 1:
            if y.ndim == 2:
                y = y.squeeze(axis=1)
                if y.ndim != 1:
                    raise ValueError(f'Cannot squeeze out axis 1 from y! '
                                     f'Ensure your labels are integer-coded '
                                     f'classes.')
            else:
                raise ValueError(f'y.ndim = {y.ndim}, but must be 1 or 2!')
        return X, y

    def fit(self, X, y=None, **kwargs) -> 'CADClassificationDetector':
        X, y = self._validate_input(X, y)

        # compute p(y)
        y_vals, y_counts = np.unique(y, return_counts=True)
        y_counts = y_counts / len(y)
        self._p_y = {int(v): c for v, c in zip(y_vals, y_counts)}

        # estimate F(X|y)
        self._estimators = {}
        for y_i in y_vals:
            y_i = int(y_i)

            where_y_i = np.where(y == y_i)[0]
            X_y_i = X[where_y_i]

            estimator = self._init_estimator(X_y_i)
            estimator.fit(X_y_i, **kwargs)

            self._estimators[y_i] = estimator

        if self.with_f_x or self.nan_percentile is not None:
            # estimate F(X)
            self._f_x = self._init_estimator(X)
            self._f_x.fit(X)

        # determine threshold
        scores = self.score_samples(X, y)
        scores.sort()

        thresh_idx = round(self.epsilon * len(X))
        self.threshold_ = scores[thresh_idx]

        return self

    # noinspection PyMethodOverriding
    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X, y)

    def predict(self, X, y, **kwargs):
        scores = self.score_samples(X, y, **kwargs)
        preds = np.ones_like(scores, dtype=int)
        preds[scores < self.threshold_] = -1
        return preds

    def score_samples(self, X, y, **kwargs):
        X, y = self._validate_input(X, y)

        y_vals = np.unique(y)

        scores = np.empty_like(y, dtype=np.float32)
        for y_i in y_vals:
            y_i = int(y_i)

            where_y_i = np.where(y == y_i)[0]
            X_y_i = X[where_y_i]

            estimator = self._estimators[y_i]
            f_x_y_i = estimator.score_samples(X_y_i, **kwargs)

            scores_i = f_x_y_i
            if self.name in {'GMM'}:
                # GMM scores are log-likelihoods
                if self.with_p_y:
                    scores_i = scores_i + np.log(self._p_y[y_i])
                if self.with_f_x:
                    scores_i -= self._f_x.score_samples(X_y_i)
            else:
                if self.with_p_y:
                    scores_i = scores_i * self._p_y[y_i]
                if self.with_f_x:
                    scores_i /= self._f_x.score_samples(X_y_i)
            scores[where_y_i] = scores_i

        if self.nan_percentile is not None:
            cutoff_score = np.percentile(scores, self.nan_percentile)
            scores[scores < cutoff_score] = np.nan

        return scores
