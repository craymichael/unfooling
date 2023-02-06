import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import OutlierMixin
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import brier_score_loss


class NaiveBayes(OutlierMixin, BaseEstimator):

    def __init__(
            self,
            epsilon=0.1,
            categorical_feature_idxs=None,
            alpha=0.5,
            **kwargs,
    ):
        self.epsilon = epsilon
        self.threshold_ = None

        if categorical_feature_idxs is not None:
            from mixed_naive_bayes import MixedNB

            self._estimator = MixedNB(
                categorical_features=categorical_feature_idxs,
                alpha=alpha,
                **kwargs,
            )
        else:
            self._estimator = GaussianNB(
                **kwargs,
            )
        if categorical_feature_idxs is None:
            categorical_feature_idxs = []
        self.categorical_feature_idxs = categorical_feature_idxs

    @staticmethod
    def _standardize_y(y):
        if y.ndim == 2:
            y = y.squeeze(axis=1)
        if y.ndim != 1:
            raise ValueError(f'y.ndim = {y.ndim} but expected 1')
        return y

    def fit(self, X, y, **kwargs) -> 'NaiveBayes':
        y = self._standardize_y(y)

        self._estimator.fit(X, y, **kwargs)

        # determine threshold
        scores = self.score_samples(X, y)
        scores.sort()

        thresh_idx = round(self.epsilon * len(X))
        self.threshold_ = scores[thresh_idx]

        clf_score = brier_score_loss(y, scores)
        print(f'Brier Loss (train split): {clf_score}')

        return self

    def score_samples(self, X, y):
        y = self._standardize_y(y)
        probas = self._estimator.predict_proba(X)
        scores = probas[np.arange(len(y)), y]
        return scores

    def predict(self, X, y, ret_scores=False):
        scores = self.score_samples(X, y)
        preds = np.ones(len(X))
        preds[scores < self.threshold_] = -1
        if ret_scores:
            return preds, scores
        return preds
