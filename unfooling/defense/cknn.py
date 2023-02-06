import numpy as np

from sklearn.base import OutlierMixin
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors


class KNNCAD(OutlierMixin, BaseEstimator):
    def __init__(
            self,
            approach='distance',
            distance_agg=np.median,
            epsilon=0.05,
            n_neighbors=20,
            weights='distance',  # sklearn default: 'uniform'
            algorithm='auto',
            distance_penalize_accurate=True,
            leaf_size=None,
            p=2,
            metric=None,
            metric_params=None,
            n_jobs=-1,
            #
            bandwidth=1.0,
            atol=0,
            rtol=0,
            breadth_first=True,
            #
            radius=1.0,
    ):
        if approach == 'clf':
            metric = metric or 'minkowski'
            leaf_size = leaf_size or 30
            self._estimator = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs,
            )
        elif approach == 'distance':
            metric = metric or 'minkowski'
            leaf_size = leaf_size or 30

            estimator_cls = NearestNeighbors
            self._estimator = estimator_cls(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                metric=metric,
                metric_params=metric_params,
                radius=radius,
                n_jobs=n_jobs,
            )
        elif approach == 'density':
            metric = metric or 'euclidean'
            leaf_size = leaf_size or 40
            self._estimator = KernelDensity(
                bandwidth=bandwidth,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                metric_params=metric_params,
                atol=atol,
                rtol=rtol,
                breadth_first=breadth_first,
            )
        else:
            raise ValueError(approach)

        self.approach = approach
        self.epsilon = epsilon
        if isinstance(distance_agg, str):
            distance_agg = getattr(np, distance_agg)
        self.distance_agg = distance_agg  # when approach='distance'
        self.distance_penalize_accurate = distance_penalize_accurate

        self.threshold_ = None

    def fit(self, X, y):
        if y.ndim == 2:
            y = y.squeeze(axis=1)
        assert y.ndim == 1, y.ndim

        self.X_train, self.y_train = X, y
        self._estimator.fit(X, y)

        # determine threshold
        scores = self.score_samples(X, y)
        scores.sort()

        thresh_idx = round(self.epsilon * len(X))
        self.threshold_ = scores[thresh_idx]

        return self

    # noinspection PyMethodOverriding
    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X, y)

    def predict(self, X, y):
        scores = self.score_samples(X, y)
        preds = np.ones_like(scores, dtype=int)
        preds[scores < self.threshold_] = -1
        return preds

    def score_samples(self, X, y):
        if y.ndim == 2:
            y = y.squeeze(axis=1)
        assert y.ndim == 1, y.ndim

        if self.approach == 'clf':
            self._estimator: KNeighborsClassifier

            probas = self._estimator.predict_proba(X)
            scores = probas[np.arange(len(y)), y]
        elif self.approach == 'distance':
            self._estimator: NearestNeighbors

            dists, idxs = self._estimator.kneighbors(X)

            scores = []
            for dist, idx, yi in zip(dists, idxs, y):
                yi_ = int(yi)
                assert yi == yi_, 'ints only please'
                yi = yi_
                assert yi in {0, 1}, yi  # binary only

                pop_yi = self.y_train[idx]
                where_y_is_0 = np.where(pop_yi == 0)
                dists_0 = dist[where_y_is_0]
                if dists_0.size:
                    dist_0 = self.distance_agg(dists_0)
                else:
                    # no support
                    dist_0 = np.inf

                where_y_is_1 = np.where(pop_yi == 1)
                dists_1 = dist[where_y_is_1]
                if dists_1.size:
                    dist_1 = self.distance_agg(dists_1)
                else:
                    # no support
                    dist_1 = np.inf

                # larger score --> inlier
                if yi == 1:
                    dist_yi, dist_other = dist_1, dist_0
                else:  # yi == 0
                    dist_yi, dist_other = dist_0, dist_1

                # logit^{-1} = sigmoid
                # score derivation:
                # score = sigmoid(dynamic_range(dist_{other}, dist_{yi}))
                #       = sigmoid(log(dist_{other} / dist_{yi}))
                #       = 1 / (1 + exp(-log(dist_{other} / dist_{yi})))
                #       = 1 / (1 + exp(log(dist_{yi} / dist_{other})))
                #       = 1 / (1 + dist_{yi} / dist_{other})
                if not self.distance_penalize_accurate and dist_yi < dist_other:
                    # how about let's not penalize if k-NN classifier would
                    #  have been accurate
                    score = 1.
                else:
                    if dist_yi == dist_other == 0:
                        score = 1.  # default in the invalid case
                    else:
                        with np.errstate(divide='ignore'):
                            score = 1 / (1 + dist_yi / dist_other)
                scores.append(score)
            scores = np.asarray(scores)
        elif self.approach == 'density':
            self._estimator: KernelDensity

            scores = self._estimator.score_samples(X)
        else:
            raise RuntimeError(self.approach)
        return scores
