import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.base import OutlierMixin, BaseEstimator
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def logdotexp(A, B):
    """
    kudos:
    https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy
    """
    max_A = np.max(A)
    max_B = np.max(B)

    exp_A = A - max_A
    np.exp(exp_A, out=exp_A)
    exp_B = B - max_B
    np.exp(exp_B, out=exp_B)

    C = np.dot(exp_A, exp_B)
    # Following scipy.special._logsumexp.py
    with np.errstate(divide='ignore'):
        np.log(C, out=C)
    C += max_A + max_B
    return C


class GMMCADFull(OutlierMixin, BaseEstimator):
    """Implementation of the GMM-CAD-Full algorithm.

    Song, Xiuyao, et al. "Conditional anomaly detection." IEEE Transactions on
    knowledge and Data Engineering 19.5 (2007): 631-645.
    """

    def __init__(
            self,
            num_gaussians=10,
            epsilon=0.1,
            max_iters=1000,
            random_state=None,
            patience=None,
            **kwargs,
    ):
        self.n_U = self.n_V = num_gaussians
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon
        self.patience = patience
        self.max_iters = max_iters
        self.kwargs = kwargs
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.rs = random_state

        self.d_U = self.d_V = None
        self.p_U = self.log_p_V_U = self.Z = self.U = self.V = None

        self.threshold_ = None

    def fit(self, X, y) -> 'GMMCADFull':
        assert X.ndim == y.ndim == 2
        assert len(X) == len(y)
        Xy = np.concatenate([X, y], axis=1)

        n = len(Xy)
        self.d_U = X.shape[1]
        self.d_V = y.shape[1]

        # === Learn a set of n_U Gaussians, Z, over the dataset ===
        self.Z = GaussianMixture(
            n_components=self.n_U,
            random_state=self.rs,
            **self.kwargs,
        )
        self.Z.fit(Xy)

        # === Determine U and V ===
        p_U = self.Z.weights_

        # === THE OLD ===
        mu_U = self.Z.means_[:, :self.d_U]
        mu_V = self.Z.means_[:, self.d_U:]

        Sigma_U = self.Z.covariances_[:, :self.d_U, :self.d_U]
        Sigma_V = self.Z.covariances_[:, self.d_U:, self.d_U:]

        self.U = [multivariate_normal(mean=mu_U_i, cov=Sigma_U_i)
                  for mu_U_i, Sigma_U_i in zip(mu_U, Sigma_U)]
        self.V = [multivariate_normal(mean=mu_V_i, cov=Sigma_V_i)
                  for mu_V_i, Sigma_V_i in zip(mu_V, Sigma_V)]

        # === Learn the mapping function ===
        # f_G_xU: n_U x n
        log_f_G_xU = np.asarray([U_i.logpdf(X) for U_i in self.U])
        # f_G_xU: n_V x n
        log_f_G_yV = np.asarray([V_i.logpdf(y) for V_i in self.V])

        # compute values reused in the loop each iteration
        # element-wise outer product. shape: n_U x n_V x n
        b_shape = (self.n_U, self.n_V, n)
        log_f_G_xU_yV = (
                np.broadcast_to(log_f_G_xU[:, None, :], b_shape) +
                np.broadcast_to(log_f_G_yV[None, :, :], b_shape)
        )
        log_f_G_xU_yV_p_U = log_f_G_xU_yV + np.log(p_U[:, None, None])

        # compute f(x_k | \bar{\Theta})
        # shape: n
        log_f_x = logsumexp(a=log_f_G_xU, b=p_U[:, None], axis=0)

        # Initialize p(V | U) randomly
        p_V_U = self.rs.rand(self.n_U, self.n_V)
        p_V_U /= p_V_U.sum(axis=1)[:, None]
        log_p_V_U = np.log(p_V_U)

        Lambda_prev = -np.inf
        Lambda_best = log_p_V_U_best = None
        patience_count = 0
        pbar = tqdm(range(self.max_iters))
        for iter_ in pbar:
            # compute b_kij for all k i j
            # shape: n_U x n_V x n
            log_b_numerator = log_f_G_xU_yV_p_U + log_p_V_U[:, :, None]
            log_b_denominator = logsumexp(a=log_b_numerator, axis=(0, 1))
            log_b = log_b_numerator - log_b_denominator[None, None, :]
            b = np.exp(log_b)

            # compute \bar{p(V_j|U_i)} and \bar{p(U_i)}
            log_b_sum_ax2 = logsumexp(log_b, axis=2)
            log_b_sum_ax12 = logsumexp(log_b, axis=(1, 2))
            log_b_sum = logsumexp(log_b, axis=(0, 1, 2))

            # shape: n_U x n_V
            log_p_V_U_bar = log_b_sum_ax2 - log_b_sum_ax12[:, None]

            # shape: n_U
            log_p_U_bar = log_b_sum_ax12 - log_b_sum

            # compute log likelihood Lambda
            Lambda = (
                    (log_f_G_yV + log_p_V_U_bar[:, :, None]
                     + log_f_G_xU + log_p_U_bar[:, None, None]
                     - log_f_x[None, None, :]) * b
            ).sum()

            # p(V_j|U_i) = \bar{p(V_j|U_i)}
            log_p_V_U = log_p_V_U_bar

            pbar.set_description(f'Lambda = {Lambda}')
            if Lambda < (Lambda_best or Lambda_prev):
                if self.patience and patience_count < self.patience:
                    patience_count += 1
                else:
                    print(f'Converged in {iter_ + 1} iterations (Lambda = '
                          f'{Lambda}, Lambda_prev={Lambda_prev})')
                    if log_p_V_U_best is not None:
                        log_p_V_U = log_p_V_U_best
                    break
            else:
                patience_count = 0
                log_p_V_U_best = log_p_V_U
                Lambda_best = Lambda

            Lambda_prev = Lambda
        else:
            print(f'Did not converge in {self.max_iters} iterations')

        self.log_p_V_U = log_p_V_U
        self.p_U = p_U

        # determine threshold
        scores = self.score_samples(X, y)
        scores.sort()

        thresh_idx = round(self.epsilon * len(X))
        self.threshold_ = scores[thresh_idx]

        return self

    def score_samples(self, X, y):
        # f_G_xU: n_U x n
        log_f_G_xU = np.asarray([U_i.logpdf(X) for U_i in self.U])
        # f_G_xU: n_V x n
        log_f_G_yV = np.asarray([V_i.logpdf(y) for V_i in self.V])

        # shape: n_U x n
        log_p_V_U_dot_f_G_yV = logdotexp(self.log_p_V_U, log_f_G_yV)

        # shape: n_U x n
        log_p_x_in_U = log_f_G_xU + np.log(self.p_U[:, None])

        # logsumexp: log(sum(exp(a), axis=axis))
        log_f_CAD = (log_p_x_in_U + log_p_V_U_dot_f_G_yV
                     - logsumexp(a=log_p_x_in_U, axis=0))
        f_CAD = np.sum(np.exp(log_f_CAD), axis=0)

        return f_CAD

    def predict(self, X, y, ret_scores=False):
        scores = self.score_samples(X, y)
        preds = np.ones(len(X))
        preds[scores < self.threshold_] = -1
        if ret_scores:
            return preds, scores
        return preds
