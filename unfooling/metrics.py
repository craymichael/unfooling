from typing import Sequence
from itertools import chain

import numpy as np
import scipy.interpolate


def _cdf_base(scores_explainer, scores_natural, robust=True, plot=False,
              plot_title=None, normalized=False,
              independent_normalization=False):
    from statsmodels.distributions.empirical_distribution import ECDF

    if robust:
        lower_quantile = 0
        upper_quantile = .9
        combine_all = False

        if combine_all:
            all_scores = np.concatenate([scores_explainer, scores_natural],
                                        axis=0)
            s_min, s_max = np.quantile(all_scores,
                                       [lower_quantile, upper_quantile])
        else:
            s_min, s_max = np.quantile(scores_explainer,
                                       [lower_quantile, upper_quantile])

        scores_explainer = scores_explainer[(scores_explainer >= s_min) &
                                            (scores_explainer <= s_max)]

        if not combine_all:
            s_min, s_max = np.quantile(scores_natural,
                                       [lower_quantile, upper_quantile])

        scores_natural = scores_natural[(scores_natural >= s_min) &
                                        (scores_natural <= s_max)]

    if normalized:
        if independent_normalization:
            scores_explainer = scores_explainer / scores_explainer.max()
            scores_natural = scores_natural / scores_natural.max()
        else:
            max_score = max(scores_natural.max(), scores_explainer.max())
            scores_explainer = scores_explainer / max_score
            scores_natural = scores_natural / max_score

    scores_explainer = np.sort(scores_explainer)
    scores_natural = np.sort(scores_natural)

    p1 = ECDF(scores_explainer)(scores_explainer)
    p2 = ECDF(scores_natural)(scores_natural)

    assert p1[-1] == p2[-1] == 1, (p1[-1], p2[-1])

    # ensure CDF bounds are the same artificially
    if scores_explainer[-1] > scores_natural[-1]:
        # max explainer score exceeds that of natural
        scores_natural = np.append(scores_natural, scores_explainer[-1])
        p2 = np.append(p2, 1)
    elif scores_explainer[-1] < scores_natural[-1]:
        # max natural score exceeds that of explainer
        scores_explainer = np.append(scores_explainer, scores_natural[-1])
        p1 = np.append(p1, 1)

    if scores_explainer[0] > scores_natural[0]:
        # min explainer score exceeds that of natural
        scores_explainer = np.append(scores_natural[0], scores_explainer)
        p1 = np.append(0, p1)
    elif scores_explainer[0] < scores_natural[0]:
        # min natural score exceeds that of explainer
        scores_natural = np.append(scores_explainer[0], scores_natural)
        p2 = np.append(0, p2)

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        df = pd.DataFrame({
            'Score': np.concatenate([scores_explainer, scores_natural], axis=0),
            'Split': ([f'Explainer'] * len(scores_explainer) +
                      [f'Natural'] * len(scores_natural)),
            'Proportion': np.concatenate([p1, p2], axis=0),
        })

        f, ax = plt.subplots()

        ax = sns.lineplot(data=df,
                          x='Score',
                          y='Proportion',
                          hue='Split',
                          ax=ax)

        if plot_title:
            ax.set_title(plot_title)

    return scores_explainer, scores_natural, p1, p2


def cdf_delta(scores_explainer, scores_natural, robust=False, ret_areas=False,
              plot=False, cdf_data=None, return_cdf_data=False, **kwargs):
    if cdf_data is not None:
        assert scores_explainer is None and scores_natural is None
        scores_explainer, scores_natural, p1, p2 = cdf_data
    else:
        scores_explainer, scores_natural, p1, p2 = _cdf_base(
            scores_explainer, scores_natural, robust=robust, plot=plot,
            **kwargs
        )

    a1 = np.trapz(x=scores_explainer, y=p1)
    a2 = np.trapz(x=scores_natural, y=p2)

    delta = a1 - a2

    if return_cdf_data:
        if ret_areas:
            return delta, a1, a2, (scores_explainer, scores_natural, p1, p2)
        return delta, (scores_explainer, scores_natural, p1, p2)
    if ret_areas:
        return delta, a1, a2
    # otherwise
    return delta


def cdf_delta_ratio(scores_explainer, scores_natural, robust=False,
                    ret_areas=False, plot=False, cdf_data=None,
                    return_cdf_data=False, **kwargs):
    if cdf_data is not None:
        assert scores_explainer is None and scores_natural is None
        scores_explainer, scores_natural, p1, p2 = cdf_data
    else:
        scores_explainer, scores_natural, p1, p2 = _cdf_base(
            scores_explainer, scores_natural, robust=robust, plot=plot,
            **kwargs
        )

    a1 = np.trapz(x=scores_explainer, y=p1)
    a2 = np.trapz(x=scores_natural, y=p2)

    delta_ratio = a1 / a2

    if return_cdf_data:
        if ret_areas:
            return delta_ratio, a1, a2, (
                scores_explainer, scores_natural, p1, p2)
        return delta_ratio, (scores_explainer, scores_natural, p1, p2)
    if ret_areas:
        return delta_ratio, a1, a2
    # otherwise
    return delta_ratio


def _process_labels(y, name):
    y = np.asarray(y)
    y_int = y.astype(int)
    if (y_int != y).all():
        raise ValueError(f'{name} must be all int, but received '
                         f'{y.dtype}')
    if y_int.ndim == 2:
        y_int = y_int.squeeze(axis=1)
    assert y_int.ndim == 1, (f'{name} should have 1 dim but received '
                             f'{y_int.ndim}')
    return y_int


def peak_deltas(scores_explainer,
                y_explainer,
                agg_func=np.mean):
    y_explainer = _process_labels(y_explainer, 'y_explainer')

    def agg_scores_by_label(scores, y):
        assert scores.ndim == 1
        assert len(y) == len(scores)

        y_unique = np.unique(y)
        scores_map = {}
        for yi in y_unique:
            idxs = np.where(y == yi)[0]
            scores_yi = scores[idxs]
            scores_map[yi] = agg_func(scores_yi) if len(scores_yi) else 0.

        return scores_map

    scores_explainer_map = agg_scores_by_label(scores_explainer, y_explainer)

    # union of all unique labels
    y_joined = sorted(scores_explainer_map)
    all_dists = []
    for i, yi in enumerate(y_joined[:-1]):
        for yj in y_joined[i + 1:]:
            all_dists.append(abs(scores_explainer_map[yi] -
                                 scores_explainer_map[yj]))

    return np.mean(all_dists) if len(all_dists) else np.nan


def pct_cdf_greater(scores_explainer, scores_natural, robust=False, plot=False,
                    cdf_data=None, return_cdf_data=False, **kwargs):
    if cdf_data is not None:
        assert scores_explainer is None and scores_natural is None
        scores_explainer, scores_natural, p1, p2 = cdf_data
    else:
        scores_explainer, scores_natural, p1, p2 = _cdf_base(
            scores_explainer, scores_natural, robust=robust, plot=plot,
            **kwargs
        )

    natural_interp = scipy.interpolate.interp1d(
        x=scores_natural, y=p2, kind='linear', assume_sorted=True)

    p2_interp = natural_interp(scores_explainer)

    pct = (p1 > p2_interp).sum() / p1.size
    if return_cdf_data:
        return pct, (scores_explainer, scores_natural, p1, p2)
    return pct


def cdf_area_above(scores_explainer, scores_natural, robust=False, plot=False,
                   cdf_data=None, return_cdf_data=False, **kwargs):
    if cdf_data is not None:
        assert scores_explainer is None and scores_natural is None
        scores_explainer, scores_natural, p1, p2 = cdf_data
    else:
        scores_explainer, scores_natural, p1, p2 = _cdf_base(
            scores_explainer, scores_natural, robust=robust, plot=plot,
            **kwargs
        )

    natural_interp = scipy.interpolate.interp1d(
        x=scores_natural, y=p2, kind='linear', assume_sorted=True)

    p2_interp = natural_interp(scores_explainer)

    greater_mask = (p1 > p2_interp)
    changes = np.where(greater_mask[:-1] != greater_mask[1:])[0] + 1

    idx_prev = 0
    area = 0
    for idx in chain(changes, [len(scores_explainer)]):
        x = scores_explainer[idx_prev:idx]
        area += (np.trapz(x=x, y=p1[idx_prev:idx]) -
                 np.trapz(x=x, y=p2_interp[idx_prev:idx]))
        idx_prev = idx

    if return_cdf_data:
        return area, (scores_explainer, scores_natural, p1, p2)
    return area


def under_threshold(scores_explainer, threshold):
    is_seq = isinstance(threshold, Sequence)
    if not is_seq:
        threshold = [threshold]
    scores = []
    for thresh in threshold:
        score_th = (scores_explainer <= thresh).sum() / scores_explainer.size
        scores.append(score_th)
    return np.mean(scores)
