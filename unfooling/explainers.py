"""
explainers.py - A Unfooling-LIME-SHAP file
Copyright (C) 2022  Zach Carmichael
"""
import warnings
from typing import Optional
from typing import Union

from tqdm import tqdm

import numpy as np
import scipy as sp
import scipy.spatial
import scipy.sparse

import shap
from shap import KernelExplainer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.mixture import GaussianMixture

from fooling.adversarial_models import AdversarialLimeModel
from fooling.adversarial_models import BaseAdversarialModel
from fooling.models import SimpleRegressionModel
from fooling.models import SimpleClassificationModel


def get_explainer(name, model, X_train, features, categorical_feature_idxs,
                  **kwargs):
    if name == 'LIME':
        if isinstance(model, SimpleClassificationModel):
            predict_func = model.predict_proba
            mode = 'classification'
        elif isinstance(model, SimpleRegressionModel):
            predict_func = model.predict
            mode = 'regression'
        else:
            try:
                result = model.predict_proba(X_train[:1])
                predict_func_ = model.predict_proba
                mode = 'classification'
            except (NotImplementedError, TypeError, AttributeError):
                result = model.predict(X_train[:1])
                predict_func_ = model.predict
                mode = 'regression'
            if isinstance(result, tuple) and len(result) == 2:
                def predict_func(x, *args_, **kwargs_):
                    return predict_func_(x, *args_, **kwargs_)[0]
            else:
                predict_func = predict_func_

        explainer = RobustLimeTabularExplainer(
            X_train,
            **kwargs,
            feature_names=features,
            discretize_continuous=False,
            categorical_features=categorical_feature_idxs,
            mode=mode,
        )

        class ExplainerProxy:
            @staticmethod
            def explain(x, num_samples=5000):
                kwargs_ = {}
                if num_samples:
                    kwargs_ = {'num_samples': num_samples}
                return explainer.explain_instance(
                    x, predict_func, **kwargs_).as_list()

            @property
            def perturbed_data(self):
                return explainer.sampled_data

    elif name == 'SHAP':
        predict_func_ = model.predict
        result = predict_func_(X_train[:1])
        if isinstance(result, tuple) and len(result) == 2:
            def predict_func(x, *args_, **kwargs_):
                return predict_func_(x, *args_, **kwargs_)[0]
        else:
            predict_func = predict_func_

        background_distribution = shap.kmeans(X_train, 20)
        explainer = RobustSHAPKernelExplainer(
            predict_func,
            background_distribution,
            silent=True,
            **kwargs,
        )

        class ExplainerProxy:
            @staticmethod
            def explain(x, num_samples=None):
                kwargs_ = {}
                if num_samples:
                    kwargs_ = {'nsamples': num_samples}
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        explanation_raw = explainer.shap_values(x, **kwargs_)
                    except FloatingPointError:
                        import traceback
                        traceback.print_exc()
                        tqdm.write(f'FAILED {x}')
                        explanation_raw = x
                    return [(features[i], explanation_raw[i])
                            for i in range(len(explanation_raw))]

            @property
            def perturbed_data(self):
                return explainer.synth_data

    else:
        raise ValueError(name)

    return ExplainerProxy()


class RobustLimeTabularExplainer(LimeTabularExplainer):
    def __init__(
            self,
            training_data,
            *args,
            robustness_model: Optional[Union[AdversarialLimeModel,
                                             GaussianMixture]] = None,
            sigma=1,
            oversample_factor=2,
            **kwargs,
    ):
        super(RobustLimeTabularExplainer, self).__init__(
            training_data, *args, **kwargs)
        self.robustness_model = robustness_model
        self.sigma = sigma
        self.training_data = training_data
        self.scaled_data = self.scaler.transform(training_data)
        self.predict_fn = None
        self.oversample_factor = oversample_factor

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         *args,
                         num_samples=5000,
                         **kwargs):
        self.predict_fn = predict_fn
        return super().explain_instance(data_row,
                                        predict_fn,
                                        *args,
                                        num_samples=num_samples,
                                        **kwargs)

    # noinspection PyPep8Naming
    def _LimeTabularExplainer__data_inverse(self,
                                            data_row,
                                            num_samples):
        """
        Modified version of `LimeTabularExplainer.__data_inverse`, lime version
        0.2.0.1

        Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        is_sampler = isinstance(self.robustness_model, GaussianMixture)
        is_adversary = isinstance(self.robustness_model, BaseAdversarialModel)
        is_defense = not (is_sampler or is_adversary or
                          self.robustness_model is None)

        if sp.sparse.issparse(data_row):
            raise NotImplementedError('sparse input not supported')
        else:
            num_cols = data_row.shape[0]
        if self.discretizer is not None:
            raise NotImplementedError('discretizer not supported')

        numeric_cols = [i for i in range(num_cols)
                        if i not in self.categorical_features]
        num_samples_start = num_samples
        if is_sampler:
            # get k points with smallest distances
            top_k_pct = .05
            top_k = max(min(10, self.scaled_data.shape[0]),
                        round(top_k_pct * self.scaled_data.shape[0]))
            dists = scipy.spatial.distance.cdist(
                [data_row], self.scaled_data).squeeze(axis=0)
            gm_train_data_local = self.training_data[
                np.argpartition(dists, top_k)[:top_k]]
            gm = GaussianMixture(n_components=10).fit(gm_train_data_local)

            data = gm.sample(num_samples_start)[0]
            first_row = data_row
        else:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_defense:
                num_samples_start *= self.oversample_factor
            data = self.random_state.normal(
                0, self.sigma, num_samples_start * num_cols).reshape(
                num_samples_start, num_cols)
            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            first_row = data_row
        data[0] = data_row.copy()
        inverse = data.copy()

        for column in self.categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values,
                                                      size=num_samples_start,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        inverse[0] = data_row

        if self.robustness_model is None or is_sampler:
            self.sampled_data = data
            return data, inverse
        # detect off-manifold samples
        if is_adversary:
            # perturbed data minus categorical columns
            adv_detect_data = inverse[1:, numeric_cols]
            ood_predictor = self.robustness_model.perturbation_identifier
            is_on_manifold_ = ood_predictor.predict(adv_detect_data)
            is_on_manifold = np.zeros_like(is_on_manifold_, dtype=bool)
            is_on_manifold[is_on_manifold_ == 1] = True
        elif is_defense:
            # perturbed data minus categorical columns
            adv_detect_data = inverse[1:]
            adv_detect_preds = self.predict_fn(adv_detect_data)
            if adv_detect_preds.ndim == 2:
                if adv_detect_preds.shape[1] == 1:
                    adv_detect_preds = adv_detect_preds.squeeze(axis=1)
                else:
                    adv_detect_preds = np.argmax(adv_detect_preds, axis=1)
            off_manifold_scores = self.robustness_model.score_samples(
                adv_detect_data, adv_detect_preds)
            is_on_manifold = np.zeros(len(adv_detect_data), dtype=bool)

            threshold = .9
            top_idxs = np.where(off_manifold_scores >= threshold)[0]

            is_on_manifold[top_idxs] = True
        else:
            raise RuntimeError('This condition should not be hit.')

        final_data = np.concatenate([data_row[None, :],
                                     data[1:][is_on_manifold]], axis=0)
        final_inverse = np.concatenate([data_row[None, :],
                                        inverse[1:][is_on_manifold]],
                                       axis=0)

        if len(final_data) < num_samples:
            num_samples_remaining = num_samples - len(final_data)
            (remaining_data,
             remaining_inverse) = self._LimeTabularExplainer__data_inverse(
                data_row, num_samples_remaining + 1)
            final_data = np.concatenate([final_data, remaining_data[1:]],
                                        axis=0)
            final_inverse = np.concatenate([final_inverse,
                                            remaining_inverse[1:]], axis=0)
        elif len(final_data) > num_samples:
            if is_defense and SCORE_SAMPLES:
                # take the highest-scoring samples
                top_k = num_samples - 1
                top_idxs_ = np.concatenate([[0], np.argpartition(
                    off_manifold_scores[top_idxs], -top_k)[-top_k:] + 1],
                                           axis=0)
                final_data = final_data[top_idxs_]
                final_inverse = final_inverse[top_idxs_]
            else:
                final_data = final_data[:num_samples]
                final_inverse = final_inverse[:num_samples]

        self.sampled_data = final_data
        return final_data, final_inverse


class RobustSHAPKernelExplainer(KernelExplainer):
    def __init__(
            self,
            *args,
            robustness_model=None,
            **kwargs,
    ):
        super(RobustSHAPKernelExplainer, self).__init__(*args, **kwargs)
        self.robustness_model = robustness_model

    # noinspection PyProtectedMember,PyAttributeOutsideInit
    def explain(self, incoming_instance, **kwargs):
        """Modified version of `KernelExplainer.explain`, shap version 0.40.0"""
        import pandas as pd
        from shap.utils._legacy import convert_to_instance
        from shap.utils._legacy import match_instance_to_data

        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        # find the feature groups we will test. If a feature does not change
        # from its current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            self.varyingFeatureGroups = [self.data.groups[i] for i in
                                         self.varyingInds]
            self.M = len(self.varyingFeatureGroups)
            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array
            # (all groups of same length)
            if self.varyingFeatureGroups and all(
                    len(groups[i]) == len(groups[0]) for i in self.varyingInds):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a
                # single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = (
                        self.varyingFeatureGroups.flatten())

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

            # pick a reasonable number of samples if the user didn't specify
            # how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2 ** 11

            # if we have enough samples to enumerate all subsets then ignore
            # the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            self.generate_synth_samples(instance)

            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to
            # contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            for d in range(self.D):
                vphi, _ = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)

        return phi

    # noinspection PyProtectedMember,PyAttributeOutsideInit
    def generate_synth_samples(self, instance):
        import copy
        import itertools
        from scipy.special import binom

        from contextlib import contextmanager

        @contextmanager
        def dummy(name):
            try:
                yield
            finally:
                pass

        timer = dummy

        with timer('init'):
            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes
            num_subset_sizes = np.int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = np.int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array(
                [(self.M - 1.0) / (i * (self.M - i))
                 for i in range(1, num_subset_sizes + 1)]
            )
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            addsample_x, addsample_m, addsample_w = [], [], []
        with timer('subsets'):
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the
                # current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    nsubsets *= 2

                # see if we have enough samples to enumerate all subsets of this
                # size
                if (num_samples_left * remaining_weight_vector[subset_size - 1]
                        / nsubsets >= 1.0 - 1e-8):
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum
                    # to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (
                                1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M,
                                                               subset_size)
                    if subset_size <= num_paired_subset_sizes:
                        w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        if self.robustness_model is None:
                            self.addsample(instance.x, mask, w)
                        else:
                            addsample_x.append(instance.x)
                            addsample_m.append(mask.copy())
                            addsample_w.append(w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            if self.robustness_model is None:
                                self.addsample(instance.x, mask, w)
                            else:
                                addsample_x.append(instance.x)
                                addsample_m.append(mask.copy())
                                addsample_w.append(w)
                else:
                    break
        if self.robustness_model is not None:
            with timer('add_samples subsets'):
                offset = offset0 = self.nsamplesAdded * self.N
                synth_data_cpy = self.synth_data.copy()
                for instance_x, mask, w in zip(addsample_x, addsample_m,
                                               addsample_w):
                    mask = mask == 1.0
                    groups = self.varyingFeatureGroups[mask]
                    evaluation_data = instance_x[0, groups]
                    offset_next = offset + self.N
                    synth_data_cpy[offset:offset_next, groups] = evaluation_data
                    offset = offset_next

                if len(addsample_x):
                    synth_data_cpy = synth_data_cpy[offset0:offset_next]
                    modelOut = self.model.f(synth_data_cpy)
                    off_manifold_scores_all = self.robustness_model.score_samples(
                        synth_data_cpy, modelOut)
                    off_manifold_scores_all = off_manifold_scores_all.reshape(
                        len(addsample_x), self.N)
                    for off_manifold_scores, instance_x, mask, w in zip(
                            off_manifold_scores_all, addsample_x,
                            addsample_m, addsample_w
                    ):
                        self.addsample(instance_x, mask, w, off_manifold_scores)

        # add random samples from what is left of the subset space
        nfixed_samples = self.nsamplesAdded
        samples_left = self.nsamples - self.nsamplesAdded
        if num_full_subsets != num_subset_sizes:
            remaining_weight_vector = copy.copy(weight_vector)
            # because we draw two samples each below
            remaining_weight_vector[:num_paired_subset_sizes] /= 2
            remaining_weight_vector = (
                remaining_weight_vector[num_full_subsets:])
            remaining_weight_vector /= np.sum(remaining_weight_vector)
            ind_set = np.random.choice(len(remaining_weight_vector),
                                       4 * samples_left,
                                       p=remaining_weight_vector)
            ind_set_pos = 0
            used_masks = {}
            with timer(f'remaining {samples_left} samples'):
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    # we call np.random.choice once to save time and then just
                    # read it here
                    ind = ind_set[ind_set_pos]
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before,
                    # otherwise just increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        if self.addsample(instance.x, mask, 1.0):
                            samples_left -= 1
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if (samples_left > 0 and
                            subset_size <= num_paired_subset_sizes):
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before,
                        # otherwise just increment a previous sample's weight
                        if new_sample:
                            if self.addsample(instance.x, mask, 1.0):
                                samples_left -= 1
                        else:
                            # we know the complement sample is the next one
                            # after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.

            # normalize the kernel weights for the random samples to equal
            # the weight left after the fixed enumerated samples have been
            # already counted
            weight_left = np.sum(weight_vector[num_full_subsets:])
            self.kernelWeights[nfixed_samples:] *= (
                    weight_left / self.kernelWeights[nfixed_samples:].sum())

    def addsample(self, x, m, w, off_manifold_scores=None):
        offset = self.nsamplesAdded * self.N
        if isinstance(self.varyingFeatureGroups, (list,)):
            raise NotImplementedError('jagged numpy arrays')
        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m == 1.0
            groups = self.varyingFeatureGroups[mask]
            if len(groups.shape) == 2:
                raise NotImplementedError('groups.ndim == 2')
            else:
                # further performance optimization in case each group has a
                # single feature
                evaluation_data = x[0, groups]
                # In edge case where background is all dense but evaluation data
                # is all sparse, make evaluation data dense
                if sp.sparse.issparse(x) and not sp.sparse.issparse(
                        self.synth_data):
                    evaluation_data = evaluation_data.toarray()

                if self.robustness_model is not None:
                    samples_to_test = None
                    if off_manifold_scores is None:
                        samples_to_test = self.synth_data[
                                          offset:offset + self.N]
                        samples_to_test = samples_to_test.copy()
                        samples_to_test[:, groups] = evaluation_data
                        modelOut = self.model.f(samples_to_test)
                        off_manifold_scores = self.robustness_model.score_samples(
                            samples_to_test, modelOut)
                    threshold = .75
                    if samples_to_test is None:
                        samples_to_test = self.synth_data[
                                          offset:offset + self.N]
                        samples_to_test = samples_to_test.copy()
                        samples_to_test[:, groups] = evaluation_data
                    good_samples_to_test = samples_to_test[
                        off_manifold_scores >= threshold]
                    if ((len(good_samples_to_test) / len(samples_to_test))
                            < .3):
                        return False
                    # - w stays the same
                    # - m stays the same
                    # - resample self.synth_data where off-manifold
                    good_samples_to_test = np.concatenate([
                        good_samples_to_test,
                        good_samples_to_test[np.random.choice(
                            np.arange(len(good_samples_to_test)),
                            size=(len(samples_to_test) -
                                  len(good_samples_to_test)),
                            replace=True
                        )]
                    ], axis=0)
                    self.synth_data[offset:offset + self.N] = (
                        good_samples_to_test)
                else:
                    self.synth_data[offset:offset + self.N, groups] = (
                        evaluation_data)
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1
        return True

    def solve(self, fraction_evaluated, dim):
        from sklearn.linear_model import LassoLarsIC, Lasso, lars_path

        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])
        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        if (self.l1_reg not in ["auto", False, 0]) or (
                fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack(
                (self.kernelWeights * (self.M - s),
                 self.kernelWeights * s)
            )
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((
                eyAdj,
                eyAdj - self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])
            ))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(
                np.vstack((self.maskMatrix, self.maskMatrix - 1))))
            # select a fixed number of top features
            if isinstance(self.l1_reg, str) and self.l1_reg.startswith(
                    "num_features("):
                r = int(self.l1_reg[len("num_features("):-1])
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1]
            # use an adaptive regularization method
            elif self.l1_reg == "auto" or self.l1_reg == "bic" or self.l1_reg == "aic":
                c = "aic" if self.l1_reg == "auto" else self.l1_reg
                nonzero_inds = np.nonzero(
                    LassoLarsIC(criterion=c).fit(mask_aug, eyAdj_aug).coef_)[0]
            # use a fixed regularization coefficient
            else:
                nonzero_inds = np.nonzero(
                    Lasso(alpha=self.l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to
        #  the output
        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
                self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))
        etmp = np.transpose(
            np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) -
            self.maskMatrix[:, nonzero_inds[-1]]
        )

        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(
            np.transpose(etmp) * np.transpose(self.kernelWeights))
        etmp_dot = np.dot(np.transpose(tmp), etmp)
        try:
            tmp2 = np.linalg.inv(etmp_dot)
        except np.linalg.LinAlgError:
            tmp2 = np.linalg.pinv(etmp_dot)
            warnings.warn(SINGULAR_WARNING)
        w = np.dot(tmp2, np.dot(np.transpose(tmp), eyAdj2))
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(
            self.fnull[dim])) - sum(w)

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))


SINGULAR_WARNING = (
    "Linear regression equation is singular, Moore-Penrose pseudoinverse is used instead of the regular inverse.\n"
    "To use regular inverse do one of the following:\n"
    "1) turn up the number of samples,\n"
    "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
    "3) group features together to reduce the number of inputs that need to be explained."
)
