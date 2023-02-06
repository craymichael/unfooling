from typing import List
from typing import Union
from typing import Dict

import re
import hashlib
from collections import namedtuple
from pprint import pprint
from pprint import pformat

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

    print('Could not load CSafeLoader.')

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fooling.utils import experiment_summary
from fooling.adversarial_models import get_adversarial_model

from unfooling.utils import timer
from unfooling.explainers import get_explainer
from unfooling.defense import get_detector
from unfooling import metrics

from experiments import get_experiment

ProcessedProblem = namedtuple(
    'ProcessedProblem',
    'problem,features,categorical_features,categorical_feature_idxs,'
    'prejudiced_model,innocuous_models,X_train,X_test,y_train,y_test')


def load_experiment_and_data(C) -> ProcessedProblem:
    problem = get_experiment(C.experiment_name)
    X = problem.X
    y = problem.y
    features = problem.features
    categorical_features = problem.categorical_features

    categorical_feature_idxs = [features.index(name)
                                for name in categorical_features]
    if C.debug:
        X = X[:250]
        y = y[:250]

    prejudiced_model = problem.prejudiced_model
    innocuous_models = [None] + problem.innocuous_models

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.test_size, random_state=0xFACE)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return ProcessedProblem(problem, features, categorical_features,
                            categorical_feature_idxs, prejudiced_model,
                            innocuous_models, X_train, X_test, y_train, y_test)


def make_model_predict_func(model, detect_proba, iname):
    if detect_proba:
        return model.predict_proba
    else:
        def model_predict_func(x):
            predicted = model.predict(x)
            if iname is None:
                return predicted[:, None]
            else:
                predicted, ood_predicted = predicted
                return predicted[:, None], ood_predicted

        return model_predict_func


def generate_explanations(
        C,
        P: ProcessedProblem,
        explainer_names: List[str] = None,
        top_k_features: int = 5,
        robustness_model=None,
        explainer_data=None,
        num_samples_explain=None,
        **kwargs,
):
    if explainer_names is None:
        explainer_names = ['LIME', 'SHAP']

    recompute_expl = explainer_data is None
    if recompute_expl:
        explainer_data = {}
    for explainer_name in explainer_names:
        print('---------------------')
        print(f'Beginning {explainer_name} {C.experiment_name} Experiments....')
        print('---------------------')

        if recompute_expl:
            explainer_data[explainer_name] = {}
        for innocuous_model in P.innocuous_models:
            iname = None if innocuous_model is None else innocuous_model.name
            if recompute_expl:
                if innocuous_model is None:
                    print('Racist model only (not adversarial)')
                    model = P.prejudiced_model
                else:
                    print(f'Adversary with racist model and innocuous model '
                          f'{innocuous_model}')
                    with timer('Adversarial model init'):
                        model = get_adversarial_model(
                            explainer_name=explainer_name,
                            prejudiced_model=P.prejudiced_model,
                            innocuous_model=innocuous_model,
                            X_train=P.X_train,
                            y_train=P.y_train,
                            features=P.features,
                            categorical_feature_idxs=P.categorical_feature_idxs,
                        )
                model_predict_func = make_model_predict_func(
                    model=model,
                    detect_proba=C.detect_proba,
                    iname=iname,
                )

                expl_kwargs = {}
                if robustness_model is not None:
                    robustness_model_trial = robustness_model[explainer_name][
                        iname]
                    expl_kwargs['robustness_model'] = robustness_model_trial

                explainer = get_explainer(
                    explainer_name,
                    model=model,
                    X_train=P.X_train,
                    features=P.features,
                    categorical_feature_idxs=P.categorical_feature_idxs,
                    **expl_kwargs,
                    **kwargs,
                )

                explanations = []
                X_samples = []

                for x_i in tqdm(P.X_test):
                    explanations.append(
                        explainer.explain(x_i, num_samples=num_samples_explain))
                    X_sample = explainer.perturbed_data
                    X_samples.append(X_sample)
                X_samples = np.concatenate(X_samples, axis=0)

                y_test_pred = model_predict_func(P.X_test)
                y_samples_pred = model_predict_func(X_samples)

                if innocuous_model is None:
                    y_test_pred_f_biased = None
                    dood = np.ones(len(y_test_pred), dtype=int)
                    dood_g = np.ones(len(y_samples_pred), dtype=int)
                else:
                    y_test_pred, dood = y_test_pred
                    y_samples_pred, dood_g = y_samples_pred
                    y_test_pred_f_biased = model.f_obscure.predict(P.X_test)

                explainer_data[explainer_name][iname] = {
                    'model_predict_func': model_predict_func,
                    'X_samples': X_samples,
                    'explanations': explanations,
                    'dood': dood,
                    'dood_g': dood_g,
                    'y_test_pred': y_test_pred,
                    'y_samples_pred': y_samples_pred,
                    'y_test_pred_f_biased': y_test_pred_f_biased,
                }
            else:
                explanations = explainer_data[explainer_name][iname][
                    'explanations']
                dood = explainer_data[explainer_name][iname]['dood']
                dood_g = explainer_data[explainer_name][iname]['dood_g']
                y_test_pred = explainer_data[explainer_name][iname][
                    'y_test_pred']
                y_test_pred_f_biased = explainer_data[explainer_name][iname][
                    'y_test_pred_f_biased']

            # Display Results
            n_unrelated = (0 if innocuous_model is None else
                           len(innocuous_model.idxs))
            print(f'{explainer_name} Ranks and top-{top_k_features} Pct '
                  f'Occurrences (1 corresponds to most important feature) for '
                  f'{n_unrelated} unrelated features:')
            pprint(experiment_summary(explanations, P.features,
                                      k=top_k_features))
            if innocuous_model is None:
                fidelity_f = fidelity_dood = 'N/A'
            else:
                fidelity_f = np.mean(y_test_pred.squeeze(axis=1) ==
                                     y_test_pred_f_biased)
                fidelity_dood = ((dood == 1).mean() + (dood_g == 0).mean()) / 2
            print(f'Fidelity (f) = {fidelity_f:.5}')
            print(f'Fidelity (dood) = {fidelity_dood:.5}')

            if C.debug:
                break
        if C.debug:
            break
    return explainer_data


class ResultCollection:
    __slots__ = '_keys', '_data'

    def __init__(self, keys=None):
        self._keys = keys
        self._data = None
        self.reset()

    def as_dict(self):
        return {k: v for k, v in self.items()}

    def keys(self):
        if self._keys is None:
            return self._data.keys()
        else:
            return self._keys

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def reset(self):
        if self._keys is not None:
            assert not any(k in self.__slots__ or k in dir(self)
                           for k in self._keys), f'invalid keys: {self._keys}'
            self._keys = tuple(sorted(self._keys))
            self._data = {k: None for k in self._keys}
        else:
            self._keys = None
            self._data = {}

    def set(self, *args, **kwargs):
        self.reset()
        self.update(*args, **kwargs)

    def get(self, key):
        return self._data.get(key)

    def update(self, *args, **kwargs):
        if len(args) == 1:
            for k, v in args[0].items():
                self[k] = v
        else:
            assert len(args) == 0, f'Expected 1 arg but found {len(args)} args'

        for k, v in kwargs.items():
            self[k] = v

    def _validate_key(self, key):
        if self._keys is not None and key not in self._keys:
            raise KeyError(f'{key} is not a valid key. Valid: {self._keys}')

    def __getattr__(self, key):
        if (key != '__dict__' and key not in self.__slots__
                and key not in dir(self)
                and (self._keys is None or key in self._keys)):
            return self[key]
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key in self.__slots__ or key in dir(self):
            super().__setattr__(key, value)
        elif self._keys is None or key in self._keys:
            self[key] = value
        else:
            raise AttributeError(f'{key} is not a valid key or existing '
                                 f'attribute of {self.__class__.__name__}')

    def __getitem__(self, key):
        self._validate_key(key)
        return self._data[key]

    def __setitem__(self, key, value):
        self._validate_key(key)
        self._data[key] = value

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__ + pformat(tuple(self.items()))


class Result:
    """"""

    def __init__(self, splits=None, detector_attributes=None):
        if splits is None:
            splits = ['train', 'test', 'explainer']
        self._splits = splits

        if detector_attributes is None:
            detector_attributes = ['threshold']
        self._detector_attributes = detector_attributes

        # meta: ['explainer', 'innocuous_model', 'n_explainer_samples', 'id']
        self._meta = ResultCollection()
        self._hparams = ResultCollection()
        self._scores = ResultCollection(self._splits)
        self._predictions = ResultCollection(self._splits)
        self._detector = ResultCollection(self._detector_attributes)
        self._metrics = ResultCollection()

    @classmethod
    def from_dict(cls, result, **kwargs):
        obj = cls(**kwargs)

        for k, v in result.items():
            for group, attr in (('hparam', 'hparams'),
                                ('scores', 'scores'),
                                ('pred_y', 'pred_y'),
                                ('detector', 'detector'),
                                ('metric', 'metrics')):
                prefix = f'{group}_'
                if k.startswith(prefix):
                    k_trunc = k[len(prefix):]
                    if attr == 'pred_y' and k_trunc == 'samples':
                        # legacy data
                        k_trunc = 'explainer'
                    setattr(getattr(obj, attr), k_trunc, v)
                    break
            else:
                setattr(getattr(obj, 'meta'), k, v)

        return obj

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        if not isinstance(value, dict):
            raise TypeError(f'Expected dict but received {type(value)}')
        self._meta.set(value)

    @property
    def hparams(self):
        return self._hparams

    @hparams.setter
    def hparams(self, value):
        if not isinstance(value, dict):
            raise TypeError(f'Expected dict but received {type(value)}')
        self._hparams.set(value)

    @property
    def scores(self):
        """OOD Scores (lower is more OOD)"""
        return self._scores

    @scores.setter
    def scores(self, value):
        if not isinstance(value, dict):
            raise TypeError(f'Expected dict but received {type(value)}')
        self._scores.set(value)

    @property
    def pred_y(self):
        return self._predictions

    @pred_y.setter
    def pred_y(self, value):
        if not isinstance(value, dict):
            raise TypeError(f'Expected dict but received {type(value)}')
        self._predictions.set(value)

    @property
    def detector(self):
        return self._detector

    @detector.setter
    def detector(self, value):
        if not isinstance(value, dict):
            raise TypeError(f'Expected dict but received {type(value)}')
        self._detector.set(value)

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        if not isinstance(value, dict):
            raise TypeError(f'Expected dict but received {type(value)}')
        self._metrics.set(value)

    def to_dict(self):
        return {
            **{k: v for k, v in self.meta.items()},
            **{f'hparam_{k}': v for k, v in self.hparams.items()},
            **{f'scores_{k}': v for k, v in self.scores.items()},
            **{f'pred_y_{k}': v for k, v in self.pred_y.items()},
            **{f'detector_{k}': v for k, v in self.detector.items()},
            **{f'metric_{k}': v for k, v in self.metrics.items()},
        }

    def __repr__(self):
        return str(self)

    def __str__(self):
        return pformat(self.to_dict())


def compute_metrics(result: Union[Result, Dict]):
    if not isinstance(result, Result):
        result = Result.from_dict(result)

    results = {}
    for name_a, name_b in [
        ('explainer', 'test'),
        ('explainer', 'train'),
        ('test', 'train'),
    ]:
        scores_a = result.scores[name_a]
        scores_b = result.scores[name_b]

        for robust in [True, False]:
            for normalized in [True, False]:
                cdf_data = None
                for i, metric in enumerate([
                    metrics.cdf_delta,
                    metrics.pct_cdf_greater,
                    metrics.cdf_area_above,
                    metrics.cdf_delta_ratio,
                ]):
                    end = '_robust' if robust else ''
                    end += '_normalized' if normalized else ''

                    metric_key = f'{metric.__name__}_{name_a}_{name_b}{end}'
                    if i == 0:
                        metric_value, area_a, area_b, cdf_data = metric(
                            scores_a, scores_b, robust=robust,
                            normalized=normalized, return_cdf_data=True,
                            ret_areas=True)
                        results[metric_key] = metric_value
                        for area_name, area in [(name_a, area_a),
                                                (name_b, area_b)]:
                            metric_key_area = f'cdf_area_{area_name}{end}'
                            results[metric_key_area] = area
                    else:
                        metric_value = metric(
                            None, None, robust=robust, normalized=normalized,
                            cdf_data=cdf_data)
                        results[metric_key] = metric_value

    for metric, secondary_group, secondary_key in [
        (metrics.peak_deltas, result.pred_y, '{a}'),
        (metrics.under_threshold, result.detector, 'threshold'),
    ]:
        for name_a in ['explainer', 'test', 'train']:
            scores_a = result.scores[name_a]
            secondary_a = secondary_group[secondary_key.format(a=name_a)]

            metric_key = f'{metric.__name__}_{name_a}'
            metric_value = metric(scores_a, secondary_a)
            results[metric_key] = metric_value

    return results


def as_py_type(value):
    if value is None:
        return None
    elif isinstance(value, (str, int, float, list, tuple)):
        return value
    elif isinstance(value, (np.number, np.ndarray, np.bool_, np.character)):
        return value.tolist()
    else:
        raise TypeError(f'Unsupported type "{type(value)}" for value "{value}"')


def hash_hparams(hparams):
    # sort pairs by key ascending as tuple of tuples (deterministic)
    hparams = tuple((k, as_py_type(v))
                    for k, v in sorted(hparams.items(), key=lambda x: x[0]))
    return hashlib.sha256(str(hparams).encode()).hexdigest()


def evaluate_detector(
        C,
        P: ProcessedProblem,
        explainer_data,
        hparams,
        n_explainer_samples=None,
        detectors=None,
):
    params_hash = hash_hparams(hparams)

    if n_explainer_samples is None:
        n_explainer_samples = len(P.X_train) * 100

    results = []
    detectors_provided = detectors is not None
    if not detectors_provided:
        detectors = {}
    for explainer_name, explainer_innoc_map in explainer_data.items():
        if not detectors_provided:
            detectors[explainer_name] = {}
        for innocuous_model_name, saved_data in explainer_innoc_map.items():
            print('---', explainer_name, innocuous_model_name, '---')

            result = Result()
            result.meta = {
                'explainer': explainer_name,
                'innocuous_model': innocuous_model_name or 'NA',
                'n_explainer_samples': n_explainer_samples,
                'id': params_hash,
            }
            result.hparams = hparams

            model_predict_func = saved_data['model_predict_func']
            X_samples = saved_data['X_samples']

            y_train_pred = model_predict_func(P.X_train)
            if innocuous_model_name is not None:
                y_train_pred, ood_train_pred = y_train_pred

            ood_test_pred = saved_data['dood']
            ood_samples_pred = saved_data['dood_g']
            y_test_pred = saved_data['y_test_pred']
            y_samples_pred = saved_data['y_samples_pred']

            if n_explainer_samples < len(X_samples):
                print(f'Down-sampling X_samples to {n_explainer_samples} '
                      f'({n_explainer_samples / len(X_samples) * 100:.2f}% of '
                      f'{len(X_samples)} samples)')
                rs = np.random.RandomState()
                sample_idxs = rs.choice(np.arange(len(X_samples)),
                                        size=n_explainer_samples,
                                        replace=False)
                X_samples = X_samples[sample_idxs]
                y_samples_pred = y_samples_pred[sample_idxs]
                ood_samples_pred = ood_samples_pred[sample_idxs]
            else:
                print(f'Using all {len(X_samples)} samples of X_samples')

            if detectors_provided:
                detector = detectors[explainer_name][innocuous_model_name]
            else:
                detector = get_detector(C.detector_name, hparams,
                                        problem=P.problem)

                with timer('Detector fit'):
                    try:
                        detector.fit(P.X_train, y_train_pred)
                    except ValueError as e:
                        if (len(e.args) and isinstance(e.args[0], str) and
                                re.search(r'but found invalid values',
                                          e.args[0])):
                            print('Invalid values encountered, invalidating '
                                  'result.')
                            results.append(result)
                            detectors[explainer_name][
                                innocuous_model_name] = None
                            continue
                        else:
                            raise e

            with timer('Detector predict OOD'):
                scores_explainer = detector.score_samples(
                    X_samples, y_samples_pred)

            scores_train = detector.score_samples(P.X_train, y_train_pred)
            scores_test = detector.score_samples(P.X_test, y_test_pred)

            def handle_nans(scores, pred_y):
                mask = ~np.isnan(scores)
                return scores[mask], pred_y[mask]

            scores_train, y_train_pred = handle_nans(scores_train, y_train_pred)
            scores_test, y_test_pred = handle_nans(scores_test, y_test_pred)
            scores_explainer, y_samples_pred = handle_nans(
                scores_explainer, y_samples_pred)

            # fidelity
            fidelity_h_soft = 1 - (
                    np.mean((ood_test_pred - scores_test) ** 2) +
                    np.mean((ood_samples_pred - scores_explainer) ** 2)
            ) / 2
            threshold = .5
            fidelity_h_hard = (
                                      np.mean(ood_test_pred ==
                                              (scores_test >= threshold).astype(
                                                  int)) +
                                      np.mean(ood_samples_pred ==
                                              (
                                                      scores_explainer >= threshold).astype(
                                                  int))
                              ) / 2
            print(f'fidelity_h (soft) = {fidelity_h_soft:.5}')
            print(f'fidelity_h (hard) = {fidelity_h_hard:.5}')

            result.scores = {
                'train': scores_train,
                'test': scores_test,
                'explainer': scores_explainer,
            }
            result.pred_y = {
                'train': y_train_pred,
                'test': y_test_pred,
                'explainer': y_samples_pred,
            }
            # noinspection PyDunderSlots,PyUnresolvedReferences
            result.detector.threshold = detector.threshold_

            result.metrics = compute_metrics(result)

            results.append(result)
            detectors[explainer_name][innocuous_model_name] = detector
            print()
    return results, detectors
