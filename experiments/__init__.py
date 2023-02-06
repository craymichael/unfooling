"""
__init__.py - A Unfooling-LIME-SHAP file
Copyright (C) 2022  Zach Carmichael
"""
from pathlib import Path

from fooling.utils import Params

from experiments.continuous_test import ContinuousCommunitiesAndCrimeExperiment
from experiments.compas import COMPASExperiment
from experiments.communities_and_crime import CommunitiesAndCrimeExperiment
from experiments.german import GermanExperiment

DEFAULT_PARAMS = (Path(__file__).parent.parent / 'model_configurations' /
                  'experiment_params.json')


def get_experiment(name, params=DEFAULT_PARAMS):
    if isinstance(params, Path):
        params = str(params)
    if isinstance(params, str):
        params = Params(params)
    return {
        'CC': CommunitiesAndCrimeExperiment,
        'CCC': ContinuousCommunitiesAndCrimeExperiment,
        'COMPAS': COMPASExperiment,
        'German': GermanExperiment,
    }[name](params)
