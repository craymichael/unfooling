"""
communities_and_crime.py - A Unfooling-LIME-SHAP file
Copyright (C) 2022  Zach Carmichael
"""
from typing import List

import numpy as np

from experiments.lib import Problem
from fooling.models import SimpleTabularModel
from fooling.models import PrejudicedClassificationModel
from fooling.models import InnocuousClassificationModel
from fooling import get_data


class CommunitiesAndCrimeExperiment(Problem):
    def load_data(self):
        # communities and crime dataset
        X, y, _ = get_data.get_and_preprocess_cc(self._params)

        # add unrelated columns, setup
        rs = np.random.RandomState()
        X['unrelated_column_one'] = rs.choice([0, 1], size=X.shape[0])
        X['unrelated_column_two'] = rs.choice([0, 1], size=X.shape[0])
        features = [*X.keys()]
        categorical_features = ['unrelated_column_one', 'unrelated_column_two']
        return X.values, y, features, categorical_features

    @property
    def biased_features(self):
        return ['racePctWhite numeric']

    @property
    def prejudiced_model(self) -> SimpleTabularModel:
        race_idx = self._features.index('racePctWhite numeric')

        # We discriminate based on race for f
        prejudiced_model = PrejudicedClassificationModel(
            negative_outcome=self._params.negative_outcome,
            positive_outcome=self._params.positive_outcome,
            racist_idxs=race_idx,
            thresholds=0,
            name='racist',
        )
        return prejudiced_model

    @property
    def innocuous_models(self) -> List[SimpleTabularModel]:
        # consider two RANDOMLY DRAWN features to display in psi
        unrelated_idx1 = self._features.index('unrelated_column_one')
        unrelated_idx2 = self._features.index('unrelated_column_two')

        innocuous_model_1 = InnocuousClassificationModel(
            negative_outcome=self._params.negative_outcome,
            positive_outcome=self._params.positive_outcome,
            unrelated_idxs=unrelated_idx1,
            thresholds=0,
            name='1Unrelated',
        )
        innocuous_model_2 = InnocuousClassificationModel(
            negative_outcome=self._params.negative_outcome,
            positive_outcome=self._params.positive_outcome,
            unrelated_idxs=[unrelated_idx1, unrelated_idx2],
            thresholds=[+.5, -.5],
            name='2Unrelated',
        )
        return [innocuous_model_1, innocuous_model_2]

    @property
    def sensitive_features(self):
        return list({*self._features} &
                    {'racepctblack numeric',
                     'racePctWhite numeric',
                     'racePctAsian numeric',
                     'racePctHisp numeric',
                     'agePct12t21 numeric',
                     'agePct12t29 numeric',
                     'agePct16t24 numeric',
                     'agePct65up numeric',
                     'whitePerCap numeric',
                     'blackPerCap numeric',
                     'indianPerCap numeric',
                     'AsianPerCap numeric',
                     'OtherPerCap numeric',
                     'HispPerCap numeric',
                     'MalePctDivorce numeric',
                     'MalePctNevMarr numeric',
                     'FemalePctDiv numeric',
                     'NumImmig numeric',
                     'PctImmigRecent numeric',
                     'PctImmigRec5 numeric',
                     'PctImmigRec8 numeric',
                     'PctImmigRec10 numeric',
                     'PctRecentImmig numeric',
                     'PctRecImmig5 numeric',
                     'PctRecImmig8 numeric',
                     'PctRecImmig10 numeric',
                     'PctForeignBorn numeric',
                     'PctBornSameState numeric',
                     'RacialMatchCommPol numeric',
                     'PctPolicWhite numeric',
                     'PctPolicBlack numeric',
                     'PctPolicHisp numeric',
                     'PctPolicAsian numeric',
                     'PctPolicMinor numeric'})
