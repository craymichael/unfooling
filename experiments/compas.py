from typing import List

import numpy as np

from experiments.lib import Problem
from fooling.models import SimpleTabularModel
from fooling.models import PrejudicedClassificationModel
from fooling.models import InnocuousClassificationModel
from fooling import get_data


class COMPASExperiment(Problem):
    def load_data(self):
        # communities and crime dataset
        X, y, _ = get_data.get_and_preprocess_compas_data(self._params)

        # add unrelated columns, setup
        rs = np.random.RandomState()
        X['unrelated_column_one'] = rs.choice([0, 1], size=X.shape[0])
        X['unrelated_column_two'] = rs.choice([0, 1], size=X.shape[0])
        features = [*X.keys()]
        categorical_features = [
            'unrelated_column_one',
            'unrelated_column_two',
            'c_charge_degree_F',
            'c_charge_degree_M',
            'two_year_recid',
            'race',
            'sex_Male',
            'sex_Female',
        ]
        return X.values, y, features, categorical_features

    @property
    def biased_features(self):
        return ['race']

    @property
    def prejudiced_model(self) -> SimpleTabularModel:
        race_idx = self._features.index('race')

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
            thresholds=[0, 0],
            name='2Unrelated',
        )
        return [innocuous_model_1, innocuous_model_2]

    @property
    def sensitive_features(self):
        return list({*self._features} &
                    {'sex',
                     'sex_Male',
                     'sex_Female',
                     'race'})
