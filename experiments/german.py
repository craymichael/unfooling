from typing import List

import numpy as np

from experiments.lib import Problem
from fooling.models import SimpleTabularModel
from fooling.models import PrejudicedClassificationModel
from fooling.models import InnocuousClassificationModel
from fooling import get_data


class GermanExperiment(Problem):
    def load_data(self):
        X, y, _ = get_data.get_and_preprocess_german(self._params)

        features = [*X.keys()]
        categorical_features = [
            'Gender', 'ForeignWorker', 'Single', 'HasTelephone',
            'CheckingAccountBalance_geq_0', 'CheckingAccountBalance_geq_200',
            'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
            'MissedPayments', 'NoCurrentLoan',
            'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank',
            'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor',
            'OwnsHouse', 'RentsHouse', 'Unemployed',
            'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4',
            'JobClassIsSkilled',
        ]
        return X.values, y, features, categorical_features

    @property
    def biased_features(self):
        return ['Gender']

    @property
    def prejudiced_model(self) -> SimpleTabularModel:
        gender_idx = self._features.index('Gender')

        # We discriminate based on race for f
        prejudiced_model = PrejudicedClassificationModel(
            negative_outcome=self._params.negative_outcome,
            positive_outcome=self._params.positive_outcome,
            racist_idxs=gender_idx,
            thresholds=0,
            name='racist',
        )
        return prejudiced_model

    @property
    def innocuous_models(self) -> List[SimpleTabularModel]:
        loan_rate_idx = self._features.index('LoanRateAsPercentOfIncome')
        mean_lrpi = np.mean(self.X[:, loan_rate_idx])

        innocuous_model_1 = InnocuousClassificationModel(
            negative_outcome=self._params.negative_outcome,
            positive_outcome=self._params.positive_outcome,
            unrelated_idxs=loan_rate_idx,
            thresholds=mean_lrpi,
            name='1Unrelated',
        )
        return [innocuous_model_1]

    @property
    def sensitive_features(self):
        return list({*self._features} &
                    {'Gender'})
