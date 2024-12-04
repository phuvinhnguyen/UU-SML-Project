import xgboost as xgb
import numpy as np
from .abstract import AbstractClass

class XGBoost_regresion(AbstractClass):
    def __init__(self,
                 objective='reg:squarederror',
                 eval_metric='rmse',
                 n_estimators=100,
                 max_depth=5,
                 learning_rate=0.1
                 ):
        self.xgb = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
            )

    def predict(self, input):
        return self.xgb.predict(input)
    
    def fit(self, dataset):
        self.xgb.fit(
            np.array([i[0] for i in dataset]),
            np.array([i[1] for i in dataset])
            )