import xgboost as xgb
import numpy as np

class XGBoost_regresion:
    def __init__(self,
                 objective='reg:squarederror',
                 eval_metric='rmse',
                 n_estimators=100,
                 max_depth=5,
                 learning_rate=0.1
                 ):
        self.xgb = xgb.XGBRegressor(
            objective=objective,
            eval_metric=eval_metric,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
            )

    def predict(self, input):
        return self.xgb.predict(input)
    
    def eval(self, dataset):
        input = np.array([i[0] for i in dataset])
        output = np.array([i[1] for i in dataset])
        return self.xgb.score(input, output)
    
    def fit(self, dataset):
        self.xgb.fit(
            np.array([i[0] for i in dataset]),
            np.array([i[1] for i in dataset])
            )