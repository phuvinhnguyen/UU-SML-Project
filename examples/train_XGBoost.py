from data import BikeDemandDataset
from methods.XGBoost import XGBoost_regresion

if __name__ == '__main__':
    dataset = BikeDemandDataset('training_data_fall2024.csv')
    xgb = XGBoost_regresion()
    xgb.fit(dataset)
    print(xgb.eval(dataset))