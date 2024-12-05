from methods.RandomForest import RandomForest
from data import BikeDemandDataset_v01 as BikeDemandDataset

if __name__ == '__main__':
    dataset = BikeDemandDataset('training_data_fall2024.csv')
    eval_dataset = BikeDemandDataset('training_data_fall2024.csv', data_type='validation')
    xgb = RandomForest()
    xgb.fit(dataset)

    print('Score on training data:')
    xgb.eval(dataset)
    print('Score on validation data:')
    xgb.eval(eval_dataset)