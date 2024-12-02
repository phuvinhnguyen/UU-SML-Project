from data import BikeDemandDataset_v01 as BikeDemandDataset
from methods.knn import KNN_classification

if __name__ == '__main__':
    dataset = BikeDemandDataset('training_data_fall2024.csv')
    eval_dataset = BikeDemandDataset('training_data_fall2024.csv', data_type='validation')
    model = KNN_classification(n_neighbors=3)
    model.fit(dataset)

    print('Score on training data:', model.eval(dataset))
    print('Score on validation data:', model.eval(eval_dataset))