from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN_classification:
    def __init__(self, n_neighbors=3, weights='uniform', algorithm='auto'):
        """
        Initializes the KNN classifier with given hyperparameters.
        
        Parameters:
        - n_neighbors: Number of neighbors to use.
        - weights: Weight function used in prediction ('uniform', 'distance', or callable).
        - algorithm: Algorithm used to compute the nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
        """
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
    
    def predict(self, input):
        """
        Predicts the class labels for the given input data.
        
        Parameters:
        - input: Array-like of shape (n_samples, n_features).
        
        Returns:
        - Predicted class labels.
        """
        return self.knn.predict(input)
    
    def eval(self, dataset):
        """
        Evaluates the model using accuracy on the given dataset.
        
        Parameters:
        - dataset: List of tuples [(input1, output1), (input2, output2), ...].
        
        Returns:
        - Accuracy score of the model.
        """
        input = np.array([i[0] for i in dataset])
        output = np.array([i[1] for i in dataset])
        return self.knn.score(input, output)
    
    def fit(self, dataset):
        """
        Trains the model on the given dataset.
        
        Parameters:
        - dataset: List of tuples [(input1, output1), (input2, output2), ...].
        """
        self.knn.fit(
            np.array([i[0] for i in dataset]),
            np.array([i[1] for i in dataset])
        )
