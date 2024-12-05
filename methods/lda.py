from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from .abstract import AbstractClass


class LDA(AbstractClass):
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()

    def predict(self, input):
        return self.model.predict(input)
    
    def fit(self, dataset):
        self.model.fit(
            np.array([i[0] for i in dataset]),
            np.array([i[1] for i in dataset])
            )