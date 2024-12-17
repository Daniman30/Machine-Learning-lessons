import numpy as np

class DistanceBasedClassifier:
    def __init__(self):
        self.centroids = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for clase in classes:
            self.centroids[clase] = X[y == clase].mean(axis=0)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = {clase: np.linalg.norm(x - centroid) for clase, centroid in self.centroids.items()}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)