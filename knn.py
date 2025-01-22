import numpy as np
from collections import Counter

class KNearestNeighbours():
    def __init__(self, k=5, mode='classification', distance='euclidean'):
        self.k =k
        self.mode = mode
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train =y

    def _compute_distance(self, x1, x2):
        if self.distance == 'euclidean':
            return np.sqrt(np.sum((x1-x2)**2))
        else:
            raise ValueError(f'Distance {self.distance} not implemented')
        
    def _get_neighbours(self, x):
        distances = []

        for i, x_train in enumerate(self.X_train):
            dist = self._compute_distance(x_train, x)
            distances.append([dist, self.y_train[i]])

        distances.sort(key=lambda tup:tup[0])
        neighbours = distances[:self.k]
        return neighbours
    
    def predict(self, X):
        predictions = []
        for x in X:
            neighbours = self._get_neighbours(x)
            values = [n[1] for n in neighbours]
            
            if self.mode == 'classification':
                most_common_labels = Counter(values)
                predictions.append(most_common_labels.most_common(1)[0][0])

            elif self.mode == 'regression':
                predictions.append(np.array(values).mean())

            else:
                raise ValueError(f'Mode {self.mode} not implemented')
            
        return np.array(predictions)
    
# Example usage:
if __name__ == "__main__":
    # For illustration, let's create a small synthetic dataset.
    np.random.seed(42)
    X_train = np.random.rand(10, 2)
    y_train_class = np.array([0, 1]*5)
    y_train_reg = np.random.rand(10)

    X_test = np.random.rand(3, 2)

    # Classification Example
    knn_clf = KNearestNeighbours(k=3, mode='classification')
    knn_clf.fit(X_train, y_train_class)
    preds_class = knn_clf.predict(X_test)
    print("Classification Predictions:", preds_class)

    # Regression Example
    knn_reg = KNearestNeighbours(k=3, mode='regression')
    knn_reg.fit(X_train, y_train_reg)
    preds_reg = knn_reg.predict(X_test)
    print("Regression Predictions:", preds_reg)