import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels





    def fit(self, xtr, ytr):
        classes = np.unique(ytr)  # todas las clases existentes
        centroids = []
        for c in classes:
            centroids.append(np.mean(xtr[ytr == c], axis=0))  # promedio de cada clase
        self._centroids = np.array(centroids)
        self._class_labels = classes

    def predict(self, xts):
        y_pred = []
        for sample in xts:
            distances = np.linalg.norm(self._centroids - sample, axis=1)  # distancia a cada centroide
            closest = np.argmin(distances)  # índice del centroide más cercano
            y_pred.append(self._class_labels[closest])
        return np.array(y_pred)