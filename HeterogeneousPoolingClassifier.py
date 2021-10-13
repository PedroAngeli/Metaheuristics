import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from sklearn.utils import resample
import Utils
import time

class HeterogeneousPoolingClassifier(BaseEstimator):
  def __init__(self, n_samples=1, metaheuristic = None):
    super().__init__()
    self.n_samples = n_samples
    self.predictors = []
    self.metaheuristic = metaheuristic

  def fit(self, X, y):
    classes, classes_frequency = np.unique(y, return_counts=True)
    self.classes_frequency = [0] * (len(classes))
    for i in range(len(classes)):
      self.classes_frequency[classes[i]] = classes_frequency[i]

    for i in range(self.n_samples):
      X_test, y_test = resample(X, y, random_state=i)
      self.predictors.append(DecisionTreeClassifier().fit(X_test, y_test))
      self.predictors.append(GaussianNB().fit(X_test, y_test))
      self.predictors.append(KNeighborsClassifier().fit(X_test, y_test))

    if self.metaheuristic != None:
      self.predictors = self.metaheuristic(X, y, self.predictors, self.classes_frequency)

  def predict(self, X):
    return Utils.predict(X = X,ensemble = self.predictors,classes_frequency = self.classes_frequency)
