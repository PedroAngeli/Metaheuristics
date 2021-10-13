import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from HeterogeneousPoolingClassifier import *
from Metaheuristics import *
import time

class HeterogeneousPooling:
  def __init__(self, metaheuristic = None): 

    self.metametaheuristic = None

    if metaheuristic == 'hill climbing':
      self.metametaheuristic = hillClimbing
    elif metaheuristic == 'simulated anneling':
      self.metametaheuristic = simulatedAnneling
    elif metaheuristic == 'genetic':
      self.metametaheuristic = genetic

    self.heterogeneousPooling = HeterogeneousPoolingClassifier(metaheuristic = self.metametaheuristic) #modelo
    self.scalar = StandardScaler() #z-score
    self.grid = {'estimator__n_samples': [3, 5, 7]} #hiperparametros do grid search
    self.pipeline = Pipeline([('transformer', self.scalar), ('estimator', self.heterogeneousPooling)]) #pipeline de transformações
    self.gs = GridSearchCV(estimator=self.pipeline, param_grid = self.grid, scoring='accuracy', cv = 4, n_jobs=-1) # definindo grid search
    self.rfk = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=36851234) #Estratégia de validação cruzada
  
  def fit(self, X, y): #Treina o modelo
    print('Treinando Heterogeneous Pooling...')
    start_time = time.time()
    self.scores = cross_val_score(self.gs, X, y, scoring='accuracy', cv = self.rfk)
    end_time = time.time()
    print('Finalizado!')
    time_minutes = (end_time-start_time)/60
    print('Tempo de treinamento {:.2f} minutos'.format(time_minutes))
  
  def results(self): #Retorna a média, desvio padrão e intervalo de confiança
    if hasattr(self, 'scores') == False:
      print('You have to fit the model first.')
      return
    mean = self.scores.mean()
    std = self.scores.std()
    inf, sup = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(len(self.scores)))
    return (mean, std, inf, sup)

  def getScores(self):
    if hasattr(self, 'scores') == False:
      print('You have to fit the model first.')
      return
    return self.scores