from numpy.core.numeric import cross
import Utils
import time
import random
import math
from sklearn.metrics import accuracy_score

def hillClimbing(X, y, predictors, classes_frequency):
  best_ensemble = []
  max_accuracy = 0

  start_time = time.time()
  end_time = start_time
  max_time = 120

  improved_accuracy = True

  while improved_accuracy and end_time-start_time <= max_time:
    improved_accuracy = False
    new_best_ensemble = best_ensemble
    best_neighbor = -1

    for i in range(len(predictors)):
      current_ensemble = best_ensemble + [predictors[i]]
      y_pred = Utils.predict(X, current_ensemble, classes_frequency)
      current_accuracy = accuracy_score(y_true = y, y_pred = y_pred)
      if current_accuracy >= max_accuracy:
        max_accuracy = current_accuracy
        new_best_ensemble = current_ensemble
        improved_accuracy = True
        best_neighbor = i

    if improved_accuracy:
      best_ensemble = new_best_ensemble
      predictors = predictors[:best_neighbor] + predictors[best_neighbor+1:]

    end_time = time.time()

  return best_ensemble

def simulatedAnneling(X, y, predictors, classes_frequency):
  best_ensemble = Utils.generateRandomState(predictors = predictors)
  max_accuracy = accuracy_score(y_true = y, y_pred = Utils.predict(X, best_ensemble, classes_frequency))

  state = best_ensemble

  start_time = time.time()
  end_time = start_time
  max_time = 120

  temperature = 200
  alfa = 0.1

  max_iter = 10

  while temperature >= 1 and end_time-start_time <= max_time:
    for _ in range(max_iter):
      state_accuracy = accuracy_score(y_true = y, y_pred = Utils.predict(X, state, classes_frequency))

      neighbors = Utils.generateNeighbors(state = state, predictors = predictors)
      random_neighbor = Utils.getRandomNeighbor(neighbors)
      random_neighbor_accuracy = accuracy_score(y_true = y, y_pred = Utils.predict(X, random_neighbor, classes_frequency))

      if random_neighbor_accuracy > state_accuracy:
        state = random_neighbor
        if random_neighbor_accuracy > max_accuracy:
          max_accuracy = random_neighbor_accuracy
          best_ensemble = random_neighbor
      else:
        probabiity = 1/(math.exp(1)**((state_accuracy - random_neighbor_accuracy)/temperature))
        value = random.uniform(0,1)
        if value <= probabiity:
          state = random_neighbor

    end_time = time.time()
    temperature = temperature * alfa

  return best_ensemble

def genetic(X, y, predictors, classes_frequency):
  best_ensemble = []
  max_accuracy = 0

  crossover_ratio = 0.9
  mutation_ratio = 0.1
  elite_percentage = 0.2
  population_size = 10
  max_generation = 20
  generation = 0

  start_time = time.time()
  end_time = start_time
  max_time = 120

  population = Utils.generateInitialPopulation(population_size = population_size, 
                                               predictors = predictors)

  while generation < max_generation and end_time-start_time <= max_time:
    #[(accuracy, individual)]
    population_accuracy = Utils.evaluate_population(population = population, X = X, y = y, classes_frequency = classes_frequency)
    
    #[individual]
    new_population = Utils.elitism(population_accuracy = population_accuracy, elite_percentage = elite_percentage)

    best_individual = new_population[0]
    best_individual_accuracy = accuracy_score(y_true = y, y_pred = Utils.predict(X, best_individual, classes_frequency))
    
    if best_individual_accuracy > max_accuracy:
      best_ensemble = best_individual
      max_accuracy = best_individual_accuracy

    #[individual]
    selected = Utils.selection(population_accuracy = population_accuracy, selection_size = population_size - len(new_population))
    
    #[individual]
    crossed = Utils.crossover_step(population =  selected, crossover_ratio = crossover_ratio)
    
    mutated = Utils.mutated_step(population = crossed, mutation_ratio = mutation_ratio)

    population = new_population + mutated

    end_time = time.time()
    generation = generation + 1
  
  return best_ensemble