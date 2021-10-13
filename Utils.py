import numpy as np
import random
import math
from sklearn.metrics import accuracy_score

def predict(X, ensemble, classes_frequency):
  results = np.array([classifier.predict(X) for classifier in ensemble]).transpose()
  y_pred = []
  for result in results:
    classes_predicted, classes_predicted_frequency = np.unique(result, return_counts=True)
    most_voted_frequency = np.amax(classes_predicted_frequency)
    most_voted = [c for (c, f) in zip(classes_predicted, classes_predicted_frequency) if f == most_voted_frequency]
    most_voted.sort(key=lambda x:-classes_frequency[x])
    y_pred.append(most_voted[0])

  return np.array(y_pred)

def generateRandomState(predictors):
  return random.sample(predictors, random.randint(1, len(predictors)))

def generateNeighbors(state, predictors):
  neighbors = []
  for i in range(len(state)):
    neighbor = state[:i] + state[i+1:]
    if neighbor != []:
      neighbors.append(neighbor)

  for predictor in predictors:
    neighbor = state + [predictor]
    neighbors.append(neighbor)
    
  return neighbors

def getRandomNeighbor(neighbors):
  return neighbors[random.randint(0,len(neighbors)-1)]

def generateInitialPopulation(population_size, predictors):
  population = []
  for _ in range(population_size):
    individual = generateRandomState(predictors = predictors)
    population.append(individual)
  return population

def evaluate_population(population, X, y, classes_frequency):
  population_accuracy = []
  for individual in population:
    individual_predictions = predict(X = X, ensemble = individual, classes_frequency = classes_frequency)
    individual_accuracy = accuracy_score(y_true = y, y_pred = individual_predictions)
    population_accuracy.append((individual_accuracy, individual))
  return population_accuracy

def elitism(population_accuracy, elite_percentage):
  individuals_number = int(max(1, math.floor(elite_percentage * len(population_accuracy))))
  accuracy_elite = sorted(population_accuracy, key = lambda x:-x[0])[:individuals_number]
  elite = [state for _, state in accuracy_elite]
  return elite

def buildRoulette(population_accuracy):
  total_accuracy = 0
  for accuracy, _ in population_accuracy:
    total_accuracy += accuracy

  roulette = []
  acc_probability = 0
  for accuracy, individual in population_accuracy:
    probability = accuracy / total_accuracy
    acc_probability += probability
    roulette.append((acc_probability, individual))

  return roulette

def runRoulette(roulette, rounds):
  selected = []
  
  while len(selected) < rounds:
    r = random.uniform(0, 1)
    for acc_probability, individual in roulette:
      if r <= acc_probability:
        selected.append(individual)
        break

  return selected

def selection(population_accuracy, selection_size):
  roulette = buildRoulette(population_accuracy)
  selected_population = runRoulette(roulette, selection_size)
  return selected_population

def crossover(parent1, parent2):
  n = random.randint(0, len(parent1) - 1)
  m = random.randint(0, len(parent2) - 1)
  son = parent1[:n] + parent2[m:]
  daugther = parent1[n:] + parent2[:m]

  return son, daugther

def crossover_step(population, crossover_ratio):
  new_population = []

  for _ in range (round(len(population)/2)):
    fst_ind = random.randint(0, len(population) - 1)
    scd_ind = random.randint(0, len(population) - 1)
    parent1 = population[fst_ind] 
    parent2 = population[scd_ind]
    r = random.uniform(0, 1)

    if r <= crossover_ratio:
      son, daughter = crossover(parent1, parent2)
    else:
      son, daughter = parent1, parent2

    new_population.append(son)
    new_population.append(daughter)

  return new_population

def mutation(indiv):
  individual = indiv.copy()
  if len(individual) > 1:
    individual = individual[1:]
  return individual

def mutated_step(population, mutation_ratio):
  ind = 0
  for individual in population:
    r = random.uniform(0, 1)
    if r <= mutation_ratio:
      population[ind] = mutation(individual)
    ind += 1
  return population