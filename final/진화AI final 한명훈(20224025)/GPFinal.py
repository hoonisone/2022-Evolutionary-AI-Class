import numpy as np
from numpy import dot
from numpy.linalg import norm
from GP import *
from GA import *
import sys
import random
import copy
import math

class  Circumstance:
    def __init__(self) -> None:
        with open("datapoint3d.txt", "r") as f:
            f.readline()
            lines = f.readlines()

        self.input_list = []
        self.output_list = []
        for line in lines:
            line = line.strip()
            line = line.split(',')
            self.input_list.append([float(line[0]), float(line[0])])
            self.output_list.append(float(line[2]))

    def mean_squared_error(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return ((x-y)**2).mean(axis=None)

    def cos_similarity(self, x, y):
        n = norm(x) * norm(y)
        if n == 0:
            return -1
        return dot(x, y)/n

    def get_predict_list(self, function):
        return [function(input) for input in self.input_list]

    def evaluate(self, function):
        predict_list = []
        for (x, y) in self.input_list: 
            predict_list.append(function(x, y))
        return self.mean_squared_error(np.array(self.output_list), np.array(predict_list))

    def accuracy(self, function):
        return self.cos_similarity(self.get_predict_list(function), self.output_list)

    def loss(self, function):
        return self.mean_squared_error(self.get_predict_list(function), self.output_list)


class GPFinal(GenericProgramming):
    def __init__(self, population_size, input_num, cross_over_p, mutation_p):
        super().__init__(population_size, input_num, cross_over_p, mutation_p)

    def loss(self, individual) -> float:
        return Circumstance().loss(lambda input : individual.calculate(input))

    def get_loss_list(self) -> list:
        return [self.loss(individual) for individual in self.population]

    def fitness(self, individual) -> float:
        fitness = self.loss(individual)
        fitness = math.log(fitness)
        return -min(1000000, fitness**3)

    def get_fitness_list(self) -> list:
        # 각 individual의 fitness 리스트 반환
        return [self.fitness(individual) for individual in self.population]

    def selection(self):
        fitness_list = self.get_fitness_list()
        winner = []
        idx_list = random.shuffle([i for i in range(len(fitness_list))])
        for i in range(len(idx_list)//2):
            idx_list


        min_fitness = min(fitness_list)
        fitness_list = [fitness - min_fitness for fitness in fitness_list]

        fitness_list = np.array(fitness_list)
        
        fitness_list = list(fitness_list)
        
        # selection 결과 idx list
        index_list = Selection.roulette_selection(fitness_list)
        
        # idx에 맞는 individual들을 복사하여 new population생성 및 반환
        return [self.population[idx].copy() for idx in index_list]

    def print_state(self) -> None:
        fitness_list = self.get_fitness_list()

        min_fitness = min(fitness_list)
        max_fitness = max(fitness_list)
        mean_fitness = sum(fitness_list)/len(fitness_list)

        loss_list = self.get_loss_list()

        min_loss = min(loss_list)
        max_loss = max(loss_list)
        mean_loss = sum(loss_list)/len(loss_list)
        print("%3dGen:"%self.generation, end=" ")
        print("[Fitness] Min: %11.3f, Max:%10.3f, Avg: %10.3f"%(min_fitness, max_fitness, mean_fitness), end="  ")
        print(   "[Loss] Min: %11.3f, Max:%10.3f, Avg: %10.3f"%(min_loss, max_loss, mean_loss))

    def crossover(self):
        population = copy.deepcopy(self.population)
        for idx1, idx2 in CrossOver.get_idx_pair_list(len(population)):
            if random.random() <= self.cross_over_p:
                GPModule.change_random_sub_tree(population[idx1], population[idx2])
        
        return population



    def tree_mutation(self):
        # 트리 내에 서브 트리를 새롭게 생성하는 mutation
        population = copy.deepcopy(self.population)
        for i, individual in enumerate(population):
            if random.random() <= self.mutation_p: 
                if GPModule.mutation(individual) == False:
                    population[i] = GPModule.get_new_module(self.input_num, 0)
                                
        return population

    def value_mutation(self):
        # 트리 내에 상수 값을 임의로 조금 바꾸는 mutation
        population = copy.deepcopy(self.population)
        for individual in population:
            if random.random() <= self.mutation_p: 
                GPModule.change_constant_value(individual)                
        return population

    def next_population1(self, preserve_num) -> None:
        top_individual_list = self.get_top_individual_list(preserve_num)

        self.population = self.selection()
        self.population = self.crossover()
        self.population = self.tree_mutation()
        self.delete_bottom_individuals(len(top_individual_list))
        self.population.extend(top_individual_list)
        self.generation += 1

    def next_population2(self, preserve_num) -> None:        
        top_individual_list = self.get_top_individual_list(preserve_num)

        self.population = self.selection()
        self.population = self.value_mutation()
        self.delete_bottom_individuals(len(top_individual_list))
        self.population.extend(top_individual_list)
        self.generation += 1