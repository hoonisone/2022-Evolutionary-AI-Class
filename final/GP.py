import random
from enum import IntEnum
from GPModule import GPModule

class GenericProgramming:
    def __init__(self, population_size, input_num, cross_over_p, mutation_p) -> None:
        self.population_size = population_size
        self.generation = 0
        self.input_num = input_num
        self.cross_over_p = cross_over_p
        self.mutation_p = mutation_p
        self.population = self.generate_population()

    def generate_population(self) -> list:
        # 새로운 population 생성
        return [GPModule.get_new_module(self.input_num, 0) for i in range(self.population_size)]

    def fitness(self, individual) -> float:
        # single individual에 대한 fitness 계산
        pass

    def get_fitness_list(self) -> list:
        # 각 individual의 fitness 리스트 반환
        return [self.fitness(individual) for individual in self.population]

    def selection(self) -> list:
        # population에 selection 수행
        pass

    def mutation(self) -> list:
        # population에 mutation 수행
        pass

    def crossover(self) -> list:
        # population에 crossover 수행
        pass

    def next_population(self, preserve_num) -> None:
        top_individual_list = self.get_top_individual_list(preserve_num)
        self.population = self.selection()
        self.population = self.crossover()
        self.population = self.mutation()
        self.population[:preserve_num] = top_individual_list
        self.generation += 1

    def get_top_individual_index_list(self, num):
        # 상위 n개의 individual들의 idx list 반환
        origin_fitness_list = self.get_fitness_list()
        sorted_fitness_list = origin_fitness_list.copy()
        sorted_fitness_list.sort(reverse=True)
        return [origin_fitness_list.index(value) for value in sorted_fitness_list[:num]]
            
    def get_top_individual_list(self, num):
        index_list = self.get_top_individual_index_list(num)
        return [self.population[idx] for idx in index_list]

    def get_bottom_individual_index_list(self, num):
        # 하위 n개의 individual들의 idx list 반환
        origin_fitness_list = self.get_fitness_list()
        sorted_fitness_list = origin_fitness_list.copy()
        sorted_fitness_list.sort()
        return [origin_fitness_list.index(value) for value in sorted_fitness_list[:num]]

    def get_bottom_individual_list(self, num):
        index_list = self.get_bottom_individual_index_list(num)
        return [self.population[idx] for idx in index_list]

    def delete_bottom_individuals(self, num):
        idx_list = self.get_bottom_individual_index_list(num)
        idx_list.sort(reverse=True)
        for idx in idx_list:
            del self.population[idx]

    def get_best_individual_idx(self):
        fitness_list = self.get_fitness_list()
        max_fitness = max(fitness_list)
        return fitness_list.index(max_fitness)
    
    def get_best_individual(self):
        return self.population[self.get_best_individual_idx()]