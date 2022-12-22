from abc import *
import random

class Selection:
    def __init__(self):
        pass

    @ staticmethod
    def roulette_selection(fitness_list):
        # fitness_list를 참조하여 새로운 population에 들어갈 idx_list를 반환
        total = sum(fitness_list)
        index_population = []
        for i in range(len(fitness_list)):
            target = random.random()*total
            acc = 0
            for j in range(len(fitness_list)):
                acc += fitness_list[j]
                if target <= acc:
                    index_population.append(j)
                    break
        return index_population

class CrossOver:
    @staticmethod
    def get_idx_pair_list(num:int) -> list:
        idx_list = list(range(num))
        random.shuffle(idx_list)
        return [idx_list[2*i:2*i+2] for i in range(num//2)]

    @staticmethod
    def get_random_idx(len:int, num:int) -> list:
         idx_list = list(range(1, len))
         random.shuffle(idx_list)
         return sorted(idx_list[0:num])

    @staticmethod
    def cross_over_for_pair(sample1, sample2, cross_point:int, idx_list = None):
        if idx_list == None:
            idx_list = CrossOver.get_random_idx(len(sample1), cross_point)
        idx_list.insert(0, 0)
        idx_list.append(len(sample1))
        new1 = []
        new2 = []
        for i in range(len(idx_list)-1):
            from_idx = idx_list[i]
            to_idx = idx_list[i+1]
            part1 = sample1[from_idx:to_idx]
            part2 = sample2[from_idx:to_idx]
            if i%2 == 0:
                new1.extend(part1)
                new2.extend(part2)
            else:
                new1.extend(part2)
                new2.extend(part1)

        return new1, new2

    @staticmethod
    def arithmetic_blend_crossover(sample1, sample2, part_size, cross_num):
        num = int(len(sample1)/part_size)
        alpha = random.random()
        cross_idx_list = [ int(alpha*part_size-1)]
        new1 = []
        new2 = []
        for i in range(num):
            part1 = sample1[i*part_size:(i+1)*part_size]
            part2 = sample2[i*part_size:(i+1)*part_size]
            part1, part2 = CrossOver.cross_over_for_pair(part1, part2, cross_num)
            new1.extend(part1)
            new2.extend(part2)
        return new1, new2

    @staticmethod
    def do(population:list, cross_num:int, probability:float) -> list:
        population = population.copy()
        idx_pair_list = CrossOver.get_idx_pair_list(len(population))
        for (a, b) in idx_pair_list:
            if random.random() <= probability:
                population[a], population[b] = CrossOver.cross_over_for_pair(population[a], population[b], cross_num)    
        return population

    @staticmethod
    def do_artithmetic(population:list, cross_num:int, probability:float, part_size) -> list:
        population = population.copy()
        idx_pair_list = CrossOver.get_idx_pair_list(len(population))
        for (a, b) in idx_pair_list:
            if random.random() <= probability:
                population[a], population[b] = CrossOver.arithmetic_blend_crossover(population[a], population[b], part_size, cross_num)    
        return population


class Mutation:
    @staticmethod
    def mutation_for_individual(individual:list, probability:float) -> list:
        individual = individual.copy()
        for i in range(len(individual)):
            if random.random() <= probability:
                individual[i] = not individual[i]
        return individual

    @staticmethod
    def do(population:list, probability:float) -> list:
        return [Mutation.mutation_for_individual(individual, probability) for individual in population]

