import sys
import random
import time
import copy
import numpy as np
from collections import OrderedDict
from agent_zoo.weight_writer import weight_writer

from agent_zoo.Eval import Eval

NAME = 'B5'
HIGH = 1.5
LOW = .5
SEED = 12
MAX_GEN = 100
MAX_POPULATION = 20
CLONE_RATE = .1
CLONES = int(.05 * MAX_POPULATION)
MAX_FRAME = 200  # how many frames is the robot simulated for

# #B3
# HIGH = 5
# LOW = -5
# SEED = 12
# MAX_GEN = 1000
# MAX_POPULATION = 30
# CLONE_RATE = .05
# CLONES = int(.5 * MAX_POPULATION)
# MAX_FRAME = 50  # how many frames is the robot simulated for

class Individual(object):

    def __init__(self, svd_dic, genotype=None):
        self.svd_dic = svd_dic #a reference to all the precomputed SVDs
        if genotype is None:
            self.genotype = [ random.uniform(LOW, HIGH) for i in range(len(svd_dic))]
        else:
            self.genotype = genotype
        self.fitness = None

    def get_weights(self):
        weights = {}
        i = 0
        for layer in self.svd_dic:
            if self.svd_dic[layer][0] is True:
                U, s, V = copy.deepcopy(self.svd_dic[layer][1])
                s *= self.genotype[i]
                weights[layer] = np.matmul( U ,np.matmul(np.diag(s), V))
            else: # a bias layer so matrix multiplication is not necessary
                weights[layer] = self.genotype[i] * copy.deepcopy(self.svd_dic[layer][1])
            i += 1
        return weights

    def mate(self, partner):
        gt = []
        for i in range(len(self.genotype)):
            if random.randint(0,99) % 2 == 0:
                gt.append(self.genotype[i])
            else:
                gt.append(partner.genotype[i])
        return Individual(self.svd_dic, genotype=gt)

# returns a list individuals that have been selected
def select_parents(population, selection_rate):
    total_fitness = 0
    for indiv in population:
        if indiv.fitness is not None:
            total_fitness += indiv.fitness
    selected = []

    # create a random list of indices
    order = [i for i in range(len(population))]
    random.shuffle(order)
    how_many = int(selection_rate * len(population))
    index = 0
    while len(selected) < how_many:
        indiv = population[order[index]]
        if (indiv.fitness / total_fitness) > random.random():
            selected.append(indiv)
        index += 1
        if index > (len(population) -1):
            index = 0

    return selected

def mutate(individual):
    index = random.randint(0, len(individual.genotype) -1)
    # individual.genotype[index] = random.uniform(LOW, HIGH) * individual.genotype[index]
    individual.genotype[index] = random.uniform(LOW, HIGH)
def clone(individuals):
    clones = []
    for indiv in individuals:
        for i in range(CLONES):
            clones.append(Individual(indiv.svd_dic, copy.deepcopy(indiv.genotype)))
    return clones

def main():
    start_time = time.process_time()
    random.seed(SEED)

    weightfile = 'RoboschoolAnt_v1_2017jul.weights'
    original = {}
    exec(open(weightfile).read(), original)
    layerNames = ['weights_dense1_w', 'weights_dense1_b', 'weights_dense2_w', 'weights_dense2_b', 'weights_final_w',
                  'weights_final_b']

    svd_dict = OrderedDict()
    for layer in layerNames:
        if len(original[layer].shape) == 2:
            U, s, V = np.linalg.svd( original[layer], full_matrices=False)
            svd_dict[layer] = True, (U, s, V)
        else:
            svd_dict[layer] = False, original[layer]




    #generate initial population
    population = [Individual(svd_dict) for i in range(MAX_POPULATION)]

    print('Starting evolution')
    # base_indiv_fitness = evaluate_individual(original)
    with open('Experiment{}_results.csv'.format(NAME), 'w') as writer_results:
        with open('logEvalb.csv', 'w') as logger:
            logger.write('time\n')
            header = 'generation, run_time, avg_fitness, top_fitness'
            print(header)
            writer_results.write(header + '\n')


            for generation in range(MAX_GEN):
                start = time.process_time()
                for indiv in population:
                    if indiv.fitness is None:
                        indiv.fitness = Eval().evaluate_individual(MAX_FRAME, indiv.get_weights(), logger)
                #select individuals for reproduction
                selected = select_parents(population, CLONE_RATE)
                #generate children
                children = clone(selected)

                for child in children:
                    mutate(child)
                #evaluate children
                for child in children:
                    if child.fitness is None:
                        child.fitness = Eval().evaluate_individual(MAX_FRAME, child.get_weights(), logger)

                population.extend(children)

                population = sorted(population, key=lambda x: x.fitness, reverse=True)

                survivors_indices = [random.randint(3, len(population) -1) for i in range(MAX_POPULATION -1)]
                survivors = []
                survivors.append(population[0])
                survivors.append(population[1])
                survivors.append(population[2])
                for index in survivors_indices:
                    survivors.append(population[index])

                population = survivors

                total_fitness = 0
                for indiv in population:
                    total_fitness += indiv.fitness
                avg_fitness = total_fitness/len(population)
                run_time = time.process_time() - start
                result = '{}, {}, {}, {}'.format(generation, run_time, avg_fitness, population[0].fitness)
                print(result)
                writer_results.write(result +'\n')
                for indiv in population:
                    indiv.fitness = None

            with open('Elite_Individual_Experiment{}.weights'.format(NAME), 'w') as wrt:
                weight_writer(wrt, population[0].get_weights())

            total_time = time.process_time() - start_time
            print('RunTime: {}'.format(str(total_time)))








if __name__ == '__main__':
    main()