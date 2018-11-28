import sys
import random
import time
import copy
import numpy
from agent_zoo.weight_writer import weight_writer

from agent_zoo.Eval2 import Eval2

class Individual(object):

    def __init__(self, genome, size, base_weights, weight_map, genotype=None):
        self.genotype = []
        self.recalc_weight = []
        self.weight_map = weight_map
        self.base_weights = base_weights
        self.weights = {}
        self.genome = genome
        for layer in self.base_weights:
            self.weights[layer] = numpy.copy(self.base_weights[layer])
        self.size = size
        self.fitness = None
        if genotype is None:
            for i in range(size):
                gene = []
                gene_size = random.randint(0, 2)
                for j in range(gene_size):
                    gene.append(genome[random.randint(0, len(genome) - 1)])
                self.genotype.append(gene)
                self.recalc_weight.append(True)
        else:
            self.genotype = genotype
            self.recalc_weight = [True for i in range(size)]

    def get_weights(self):
        for i in range(self.size):
            if self.recalc_weight[i] is True:
                if len(self.weight_map[i]) == 2:
                    layer, dim1 = self.weight_map[i]
                    self.weights[layer][dim1] = self.base_weights[layer][dim1]
                    for f in self.genotype[i]:
                        self.weights[layer][dim1] = f(self.weights[layer][dim1])
                elif len(self.weight_map[i]) == 3:
                    layer, dim1, dim2 = self.weight_map[i]
                    self.weights[layer][dim1][dim2] = self.base_weights[layer][dim1][dim2]
                    #for f in self.genotype[i]:
                    #    self.weights[layer][dim1][dim2] = f(self.weights[layer][dim1][dim2])
                self.recalc_weight[i] = False

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

# Genome is a list of functions
def random_gene(genome):
    size = random.randint(1, 5)
    return [genome[random.randint(0, len(genome) - 1)]for i in range(size)]

def randfunc():
    dice = random.randint(0, 99)  # roll three sided dice
    y = random.uniform(-2, 2)
    if dice % 2 != 0:
        y = random.uniform(-0.1, 0.1)
        def f(x):
            return x + y
        return f
    else:
        y = random.uniform(-1.1, 1.1)
        def f(x):
            return x * y
        return f

def mutate(individual):
    howmany = random.randint(1, 10)
    for n in range(howmany):
        index = random.randint(0, individual.size - 1)
        gene = individual.genotype[index]
        #flip a coin if even just add to end of gene
        coin = random.randint(0,99)
        if coin % 2 == 0 or len(gene) == 0:
            gene.append(individual.genome[random.randint(0, len(individual.genome) -1)])
        else:
            func_index = random.randint(0, len(gene) -1)
            gene[func_index] = individual.genome[random.randint(0, len(individual.genome) -1)]
        individual.recalc_weight[index] = True
    individual.fitness = None

def mate(individuals):
    children = []
    for indiv in individuals:
        genotype = copy.deepcopy(indiv.genotype)
        children.append(Individual(indiv.genome, indiv.size, indiv.base_weights, indiv.weight_map, genotype=genotype))
    return children

def generate_genome(genome_size):
    genome = []
    for i in range(genome_size):
        genome.append(randfunc())
    return genome


def main():
    seed = 12
    genome_size = 10000
    MAX_GEN = 100
    selectionRate = 0.55
    IND_SIZE = 128 + 64 + 8
    MAX_POPULATION = 20
    MUTATION_RATE = .05
    CROSSOVER_RATE = .25
    MAX_FRAME = 600 #how many frames in the robot simulated for

    random.seed(seed)
    genome = generate_genome(genome_size)
    weightfile = 'RoboschoolAnt_v1_2017jul.weights'
    original = {}
    exec(open(weightfile).read(), original)
    layerNames = ['weights_dense1_w', 'weights_dense1_b', 'weights_dense2_w', 'weights_dense2_b', 'weights_final_w',
                  'weights_final_b']
    weight_map = []

    shapes = {}
    for layer in layerNames:
        shapes[layer] = original[layer].shape

    index = 0
    for layer in layerNames:
        if len(shapes[layer]) == 1:
            for i in range(shapes[layer][0]):
                weight_map.append((layer, i))
                index += 1
        #elif len(shapes[layer]) == 2:
         #   for i in range(0, shapes[layer][0]):
         #       for j in range(shapes[layer][1]):
         #           weight_map.append((layer, i, j))
         #           index += 1

    #generate initial population
    population = [Individual(genome, IND_SIZE, original, weight_map) for i in range(MAX_POPULATION)]
    #evaluate each individual in the initial population


    hall_of_fame = []

    # base_indiv_fitness = evaluate_individual(original)
    with open('results.csv', 'w') as writer_results:
        with open('Elite_Individual_Eperiment4.weights', 'w') as wwriter:
            with open('logEval.csv', 'w') as logger:
                logger.write('time\n')
                writer_results.write('generation, top_fitness\n')
                print('Starting evolution')

                for generation in range(MAX_GEN):
                    for indiv in population:
                        if indiv.fitness is None:
                            indiv.get_weights()
                            indiv.fitness = Eval2().evaluate_individual(MAX_FRAME, indiv.weights, logger)
                    #select individuals for reproduction
                    selected = select_parents(population, CROSSOVER_RATE)
                    #generate children
                    children = mate(selected)
                    #select children for mutation
                    for child in children:
                        if MUTATION_RATE > random.random():
                            mutate(child)

                    #evaluate children
                    for child in children:
                        if child.fitness is None:
                            child.get_weights()
                            child.fitness = Eval2().evaluate_individual(MAX_FRAME, child.weights, logger)

                    population.extend(children)

                    population = sorted(population, key=lambda x: x.fitness, reverse=True)

                    #select survivors
                    while len(population) > MAX_POPULATION:
                        population.pop()
                    result = '{}, {}'.format(generation, population[0].fitness)
                    print(result)
                    writer_results.write(result +'\n')

                    #add back original
                    population.append(Individual(genome, IND_SIZE, original, weight_map, genotype=[[] for i in range(genome_size)]))

                weight_writer(wwriter, population[0].weights)








if __name__ == '__main__':
    main()