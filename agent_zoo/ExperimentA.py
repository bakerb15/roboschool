import sys
import random
import time
import copy
import numpy
import math

from agent_zoo.Eval import Eval
from agent_zoo.weight_writer import weight_writer



#NAME = 'A2'
# SEED = 12
# GENOME_SIZE = 100000
# MAX_GEN = 200
# #SELECTION_RATE = 0.55
# IND_SIZE = 12488
# MAX_POPULATION = 20
# MUTATION_RATE = .35
# CROSSOVER_RATE = .15
# MAX_FRAME = 150
# LOW, HIGH = -2, 2
# MAX_GENE_INIT = 10
# MAX_GENE_LENGTH = 35


# NAME = 'A3'
# SEED = 12
# GENOME_SIZE = 100000
# MAX_GEN = 200
# IND_SIZE = 12488
# MAX_POPULATION = 20
# MUTATION_RATE = .35
# CROSSOVER_RATE = .15
# MAX_FRAME = 150
# LOW_MUL, HIGH_MUL = 0.99, 1.01
# LOW_ADD, HIGH_ADD = -0.1, 0.1
# MIN_GENE_INIT, MAX_GENE_INIT = 0 , 10
# MAX_GENE_LENGTH = 50

NAME = 'A5'
SEED = 12
GENOME_SIZE = 100000
MAX_GEN = 50
IND_SIZE = 12488
MAX_POPULATION = 50
MUTATION_RATE = .35
CROSSOVER_RATE = .15
MAX_FRAME = 200
LOW_MUL, HIGH_MUL = 0.99, 1.01
LOW_ADD, HIGH_ADD = -0.1, 0.1
MIN_GENE_INIT, MAX_GENE_INIT = 0 , 10
MAX_GENE_LENGTH = 50
NUM_OF_EVALS = 4

# NAME = 'A4'
# SEED = 12
# GENOME_SIZE = 100000
# MAX_GEN = 100
# IND_SIZE = 12488
# MAX_POPULATION = 25
# MUTATION_RATE = .65
# CROSSOVER_RATE = .25
# MAX_FRAME = 200
# LOW_MUL, HIGH_MUL = 0.9, 1.1
# LOW_ADD, HIGH_ADD = -0.1, 0.1
# MIN_GENE_INIT, MAX_GENE_INIT = 40, 50
# MAX_GENE_LENGTH = 100

# NAME = 'ExperB'
# SEED = 12
# GENOME_SIZE = 100000
# MAX_GEN = 50
# IND_SIZE = 12488
# MAX_POPULATION = 1
# MUTATION_RATE = .0
# CROSSOVER_RATE = .0
# MAX_FRAME = 200
# LOW_MUL, HIGH_MUL = 0.9, 1.1
# LOW_ADD, HIGH_ADD = -0.1, 0.1
# MIN_GENE_INIT, MAX_GENE_INIT = 0, 0
# MAX_GENE_LENGTH = 0

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
                gene_size = random.randint(MIN_GENE_INIT, MAX_GENE_INIT)
                for j in range(gene_size):
                    gene.append(genome[random.randint(0, len(genome) - 1)])
                self.genotype.append(gene)
                self.recalc_weight.append(True)
        else:
            for g in genotype:
                self.genotype.append(copy.deepcopy(g))
            self.recalc_weight = [True for i in range(self.size)]

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
                    for f in self.genotype[i]:
                        self.weights[layer][dim1][dim2] = f(self.weights[layer][dim1][dim2])
                self.recalc_weight[i] = False
        return self.weights

# returns a list individuals that have been selected for parenthood
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
    dice = random.randint(0, 99) % 2
    if dice == 0:
        y = random.uniform(LOW_ADD, HIGH_ADD)
        def f(x):
            return x + y
        return f
    elif dice == 1:
        y = random.uniform(LOW_MUL, HIGH_MUL)
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
        if len(gene) > MAX_GENE_LENGTH:
            while len(gene) > MAX_GENE_LENGTH:
                gene.pop()
    individual.fitness = None

def generate_genome(genome_size):
    genome = []
    for i in range(genome_size):
        genome.append(randfunc())
    return genome

def mate(individuals):
    children = []
    for indiv in individuals:
        mate = individuals[random.randint(0, len(individuals) -1)]
        genotype_ref = []
        coin = random.randint(0,99) % 2
        crossover_point = random.randint(0, indiv.size - 1)
        for index in range(crossover_point):
            genotype_ref.append(indiv.genotype[index])
        for index in range(crossover_point, indiv.size):
            genotype_ref.append(mate.genotype[index])
        children.append(Individual(indiv.genome, indiv.size, indiv.base_weights, indiv.weight_map, genotype_ref))
    return children


def select_survivors(population):
    total_fitness = 0
    for indiv in population:
        if indiv.fitness is not None:
            total_fitness += indiv.fitness

    surv_dic = {}
    survivors = []
    attempts = 0
    while len(survivors) < MAX_POPULATION and attempts < 2:
        for indiv in population:
            if indiv not in surv_dic and indiv.fitness is not None:
                if random.random() < indiv.fitness/total_fitness:
                    survivors.append(indiv)
                    surv_dic[indiv] = True
        attempts += 1
    while len(survivors) < MAX_POPULATION:
        for indiv in population:
            if indiv not in surv_dic and indiv.fitness is not None:
                survivors.append(indiv)
                surv_dic[indiv] = True
                if len(survivors) >= MAX_POPULATION:
                    break

    return survivors



def main():

    start_time = time.process_time()
    random.seed(SEED)
    genome = generate_genome(GENOME_SIZE)
    #weightfile = 'RoboschoolAnt_v1_2017jul.weights'
    weightfile = 'Elite_Individual_ExperimentB.weights'
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
        elif len(shapes[layer]) == 2:
            for i in range(0, shapes[layer][0]):
                for j in range(shapes[layer][1]):
                    weight_map.append((layer, i, j))
                    index += 1

    #generate initial population
    population = [Individual(genome, IND_SIZE, original, weight_map) for i in range(MAX_POPULATION)]
    #evaluate each individual in the initial population


    # base_indiv_fitness = evaluate_individual(original)
    with open('Experiment{}_results.csv'.format(NAME), 'w') as writer:
        with open('logEval.csv', 'w') as logger:
            logger.write('time\n')
            writer.write('generation, avg, fitness, top_fitness\n')
            print('Starting evolution')

            for generation in range(MAX_GEN):
                for indiv in population:
                    indiv.fitness = 0
                    for i in range(NUM_OF_EVALS):
                        indiv.fitness += Eval().evaluate_individual(MAX_FRAME, indiv.get_weights(), logger)
                    indiv.fitness = indiv.fitness / NUM_OF_EVALS
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
                    child.fitness = 0
                    for i in range(NUM_OF_EVALS):
                        child.fitness += Eval().evaluate_individual(MAX_FRAME, child.get_weights(), logger)
                    child.fitness = child.fitness/ NUM_OF_EVALS
                population.extend(children)
                # select survivors
                population = select_survivors(population)


                population = sorted(population, key=lambda x: x.fitness, reverse=True)

                avg_fitness = 0
                total_fitness = 0
                for indiv in population:
                    total_fitness += indiv.fitness
                avg_fitness = total_fitness/ len(population)
                results = '{}, {}, {}'.format(generation, avg_fitness, population[0].fitness)
                print(results)
                writer.write(results + '\n')

    with open('Elite_Individual_Experiment{}.weights'.format(NAME), 'w') as wwriter:
        weight_writer(wwriter, population[0].get_weights())

    total_time = time.process_time() - start_time
    print('RunTime: {}'.format(str(total_time)))



if __name__ == '__main__':
    main()