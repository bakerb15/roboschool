import sys
import random
import time
import copy
import numpy

from agent_zoo.Eval import evaluate_individual

class Individual(object):

    def __init__(self, genome, size, base_weights, weight_map):
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
        for i in range(size):
            gene = []
            gene_size = random.randint(0, 2

                                       )
            for j in range(gene_size):
                gene.append(genome[random.randint(0, len(genome) - 1)])
            self.genotype.append(gene)
            self.recalc_weight.append(True)

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

def generate_genome(genome_size):
    genome = []
    for i in range(genome_size):
        genome.append(randfunc())
    return genome


def main():
    seed = 12
    genome_size = 1000000
    MAX_GEN = 25
    selectionRate = 0.55
    IND_SIZE = 12488
    MAX_POPULATION = 50
    MUTATION_RATE = .35
    CROSSOVER_RATE = .35

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
        elif len(shapes[layer]) == 2:
            for i in range(0, shapes[layer][0]):
                for j in range(shapes[layer][1]):
                    weight_map.append((layer, i, j))
                    index += 1




    #generate initial population
    population = [Individual(genome, IND_SIZE, original, weight_map) for i in range(MAX_POPULATION)]
    #evaluate each individual in the initial population

    base_indiv_fitness = evaluate_individual(original)
    with open('evolutionExperiment2.results', 'w') as writer:
        with open('evolutionExperiment2.timing', 'w') as writer_timing:
            writer.write('generation, top_fitness\n')
            writer_timing.write('generation, eval_time, sort_tim, mutation_time, gen_time, avg_get_weight, avg_eval\n')
            print('Starting evolution')
            for generation in range(MAX_GEN):
                start = time.process_time()
                total_time_get_weights = 0
                total_time_eval = 0
                for indiv in population:
                    if indiv.fitness is None:
                        start_get_weight = time.process_time()
                        w = indiv.get_weights()
                        total_time_get_weights += time.process_time() - start_get_weight
                        start_eval = time.process_time()
                        indiv.fitness = evaluate_individual(w)
                        total_time_eval += time.process_time() - start_eval
                avg_get_weight = total_time_get_weights/ len(population)
                avg_eval = total_time_eval/ len(population)
                end_eval = time.process_time()

                population = sorted(population, key=lambda x: x.fitness)
                end_sort = time.process_time()
                writer.write('{}, {}\n'.format(generation, population[0].fitness))
                for indiv in population:
                    mutate(indiv)
                end_mutation = time.process_time()

                eval_time = end_eval - start
                sort_time = end_sort - end_eval
                mutation_time = end_mutation - end_sort
                gen_time = end_mutation - start
                timing_str = '{}, {}, {}, {}, {}, {}, {}'.format(
                    generation, eval_time, sort_time, mutation_time, gen_time, avg_get_weight, avg_eval)
                print(timing_str)
                writer_timing.write(timing_str + '\n')


                #select individuals for reproduction

                #generate children

                #select children for mutation

                #evaluate individuals with no fitness

                #select survivors

if __name__ == '__main__':
    main()