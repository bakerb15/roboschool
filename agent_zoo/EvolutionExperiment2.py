import sys
import random
import time

from deap import base
from deap import creator
from deap import tools

# Genome is a list of functions
def random_gene(genome):
    size = random.randint(1, 5)
    return [genome[random.randint(0, len(genome) - 1)]for i in range(size)]


def randfunc():
    dice = random.randint(0, 99)  # roll three sided dice
    y = random.uniform(-2, 2)
    if dice % 2 == 0:
        def f(x):
            return x + y
        return f
    else:
        def f(x):
            return x * y
        return f


def generate_genome(genome_size):
    genome = []
    for i in range(genome_size):
        genome.append(randfunc())
    return genome


def main():
    seed = int(sys.argv[1])
    genome_size = int(sys.argv[2])
    random.seed(seed)
    genome = generate_genome(genome_size)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    IND_SIZE = 12488
    MAX_POPULATION = 100

    toolbox = base.Toolbox()
    toolbox.register("attr_gene", random_gene, genome)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_gene, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    start = time.process_time()
    toolbox.population(n=MAX_POPULATION)
    end = time.process_time()
    print('Initial population of {} individuals has been generated in {} seconds.'.format(MAX_POPULATION, end - start))



if __name__ == '__main__':
    main()