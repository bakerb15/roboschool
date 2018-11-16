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

def evaluate(individual):
    return 1,

def mutate(individual):
    return individual

def generate_genome(genome_size):
    genome = []
    for i in range(genome_size):
        genome.append(randfunc())
    return genome


def main():
    seed = 12
    genome_size = 10
    MAX_GEN = 100
    tourneysize = 5
    selectionRate = 0.55
    IND_SIZE = 12488
    MAX_POPULATION = 100
    MUTATION_RATE = .35
    CROSSOVER_RATE = .35

    random.seed(seed)
    weightfile = 'RoboschoolAnt_v1_2017jul.weights'
    original = {}
    exec(open(weightfile).read(), original)
    genome = generate_genome(genome_size)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('attr_gene', random_gene, genome)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.attr_gene, n=IND_SIZE)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate)
    toolbox.register('mutate', mutate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register('select', tools.selTournament, tourneysize=5)
    start = time.process_time()
    toolbox.population(n=MAX_POPULATION)
    end = time.process_time()
    print('Initial population of {} individuals has been generated in {} seconds.'.format(MAX_POPULATION, end - start))

    for generation in range(MAX_GEN):
        offsprings = toolbox.select(toolbox.population, 20)
        offsprings = map(toolbox.clone, offsprings)

        for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offsprings:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        need_evaluating = [ind for ind in offsprings if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, need_evaluating)
        for ind, fit in zip(need_evaluating, fitnesses):
            ind.fitness.values = fit

        toolbox.population[:] = offsprings

if __name__ == '__main__':
    main()