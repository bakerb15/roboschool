import os
import numpy as np
import random
import time

def f0(x):
    return x + 0.00001

def f1(x):
    return x

def f2(x):
    return x - 0.0001

def f3(x):
    return x + 0.03

def f4(x):
    return 1.002 * x

def f5(x):
    return x * 0.009

def f6(x):
    return x - .00031

def f7(x):
    return x * 1.001

funcs = [f0, f1, f2, f3, f4, f5, f6, f7]

def GeneratePopulation( maxPopulation, seed):
    listA = [f6]
    listB = [f6]
    random.seed(seed)
    weightfile = 'RoboschoolAnt_v1_2017jul.weights'
    weights = {}
    exec(open(weightfile).read(), weights)

    layerNames = ['weights_dense1_w','weights_dense1_b', 'weights_dense2_w', 'weights_dense2_b','weights_final_w', 'weights_final_b']

    shapes = {}
    for layer in layerNames:
        shapes[layer] = weights[layer].shape

    size = 12488

    population = []
    start = time.process_time()
    #we always include base individual so subtract 1 from maxPopulation
    for i in range(maxPopulation - 1):
        dna = []
        for j in range(size):
            chance = random.randint(0, 100)
            if chance > 90:
                index = random.randint(0, len(funcs) - 1)
            else:
                index = 1
            dna.append(index)
        population.append(dna)
    end = time.process_time()

    print('Initial population of {} generated from seed {}'.format(maxPopulation, seed))
    print('Time to generate initial population: {}'.format(end - start))


    populationWeightDicts = []
    organismID = 0
    for funclist in population:
        count = 0
        individual ={}
        for layer in layerNames:
            individual[layer] = []
            if len(shapes[layer]) == 1:
                for i in range(shapes[layer][0]):
                    individual[layer].append(funcs[funclist[count]](weights[layer][i]))
                    count += 1
            elif len(shapes[layer]) == 2:
                for i in range(0, shapes[layer][0]):
                    row = []
                    for j in range(shapes[layer][1]):
                        #print('func: {}\tlayer: {}\ti: {}\tj: {}'.format( count, layer, i , j))
                        w = funcs[funclist[count]](weights[layer][i][j])
                        row.append(w)
                        count += 1
                    individual[layer].append(row)
            individual[layer] = np.array(individual[layer])
        populationWeightDicts.append((organismID, individual))
        organismID += 1

    #add base individual
    populationWeightDicts.append(('base', weights))
    return populationWeightDicts
