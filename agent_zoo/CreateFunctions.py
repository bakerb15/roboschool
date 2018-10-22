import random
import math

def create_random_function():
    def f(x):
        return math.pow(x + random.random(), random.randint(2, 8))
    return f


def create_add_function(real):
    return lambda x: x + real

def create_constant_funcs(weight):
    return lambda x: weight


if __name__=="__main__":
    weights = [2.3, 4.5, 1.1, 5.5]
    functionlist = []
    for w in weights:
        functionlist.append(create_constant_funcs(w))
    for i in range(10):
        functionlist.append(create_random_function())
    for f in functionlist:
        print(type(f))
        print(f(1))