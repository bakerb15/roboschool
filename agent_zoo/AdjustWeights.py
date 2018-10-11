import os

weightfile = 'RoboschoolAnt_v1_2017jul.weights'
weights = {}
exec(open(weightfile).read(), weights)

layerNames = ['weights_dense1_w','weights_dense1_b', 'weights_dense2_w', 'weights_dense2_b','weights_final_w', 'weights_final_b']

size = 12488

count = 0
for item in layerNames:
    if len(weights[item].shape) == 1:

    elif len(weights[item].shape) == 2:


