import numpy

ORDER =  ["weights_dense1_w", "weights_dense1_b", "weights_dense2_w", "weights_dense2_b", "weights_final_w", "weights_final_b"]

def weight_writer(writer, weight_dictionary):
    writer.write('import numpy as np\n\n')
    for name in ORDER:
        if len(weight_dictionary[name].shape) == 1:
            items = ""
            for num in weight_dictionary[name]:
                items += '{},'.format(str(num))
            writer.write('{} = np.array([ {} ])\n\n'.format(name, items))
        elif len(weight_dictionary[name].shape) == 2:
            writer.write('{} = np.array([\n'.format(name))
            for row in weight_dictionary[name]:
                items = ""
                for num in row:
                    items += '{},'.format(str(num))
                writer.write('[ {} ],\n'.format(items))
            writer.write('])\n\n')

