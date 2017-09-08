
from time import time
from itertools import chain
import h5py
import numpy as np
from pyNN.recording.syncell_reader import Network

describe_network = False

t3 = time()

network2 = Network.from_syncell_files("test_neurons.h5", "test_synapses.h5")

t4 = time()
print("Time to load network from file: {} s".format(t4 - t3))

if describe_network:
    for item in chain(network2.populations, network2.stim_populations, network2.stim_spontaneous, network2.other_populations):
        print(item.describe(template="-- Population '{{label}}' {{size}} {{celltype.name}} neurons, {{first_id}}-{{last_id}} {{annotations}}",
                            engine='jinja2'))
    for item in chain(network2.projections, network2.stim_projections, network2.spontaneous_projections, network2.other_projections):
        print(item.describe(template="-- Projection '{{label}}' with {{size}} connections from {{pre.label}} to {{post.label}}, receptor type '{{receptor_type}}'",
                            engine='jinja2'))
        #if item.size() > 0:
        #    print(tabulate(item.get(('weight', 'delay'), format='list')))


t4a = time()
print("Time to describe loaded network: {} s".format(t4a - t4))

network2.save_to_syncell_files("test2_neurons.h5", "test2_synapses.h5")

t5 = time()
print("Time to save network to file again: {} s".format(t5 - t4a))

# check the two pairs of files are identical
f1 = h5py.File("test_neurons.h5", "r")
f2 = h5py.File("test2_neurons.h5", "r")
for table_name in ("neurons",):
    print("\n======== {} ========\n".format(table_name))
    for name in f1[table_name]["default"].keys():
        x1 = f1[table_name]["default"][name].value
        x2 = f2[table_name]["default"][name].value
        if not np.all(np.abs(x1 - x2) < 1e-9):
            errmsg = "\n{}:\n{}\nnot equal to\n{}\n".format(name, x1, x2)
            print(errmsg)
f1.close()
f2.close()
f1 = h5py.File("test_synapses.h5", "r")
f2 = h5py.File("test2_synapses.h5", "r")
for table_name in ("presyn", "postreceptors", "stimulation"):
    print("\n======== {} ========\n".format(table_name))
    for name in f1[table_name]["default"].keys():
        x1 = f1[table_name]["default"][name].value
        x2 = f2[table_name]["default"][name].value
        if not np.all(np.abs(x1 - x2) < 1e-9):
            errmsg = "\n{}:\n{}\nnot equal to\n{}\n".format(name, x1, x2)
            print(errmsg)
f1.close()
f2.close()

"""
I think it is fair to say that many (a majority of?) neurophysiologists aren’t interested in sharing data, only wish to share data in the context of a formal collaboration with clear goals and a co-authorship agreement, and/or don’t think there is any point in sharing most data since a given dataset is usually tightly coupled to a particular and precise experimental context.
"""