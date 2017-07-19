
from time import time
import h5py
import numpy as np
from pyNN.parameters import Sequence
from pyNN.random import NumpyRNG, RandomDistribution
import pyNN.nest as sim
from pyNN.recording.syncell_reader import Network

t1 = time()

sim.setup()

rng = NumpyRNG(seed=497562396)

neuron_params = {
    "cm": 0.2,  # nF
    "tau_m": 0.2/0.01,  # nF/uS
    "tau_refrac": 4.0,
    "v_reset": -50.0,
    "v_rest": -65.0,
    "delta_v": 0.5,  # mV
    "v_t_star": -48.0,
    "lambda0": 1.0,  # 1/s
    "tau_eta": Sequence([10.0, 50.0, 250.0]), # ms
    "a_eta": Sequence([0.2, 0.05, 0.025]),
    "tau_gamma": Sequence([5.0, 200.0, 250.0]),
    "a_gamma": Sequence([15.0, 3.0, 1.0]),
    "g_max": 1.0,  # uS
    "i_offset": 0.0,  # nA
    "tau_syn_fast_rise": {
        "exc": 0.2,
        "inh": 0.2,
    },  # ms
    "tau_syn_fast_decay": {
        "exc": 1.7,
        "inh": 8.0,
    },  # ms
    "tau_syn_slow_rise": {
        "exc": 0.29,
        "inh": 3.5,
    },  # ms
    "tau_syn_slow_decay": {
        "exc": 43.0,
        "inh": 260.9,
    },  # ms
    "e_syn_fast": {
        "exc": 0.0,
        "inh": -97.0
    },  # mV
    "e_syn_slow": {
        "exc": 0.0,
        "inh": -80.0
    },  # mV
    "ratio_slow_fast": {
        "exc": 0.5,
        "inh": 0.0
    },
    "mg_conc": {
        "exc": 1.0,
        "inh": 0.0
    },  # mM
    "tau_corr": {
        "exc": 5.0,
        "inh": 5.0
    },
}
synapse_parameters = {
    "weight": RandomDistribution("normal", (0.5, 0.1), rng=rng),
    "delay": RandomDistribution("normal", (1.0, 0.1), rng=rng),
    "U":  RandomDistribution("normal", (0.8, 0.01), rng=rng),
}

neurons = sim.Population(1000, sim.RoessertEtAl(**neuron_params))
connections_exc = sim.Projection(neurons, neurons, sim.FixedProbabilityConnector(p_connect=0.05),
                                 sim.TsodyksMarkramSynapseEM(**synapse_parameters),
                                 receptor_type="exc")
connections_inh = sim.Projection(neurons, neurons, sim.CloneConnector(connections_exc),
                                 sim.TsodyksMarkramSynapseEM(**synapse_parameters),
                                 receptor_type="inh")

network = Network()
network.populations = [neurons]
network.projections = [connections_exc, connections_inh]

t2 = time()
print("Time to build network: {} s".format(t2 - t1))

network.save_to_syncell_file("test.h5")

t3 = time()
print("Time to save network to file: {} s".format(t3 - t2))


network2 = Network.from_syncell_file("test.h5")

assert network2.populations[0].size == neurons.size
assert network2.projections[0].size() == connections_exc.size(), "{} != {}".format(network2.projections[0].size(), connections_exc.size())
assert network2.projections[1].size() == connections_inh.size(), "{} != {}".format(network2.projections[1].size(), connections_inh.size())

t4 = time()
print("Time to load network from file: {} s".format(t4 - t3))

network2.save_to_syncell_file("test2.h5")

t5 = time()
print("Time to save network to file again: {} s".format(t5 - t4))

# check the two files are identical
f1 = h5py.File("test.h5", "r")
f2 = h5py.File("test2.h5", "r")
for table_name in ("neurons", "presyn", "postsyn"):
    print("\n======== {} ========\n".format(table_name))
    for name in f1[table_name]["default"].keys():
        x1 = f1[table_name]["default"][name].value
        x2 = f2[table_name]["default"][name].value
        if not np.all(np.abs(x1 - x2) < 1e-9):
            errmsg = "\n{}:\n{}\nnot equal to\n{}\n".format(name, x1, x2)
            print(errmsg)


"""
I think it is fair to say that many (a majority of?) neurophysiologists aren’t interested in sharing data, only wish to share data in the context of a formal collaboration with clear goals and a co-authorship agreement, and/or don’t think there is any point in sharing most data since a given dataset is usually tightly coupled to a particular and precise experimental context.
"""