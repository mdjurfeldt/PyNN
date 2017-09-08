
from time import time
from itertools import chain
import h5py
import numpy as np
from pyNN.parameters import Sequence
from pyNN.random import NumpyRNG, RandomDistribution
import pyNN.nest as sim
from pyNN.recording.syncell_reader import Network

describe_network = False
skip_creating_spontaneous_connections = True

t1 = time()

sim.setup()

rng = NumpyRNG(seed=497562396)

neuron_params = {
    "cm": RandomDistribution('normal', (0.2, 0.02)),  # nF
    "tau_m": RandomDistribution('normal', (20.0, 2.0)),  # nF/uS
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
    "i_rho_thresh": 0.0123,
    "f_rho_stim": 5.0,  # %,
    "i_hyp": 0.0789,
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

n_neurons = 4000
n_stim = n_neurons//10
n_other = n_neurons//10
n_connections = 200  # /per neuron

neurons = sim.Population(n_neurons, sim.RoessertEtAl(**neuron_params), label="neurons")
connections_exc = sim.Projection(neurons, neurons, sim.FixedNumberPostConnector(n=n_connections,
                                                                                with_replacement=False,
                                                                                allow_self_connections=False),
                                 sim.TsodyksMarkramSynapseEM(**synapse_parameters),
                                 receptor_type="exc", label="excitatory connections")
connections_inh = sim.Projection(neurons, neurons, sim.CloneConnector(connections_exc),
                                 sim.TsodyksMarkramSynapseEM(**synapse_parameters),
                                 receptor_type="inh", label="inhibitory connections")
stim = sim.Population(n_stim, sim.SpikeSourceArray(), label="external stimulation")
connections_stim = sim.Projection(stim, neurons, sim.FixedNumberPostConnector(n=n_connections,
                                                                              with_replacement=False),
                                  sim.TsodyksMarkramSynapseEM(**synapse_parameters),
                                  receptor_type="exc", label="external input connections")

connections_exc._size = n_neurons * n_connections  # optimization
connections_inh._size = n_neurons * n_connections
connections_stim._size = n_stim * n_connections

t1a = time()
print("Time to build network (part 1): {} s".format(t1a - t1))

stim_spontaneous = []
connections_spontaneous = []
for projection in (connections_exc, connections_inh, connections_stim):
    #assert projection.size() == projection.pre.size * n_connections, "{} != {}".format(projection.size(), projection.pre.size * n_connections)
    stim_spontaneous.append(
        sim.Population(projection.pre.size * n_connections,
                       sim.SpikeSourcePoisson(rate=10.0),
                       label="Poisson source for {}".format(projection.label))
    )
    if not skip_creating_spontaneous_connections:
        # creating these connections is very slow because of the projection.get(),
        # and they are not needed for writing the synapse file
        connection_properties = np.array(projection.get(('weight', 'delay', 'U'), format='list'))
        connection_properties[:, 0] = np.arange(projection.size())
        connections_spontaneous.append(
            sim.Projection(stim_spontaneous[-1], projection.post,
                           sim.FromListConnector(connection_properties, column_names=('weight', 'delay', 'U')),
                           sim.TsodyksMarkramSynapseEM(),
                           receptor_type=projection.receptor_type,
                           label="spontaneous mEPSP connection for {}".format(projection.label))
        )

t1b = time()
print("Time to build network (part 2): {} s".format(t1b - t1a))

other_hypercolumn = sim.Population(n_other, sim.RoessertEtAl(**neuron_params), label="other neurons")
# note that we do not export "other_hypercolumn" to the neurons file,
# but we do export the connections coming from it to the synapses file
connections_other = sim.Projection(other_hypercolumn, neurons,
                                   #sim.FixedProbabilityConnector(p_connect=0.03),
                                   sim.FixedNumberPostConnector(n=n_connections,
                                                                with_replacement=False),
                                   sim.TsodyksMarkramSynapseEM(**synapse_parameters),
                                   receptor_type="exc", label="excitatory connections from other hypercolumn")
connections_other._size = n_other * n_connections


network = Network()
network.populations = [neurons]
network.stim_populations = [stim]
network.projections = [connections_exc, connections_inh]
network.stim_projections = [connections_stim]
network.stim_spontaneous = stim_spontaneous
network.spontaneous_projections = connections_spontaneous
network.other_populations = [other_hypercolumn]
network.other_projections = [connections_other]

t2 = time()
print("Time to build network (part 3): {} s".format(t2 - t1b))
print("Time to build network (total): {} s".format(t2 - t1))

if describe_network:

    #from tabulate import tabulate

    for item in chain(network.populations, network.stim_populations, network.stim_spontaneous, network.other_populations):
        print(item.describe(template="-- Population '{{label}}' {{size}} {{celltype.name}} neurons, {{first_id}}-{{last_id}} {{annotations}}",
                            engine='jinja2'))
    for item in chain(network.projections, network.stim_projections, network.spontaneous_projections, network.other_projections):
        print(item.describe(template="-- Projection '{{label}}' with {{size}} connections from {{pre.label}} to {{post.label}}, receptor type '{{receptor_type}}'",
                            engine='jinja2'))
        #print(tabulate(item.get(('weight', 'delay'), format='list')))


t2a = time()
print("Time to describe network: {} s".format(t2a - t2))

network.save_to_syncell_files("test_neurons.h5", "test_synapses.h5")

t3 = time()
print("Time to save network to file: {} s".format(t3 - t2a))


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


    assert network2.populations[0].size == neurons.size
    assert network2.stim_populations[0].size == stim.size
    assert network2.projections[0].size() == connections_exc.size(), "{} != {}".format(network2.projections[0].size(), connections_exc.size())
    assert network2.projections[1].size() == connections_inh.size(), "{} != {}".format(network2.projections[1].size(), connections_inh.size())
    assert network2.stim_projections[0].size() == connections_stim.size(), "{} != {}".format(network2.stim_projections[0].size(), connections_stim.size())

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