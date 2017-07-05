# encoding: utf-8
"""



"""

from pprint import pprint
import numpy as np
from quantities import ms, kHz
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel
from pyNN.parameters import Sequence
from elephant.spike_train_generation import homogeneous_gamma_process
import nest

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

dt = 0.1
min_delay = 2.0
t_sim = 2000.0
np.random.seed(4293406)

sim.setup(timestep=dt,
          min_delay=min_delay,
          rng_seeds=[63256236])

def input_spikes(freq, intervals):
    segments = []
    for t_start, t_stop in intervals:
        segments.append(homogeneous_gamma_process(2, freq,
                                                  t_start=t_start,
                                                  t_stop=t_stop,
                                                  as_array=True))
    return Sequence(np.hstack(segments))

inputs = sim.Population(2,
                        sim.SpikeSourceArray(spike_times=[input_spikes(0.1*kHz, [(100*ms, 800*ms), (1100*ms, 1900*ms)]),
                                                          input_spikes(0.1*kHz, [(100*ms, 800*ms), (1100*ms, 1900*ms)])])
                        )

neuron_params = {
    "cm": 0.2,  # nF
    "tau_m": 0.2/0.01,  # nF/uS
    "tau_refrac": 4.0,
    "v_reset": -50.0,
    "v_rest": -65.0,
    "delta_v": 0.5,  # mV
    "v_t_star": -48.0,
    "lambda0": 1.0,  # 1/s
    "tau_eta": Sequence([10.0, 50.0, 250.0]),
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
cell_type = sim.RoessertEtAl(**neuron_params)
neurons = sim.Population(1, cell_type)

neurons.record(['v', 'i_syn'])

synapse_params = {
    "U": 1.0,
    "tau_facil": 17.0,  # ms
    "tau_rec": 671.0,  # ms
    "weight": 0.01,
    "delay": 2.0,  # ms
}
synapse_type = {
    "exc": sim.TsodyksMarkramSynapseEM(**synapse_params)
}
synapse_params["weight"] /= 2.0
synapse_type["inh"] = sim.TsodyksMarkramSynapseEM(**synapse_params)
connections = {
    "exc": sim.Projection(inputs, neurons, sim.FromListConnector([(0, 0)]),
                          synapse_type["exc"], receptor_type="exc"),
    "inh": sim.Projection(inputs, neurons, sim.FromListConnector([(1, 0)]),
                          synapse_type["inh"], receptor_type="inh")
}

pprint(nest.GetStatus(neurons.all_cells.tolist()))
pprint(nest.GetStatus(nest.GetConnections(source=inputs.all_cells.tolist())))
sim.run(t_sim)

file_stem = "Results/roessert_neuron_pyNN.{}".format("nest")
if options.plot_figure:
    data = neurons.get_data().segments[0]

    figure_filename = file_stem + ".png"
    Figure(
            Panel(data.filter(name='v')[0],
                  ylabel="Membrane potential (mV)",
                  xticks=True, xlabel="Time (ms)",
                  yticks=True), #ylim=(-66, -48)),
            Panel(data.filter(name='i_syn')[0],
                  xticks=True, xlabel="Time (ms)",
                  ylabel="Total synaptic current (nA)",
                  yticks=True),
            title="Rössert et al. neuron with multiple synapse time constants",
            annotations="Simulated with %s via PyNN" % options.simulator.upper()
        ).save(figure_filename)
else:
    neurons.write_data(file_stem + ".pkl")

sim.end()