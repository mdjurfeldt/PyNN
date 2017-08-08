"""
A demonstration of the use of the AdExp model, which allows an unlimited number
of different synapse models.


Usage: multi_synapse.py [-h] [--plot-figure] [--debug DEBUG] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file.
  --debug DEBUG  Print debugging information

"""


from pyNN.parameters import Sequence
from pyNN.utility import get_simulator, init_logging, normalized_filename


# === Configure the simulator ================================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=0.1, min_delay=1.0)


# === Build and instrument the network =======================================

celltype = sim.AdExp(tau_m=10.0,
                     v_rest=-60.0,
                     tau_syn={"AMPA": 1.0, "NMDA": 20.0, "GABAA": 1.5, "GABAB": 15.0},
                     e_syn={"AMPA": 0.0, "NMDA": 0.0, "GABAA": -70.0, "GABAB": -90.0}
                     )

neurons = sim.Population(1, celltype, initial_values={'v': -60.0})
neurons.record(['v'])  #, 'gsyn_AMPA', 'gsyn_NMDA', 'gsyn_GABAA', 'gsyn_GABAB'])

#import pdb; pdb.set_trace()

inputs = sim.Population(4,
                        sim.SpikeSourceArray(spike_times=[
                            Sequence([30.0]),
                            Sequence([60.0]),
                            Sequence([90.0]),
                            Sequence([120.0])])
                        )

connections = {
    "AMPA": sim.Projection(inputs[0:1], neurons, sim.OneToOneConnector(),
                           synapse_type=sim.StaticSynapse(weight=0.01, delay=1.5),
                           receptor_type="AMPA", label="AMPA"),
    "GABAA": sim.Projection(inputs[1:2], neurons, sim.OneToOneConnector(),
                            synapse_type=sim.StaticSynapse(weight=0.1, delay=1.5),
                            receptor_type="GABAA", label="GABAA"),
    "NMDA": sim.Projection(inputs[2:3], neurons, sim.OneToOneConnector(),
                           synapse_type=sim.StaticSynapse(weight=0.005, delay=1.5),
                           receptor_type="NMDA", label="NMDA"),
}

# === Run the simulation =====================================================

sim.run(200.0)


# === Save the results, optionally plot a figure =============================

#filename = normalized_filename("Results", "multi_synapse", "pkl", options.simulator)
filename = "Results/multi_synapse_{}.pkl".format(options.simulator)
data = neurons.get_data().segments[0]

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel(data.filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              xticks=True, xlabel="Time (ms)",
              yticks=True), #ylim=(-66, -48)),
        # Panel(data.filter(name='g_syn_AMPA')[0],
        #       xticks=True, xlabel="Time (ms)",
        #       ylabel="u (mV/ms)",
        #       yticks=True),
        title="Neuron with multiple synapse time constants",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()