"""
Simple network with a Poisson spike source projecting to a pair of
IF_curr_alpha neurons.  Based on simpleNetwork.py.

Mikael Djurfeldt, PDC, KTH
March 2012

"""

from socket import gethostname
import numpy
from pyNN import music
from pyNN.utility import init_logging

#init_logging(None, debug=True)

sim2, sim1 = music.setup(music.Config("nest", 2), music.Config("neuron", 2))
local_sim = music.is_proxy(sim1) and sim2 or sim1

tstop = 1000.0
rate = 50.0

sim1.setup(timestep=0.1, min_delay=0.2, max_delay=1.0)
sim2.setup(timestep=0.1, min_delay=0.2, max_delay=1.0)

print "%s process %d of %d, running on %s." % (local_sim.__name__, local_sim.rank()+1, local_sim.num_processes(), gethostname())

cell_params = {'tau_refrac':2.0,'v_thresh':-50.0,'tau_syn_E':2.0, 'tau_syn_I':2.0}
output_population = sim2.Population(4, sim2.IF_curr_alpha, cell_params,
                                    label="output")

number = int(2*tstop*rate/1000.0)
numpy.random.seed(26278343)
spike_times = numpy.add.accumulate(numpy.random.exponential(1000.0/rate,
                                                            size=number))

input_population = sim1.Population(20, sim1.SpikeSourceArray,
                                   {'spike_times': spike_times}, label="input")

# The connector is used on the receiving side (sim2)
projection = music.Projection(input_population, output_population,
                              sim2.FixedProbabilityConnector(0.5),
                              sim2.StaticSynapse(weight=0.1))
#projection.set(weight=1.0)
#print projection.get('weight', format="array")

input_population.record('spikes')
output_population.record(['spikes', 'v'])

music.run(tstop)

input_population.write_data("Results/music_simple_input_%s.pkl" % sim1.name)
output_population.write_data("Results/music_simple_output_%s.pkl" % sim2.name)
#gather = True
#input_data = input_population.get_data(gather=gather)
#output_data = output_population.get_data(gather=gather)

# hangs with gather=True when writing output data if output population is run on multiple processes
# this is perhaps because of the mistaken assumption that MPI_ROOT is always 0?
# if we use gather=False, it seems to work.

n_spikes = output_population.mean_spike_count()
if not music.is_proxy(sim2) and sim2.rank() == 0:
    print "Firing rate: %s Hz" % (n_spikes*1000/tstop,)

music.end()
