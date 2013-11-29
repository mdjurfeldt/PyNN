"""
A demonstration of using PyNN with MUSIC.

We create identical input populations in NEST and in NEURON, and then two
output populations in each simulator, one of which will receive connections from
the same simulator, one from the other, i.e.:

  NEST   --> NEST
  NEST   --> NEURON
  NEURON --> NEST
  NEURON --> NEURON

All four output populations should have identical activity.

"""

from socket import gethostname
from itertools import chain
from time import sleep
import numpy
from pyNN import music
from pyNN.utility import normalized_filename

w = 0.1
d = 0.5

# Instead of importing the backends directly, we import them via music.setup()
# which sets up the MPI communications.
nest, nrn = music.setup(music.Config("nest", 2), music.Config("neuron", 2))
local_sim = music.is_proxy(nest) and nrn or nest

nest.setup(timestep=0.1, min_delay=0.5)
nrn.setup(timestep=0.1, min_delay=0.5)

spike_times = [[10, 20, 30],
               [25, 45, 65]]
input = {
    "nest": nest.Population(2, nest.SpikeSourceArray(spike_times=spike_times), label="input (NEST)"),
    "neuron": nrn.Population(2, nrn.SpikeSourceArray(spike_times=spike_times), label="input (NEURON)")
}

output = {
    "nest-nest":     nest.Population(4, nest.IF_cond_exp(), label="NEST-NEST"),
    "neuron-nest":   nest.Population(4, nest.IF_cond_exp(), label="NEURON-NEST"),
    "nest-neuron":   nrn.Population(4, nrn.IF_cond_exp(), label="NEST-NEURON"),
    "neuron-neuron": nrn.Population(4, nrn.IF_cond_exp(), label="NEURON-NEURON"),
    #"nest-music-nest":   nest.Population(4, nest.IF_cond_exp(), label="NEST-MUSIC-NEST"),
    #"neuron-music-neuron": nrn.Population(4, nrn.IF_cond_exp(), label="NEURON-MUSIC-NEURON")
}

projections = [
    nest.Projection(input["nest"], output["nest-nest"], nest.AllToAllConnector(), nest.StaticSynapse(weight=w, delay=d)),
    music.Projection(input["neuron"], output["neuron-nest"], nest.AllToAllConnector(), nest.StaticSynapse(weight=w, delay=d)),
    music.Projection(input["nest"], output["nest-neuron"], nrn.AllToAllConnector(), nrn.StaticSynapse(weight=w, delay=d)),
    nrn.Projection(input["neuron"], output["neuron-neuron"], nrn.AllToAllConnector(), nrn.StaticSynapse(weight=w, delay=d)),
    #music.Projection(input["nest"], output["nest-music-nest"], nest.AllToAllConnector(), nest.StaticSynapse(weight=w, delay=d)),
    #music.Projection(input["neuron"], output["neuron-music-neuron"], nrn.AllToAllConnector(), nrn.StaticSynapse(weight=w, delay=d)),
]

for population in input.values():
    population.record('spikes')
for population in output.values():
    population.record(['spikes', 'v', 'gsyn_exc'])

music.run(100.0)

filenames = {}
for label, population in chain(input.items(), output.items()):
    filenames[label] = "Results/music_demo_%s_np%d.pkl" % (label,
                                                           local_sim.num_processes())
    population.write_data(filenames[label])

music.end()
