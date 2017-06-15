
import numpy as np
from quantities import mV, pA
import neo
from pyNN.utility.plotting import comparison_plot


filenames = {
    "nest": "Results/roessert_neuron_nest-6-0.dat",
    "pyNN.nest": "Results/roessert_neuron_pyNN.nest.pkl"
}

io = neo.io.NestIO(filenames["nest"])
data_nest = io.read(gid_list=[], value_columns_dat=[2, 3], value_units=[mV, pA])[0].segments[0]
for signal, var_name in zip(data_nest.analogsignals, ("v", "i_syn")):
    signal.channel_index = neo.ChannelIndex(index=np.arange(1))
    signal.name = var_name

io = neo.get_io("Results/roessert_neuron_pyNN.nest.pkl")
data_pyNN = io.read()[0].segments[0]
for signal in data_pyNN.analogsignals:
    signal.channel_index = neo.ChannelIndex(index=np.arange(1))

comparison_plot([data_nest, data_pyNN],
                labels=("NEST", "pyNN.nest"),
                title="Rössert et al. neuron comparison"
                ).save("Results/roessert_neuron_comparison.png")
