# encoding: utf-8
"""



"""

from pprint import pprint
import numpy as np
import neo
from quantities import mV, pA, kHz, ms
import nest
from elephant.spike_train_generation import homogeneous_gamma_process
from pyNN.utility.plotting import Figure, Panel


dt = 0.1
min_delay = 2.0
t_sim = 2000.0
np.random.seed(4293406)

nest.Install('rossert_et_almodule')

nest.SetKernelStatus({"rng_seeds": [63256236],
                      "overwrite_files": True,
                      "data_path": "Results"})

nest.SetDefaults('spike_generator', {'precise_times': True})
inputs = nest.Create("spike_generator", n=2)

def input_spikes(freq, intervals):
    segments = []
    for t_start, t_stop in intervals:
        segments.append(homogeneous_gamma_process(2, freq,
                                                  t_start=t_start,
                                                  t_stop=t_stop,
                                                  as_array=True))
    return np.hstack(segments)


nest.SetStatus(inputs,
               [{"spike_times": input_spikes(0.1*kHz, [(100*ms, 800*ms), (1100*ms, 1900*ms)])},
                {"spike_times": input_spikes(0.1*kHz, [(100*ms, 800*ms), (1100*ms, 1900*ms)])}
                ])
sources = nest.Create("parrot_neuron", n=len(inputs))
nest.Connect(inputs, sources, "one_to_one")

neuron_params = {
    "C_m": 200.0,  # pF
    "g_L": 10.0,  # nS
    "tau_stc": [10.0, 50.0, 250.0],  # ms
    "q_stc": [200.0, 50.0, 25.0],  # pA
    "tau_sfa": [5.0, 200.0, 250.0],  # ms
    "q_sfa": [15.0, 3.0, 1.0],  # mV
    "t_ref": 4.0,  # ms
    "V_reset": -50.,  # -55. mV
    "E_L": -65.0,  # -70. mV
    "V_m": -65.0,
    "Delta_V": 0.5,  # mV
    "lambda_0": 1.,  # 1/s
    "V_T_star": -48.,  # -35. mV
    "I_e": 0.0,  # pA
    "g_max": 1.0,  # uS
    "tau_r_fast": [0.2, 0.2],  # ms
    "tau_d_fast": [1.7, 8.0],  # ms
    "tau_r_slow": [0.29, 3.5],  # ms
    "tau_d_slow": [43.0, 260.9],  # ms
    "E_rev": [0.0, -80.0],  # mV
    "E_rev_B": [0.0, -97.0],  # mV
    "ratio_slow": [0.5, 0.0],
    "mg": [1.0, 0.0],  # mM
    "tau_corr": [5.0, 5.0],
}

neurons = nest.Create("rossert_et_al", n=1)
nest.SetStatus(neurons, neuron_params)

recorder = nest.Create("multimeter", n=1)
nest.SetStatus(recorder,
               {
                   "interval": dt,
                   "record_from": ["V_m", "I_syn"],
                   "to_file": True,
                   "label": "roessert_neuron_nest"
               })
nest.Connect(recorder, neurons, "all_to_all",
             {"model": "static_synapse", "delay": min_delay})


synapse_params = {
    "U": 1.0,
    "tau_fac": 17.0,  # ms
    "tau_rec": 671.0,  # ms
    "weight": 10.0,
    "delay": 2.0,  # ms
    "receptor_type": 1  # 1=exc 2=inh
}

nest.CopyModel("MarkramConnection", "MarkramConnection_exc", synapse_params)
synapse_params["receptor_type"] = 2
synapse_params["weight"] /= 2
nest.CopyModel("MarkramConnection", "MarkramConnection_inh", synapse_params)
nest.Connect(sources[0:1], neurons,
             conn_spec="all_to_all",
             syn_spec={"model": "MarkramConnection_exc"})
nest.Connect(sources[1:2], neurons,
             conn_spec="all_to_all",
             syn_spec={"model": "MarkramConnection_inh"})

pprint(nest.GetStatus(neurons))
pprint(nest.GetStatus(nest.GetConnections(source=sources)))

nest.Simulate(t_sim)

filename = nest.GetStatus(recorder, "filenames")[0][0]
print(filename)

io = neo.io.NestIO(filename)
data = io.read(gid_list=[], value_columns_dat=[2, 3], value_units=[mV, pA])[0].segments[0]
vm = data.analogsignals[0]
i_syn = data.analogsignals[1]
vm.channel_index = neo.ChannelIndex(index=np.arange(1))
i_syn.channel_index = neo.ChannelIndex(index=np.arange(1))

figure_filename = "Results/roessert_neuron_{}.png".format("nest")
Figure(
        Panel(vm,
              ylabel="Membrane potential (mV)",
              xticks=True, xlabel="Time (ms)",
              yticks=True), #ylim=(-66, -48)),
        Panel(i_syn,
              xticks=True, xlabel="Time (ms)",
              ylabel="Summed synaptic current (pA)",
              yticks=True),
        title="Rössert et al. neuron with multiple synapse time constants",
        annotations="Simulated directly with NEST"
    ).save(figure_filename)
