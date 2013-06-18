"""
Example script to help explore what needs to be done to integrate MUSIC with PyNN

Script does not work at present.

"""

from pyNN import music
from pyNN.space import Grid2D

#vizapp = music.Config("vizapp", 1, "/path/to/vizapp", "args")

sim1, sim2 = music.setup(music.Config("neuron", 5),
                         music.Config("nest", 3),
                        )#vizapp)

sim1.setup(timestep=0.025)
sim2.setup(timestep=0.1)
cell_parameters = {"tau_m": 12.0, "cm": 0.8, "v_thresh": -50.0, "v_reset": -65.0}
pE = sim1.Population(100, sim1.IF_cond_exp, cell_parameters, label="excitatory neurons", structure=Grid2D())
pI = sim2.Population(36, sim2.IF_cond_exp, cell_parameters, label="inhibitory neurons", structure=Grid2D())
#all = pE + pI
def connector(sim):
    #return getattr(sim, "FixedProbabilityConnector")(0.1)
    #return getattr(sim, "FromListConnector")([(0, 0, 0.05, 0.5)])
    DDPC = getattr(sim, "DistanceDependentProbabilityConnector")
    return DDPC("exp(-d**2/400.0)")
def synapse_type(sim):
    return getattr(sim, "StaticSynapse")(weight=0.05, delay=0.5) #"0.5+0.01d")
e2e = sim1.Projection(pE, pE, connector(sim1), synapse_type(sim1), receptor_type="excitatory")
e2i = music.Projection(pE, pI, connector(sim2), synapse_type(sim2), receptor_type="excitatory")
i2i = sim2.Projection(pI, pI, connector(sim2), synapse_type(sim2), receptor_type="inhibitory")

#output = music.Port(pE, "spikes", vizapp, "in")

music.run(1000.0)

music.end()