

import quantities as pq
import ninemlcatalog
import nineml.units as un

#from pype9.simulate.neuron import Cell as NineMLCell

from pyNN.utility import get_simulator
from pyNN.utility.plotting import Figure, Panel

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}))
exec("from pype9.simulate.{} import CellMetaClass, Simulation".format(options.simulator))
exec("from pype9.simulate.{}.network import PyNNCellWrapperMetaClass".format(options.simulator))


sim.setup(timestep=0.001)

model = ninemlcatalog.load('neuron/Izhikevich', 'Izhikevich')
properties = ninemlcatalog.load('neuron/Izhikevich',
                                'SampleIzhikevich')

#with Simulation(dt=0.1*un.ms) as sim:
celltype = PyNNCellWrapperMetaClass(component_class=model,
                                    default_properties=properties,
                                    initial_state=list(properties.initial_values),
                                    initial_regime=properties.initial_regime)

class Thing(object):
    t_start = 0.0

class MockSimulation(object):

    def active(self):
        return Thing()

celltype.model.Simulation = MockSimulation()

parameters = {
    'C_m': 0.001,
    'a': 0.02,
    'alpha': 0.04,
    'b': 0.2,
    'beta': 5.0,
    'c': -65.0,
    'd': 6.0,
    'theta': 30.0,
    'zeta': 140.0
}
init = {'V': -70.0}
init['U'] = parameters['b'] * init['V']

p = sim.Population(1, celltype(**parameters), initial_values=init)
p.record(['V', 'U'])

step = sim.DCSource(start=20.0, amplitude=0.014, stop=100.0)
step.inject_into(p)

sim.run(100.0)

data = p.get_data().segments[0]
vm = data.filter(name='V')[0]

Figure(
        Panel(vm, xticks=True, yticks=True,
              xlabel="Time (ms)", ylabel="Membrane potential (mV)",
              #ylim=(-96, -59)
              ),
        title="izh 9ml",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save("izh_9ml_{}.png".format(options.simulator))
