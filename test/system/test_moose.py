from nose.plugins.skip import SkipTest
from scenarios import scenarios
from nose.tools import assert_equal, assert_almost_equal
from pyNN.random import RandomDistribution
from pyNN.utility import init_logging

try:
    import pyNN.moose
    have_moose = True
except ImportError:
    have_moose = False


def test_scenarios():
    for scenario in scenarios:
        if "moose" not in scenario.exclude:
            scenario.description = scenario.__name__
            if have_moose:
                yield scenario, pyNN.moose
            else:
                raise SkipTest


def test_recording():
    if not have_moose:
        raise SkipTest

    sim = pyNN.moose
    sim.setup()

    p = sim.Population(2, sim.HH_cond_exp, {'i_offset': 0.1})
    p.initialize(v=-65.0)
    p.record('v')

    sim.run(100.0)

    data = p.get_data()
    # assert something here
    sim.end()

    return data


def test_synaptic_connections():
    if not have_moose:
        raise SkipTest

    import numpy
    sim = pyNN.moose
    sim.setup()

    p1 = sim.Population(1, sim.SpikeSourceArray, {'spike_times': numpy.arange(3.0, 103, 10.0)})
    #p1 = sim.Population(1, sim.SpikeSourcePoisson, {'rate': 100.0})
    #p1 = sim.Population(1, sim.HH_cond_exp, {'i_offset': 1.0})
    #p2 = sim.Population(1, sim.HH_cond_exp)
    p2 = sim.Population(2, sim.IF_cond_exp(tau_refrac=0.0, v_rest=-61.2), initial_values={'v': -61.2})

    connector = sim.FromListConnector([(0, 1, 1e-4, 0.5)])
    prj = sim.Projection(p1, p2, connector,
                         sim.StaticSynapse(weight=1e-4))

    p2.record(['v', 'gsyn_exc'])

    sim.run(100.0)

    data = p2.get_data()
    sim.end()

    vm = data.segments[0].filter(name='v')[0]
    assert_equal(vm.shape, (1001, 2))
    vm0 = vm[:, numpy.where(vm.channel_index==0)[0][0]]
    vm1 = vm[:, numpy.where(vm.channel_index==1)[0][0]]
    assert_equal(vm0.max(), -61.2)  # no synaptic input 
    assert vm1.max() > -61.2        # receives synaptic input
    return data