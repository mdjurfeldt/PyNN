"""
MUSIC support for the NEST backend
"""

import numpy
import nest
from pyNN.models import BaseCellType
from . import space, simulator
from .populations import Population
from .projections import Projection

music_support = True

def music_export(population, port_name):
    """
    """
    music_proxy = nest.Create("music_event_out_proxy",
                              params={"port_name": port_name})

    # We can't use PyNEST's ConvergentConnect here, as it does not
    # support a params dict for the connections at the moment. Once
    # that variant exists, we don't have to iterate here ourselves
    # anymore
    channel = 0
    for pre in population:
        conn_params = {"music_channel": channel}
        nest.Connect([pre], music_proxy, conn_params)
        channel += 1


class MusicProxyCellType(BaseCellType):
    nest_name = {"on_grid": "music_event_in_proxy",
                 "off_grid": "music_event_in_proxy"}
    
    def __init__(self, parameters):
        self.parameters = parameters


class MusicPopulation(Population):
    
    def _create_cells(self):
        nest_model = self.celltype.nest_name[simulator.state.spike_precision]
        params = self.celltype.parameters
        self.all_cells = nest.Create(nest_model, self.size, params=params)
        self._mask_local = numpy.array(nest.GetStatus(self.all_cells, 'local'))
        self.all_cells = numpy.array([simulator.ID(gid) for gid in self.all_cells], simulator.ID)
        for gid in self.all_cells:
            gid.parent = self

class MusicProjection(Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    def __init__(self, port, width, postsynaptic_population,
                 connector, synapse_type=None, source=None,
                 receptor_type=None, space=space.Space(), label=None):
        """
        port - MUSIC event input port name
        width - port width (= size of remote population)
        postsynaptic_population - Population object.

        All other arguments are as for the standard Projection class.
        """
        params = [{"port_name": port, "music_channel": c} for c in xrange(width)]
        pre_pop = MusicPopulation(width, MusicProxyCellType(params))
        Projection.__init__(self, pre_pop, postsynaptic_population,
                            connector, synapse_type, source=source,
                            receptor_type=receptor_type, space=space,
                            label=label)