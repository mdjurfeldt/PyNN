"""
MUSIC support for the NEURON backend
"""

import os
import os.path as osp
import ctypes
import numpy
from itertools import repeat
try:
    import neuronmusic as nmusic
    music_support = True
except ImportError:
    music_support = False
from .. import common, space, core
from . import simulator
from .populations import Population, Assembly
from .projections import Projection
from .recording import Recorder
from .cells import NativeCellType


# Need to preserve ports so that they don't get deallocated
ports = []

def music_export(population, port_name):
    """
    """
    port = nmusic.publishEventOutput (port_name)
    ports.append (port)

    channel = 0
    for pre in population.all():
        port.gid2index (pre, channel)
        channel += 1


class PortIndex(NativeCellType):
    """
    """
    always_local = True


class MusicID(int, common.IDMixin):
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


class IndexPopulation(Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    _simulator = simulator
    recorder_class = Recorder
    assembly_class = Assembly
    
    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        __doc__ = common.Population.__doc__
        common.Population.__init__(self, size, cellclass, cellparams, structure,
                                   initial_values, label)
        #simulator.initializer.register(self)

    def _create_cells(self):
        """
        Create cells in NEURON.
        
        `cellclass`  -- a PyNN standard cell or a native NEURON cell class that
                       implements an as-yet-undescribed interface.
        `cellparams` -- a dictionary of cell parameters.
        `n`          -- the number of cells to create
        """
        # this method should never be called more than once
        # perhaps should check for that
        self.first_id = 0
        self.last_id = self.size - 1
        self.all_cells = numpy.array([id for id in range(self.first_id, self.last_id+1)], MusicID)
        for i in self.all_cells:
            self.all_cells[i] = MusicID (i)
            self.all_cells[i].parent = self


class MusicConnection(simulator.Connection):
    """
    """
    
    def __init__(self, projection, pre, post, **parameters):
        """
        Create a new connection.
        """
        #logger.debug("Creating connection from %d to %d, weight %g" % (pre, post, parameters['weight']))
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.presynaptic_cell = projection.pre[pre]
        self.postsynaptic_cell = projection.post[post]
        if "." in projection.receptor_type:
            section, target = projection.receptor_type.split(".")
            target_object = getattr(getattr(self.postsynaptic_cell._cell, section), target)
        else:
            target_object = getattr(self.postsynaptic_cell._cell, projection.receptor_type)
        self.nc = projection.port.index2target(self.presynaptic_index, target_object)
        self.nc.weight[0] = parameters.pop('weight')
        # if we have a mechanism (e.g. from 9ML) that includes multiple
        # synaptic channels, need to set nc.weight[1] here
        if self.nc.wcnt() > 1 and hasattr(self.postsynaptic_cell._cell, "type"):
            self.nc.weight[1] = self.postsynaptic_cell._cell.type.receptor_types.index(projection.receptor_type)
        self.nc.delay  = parameters.pop('delay')
        if projection.synapse_type.model is not None:
            self._setup_plasticity(projection.synapse_type, parameters)
        # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested
    
    def _setup_plasticity(self, synapse_type, parameters):
        raise NotImplementedError


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
        self.port = nmusic.publishEventInput (port)
        ports.append (self.port)
        pre_pop = IndexPopulation(width, PortIndex())
        Projection.__init__(self, pre_pop, postsynaptic_population,
                            connector, synapse_type, source=source,
                            receptor_type=receptor_type, space=space,
                            label=label)
        #print ", ".join("<%d>: %d" % (id, len(self._connections[id])) for id in self._connections)

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `presynaptic_cells`     -- a 1D array of pre-synaptic cell IDs
        `postsynaptic_cell`     -- the ID of the post-synaptic cell.
        `connection_parameters` -- each parameter should be either a
                                   1D array of the same length as `sources`, or
                                   a single value.
        """
        #logger.debug("Convergent connect. Weights=%s" % connection_parameters['weight'])
        postsynaptic_cell = self.post[postsynaptic_index]
        if not isinstance(postsynaptic_cell, int) or postsynaptic_cell > simulator.state.gid_counter or postsynaptic_cell < 0:
            errmsg = "Invalid post-synaptic cell: %s (gid_counter=%d)" % (postsynaptic_cell, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)
        for name, value in connection_parameters.items():
            if isinstance(value, (float, int)):
                connection_parameters[name] = repeat(value)
        assert postsynaptic_cell.local
        for pre_idx, values in core.ezip(presynaptic_indices, *connection_parameters.values()):
            parameters = dict(zip(connection_parameters.keys(), values))
            #logger.debug("Connecting neuron #%s to neuron #%s with synapse type %s, receptor type %s, parameters %s", pre_idx, postsynaptic_index, self.synapse_type, self.receptor_type, parameters)
            self._connections[postsynaptic_index][pre_idx] = \
                MusicConnection(self, pre_idx, postsynaptic_index, **parameters)


def find_libnrnmpi():
    found = False
    for path in os.getenv('PATH').split(osp.pathsep):
        if osp.exists(osp.join(path, "nrnivmodl")):
            found = path
            break
    if found:
        real_path = osp.split(osp.realpath(osp.join(path, "nrnivmodl")))[0]
        candidate_lib_path = osp.join(real_path, osp.pardir, "lib", "libnrnmpi.so")
        if osp.exists:
            return ctypes.CDLL(candidate_lib_path)
        else:
            raise Exception("Couldn't find libnrnmpi.so")
    else:
        raise Exception("Couldn't find nrnivmodl")


if music_support:
    libnrnmpi = find_libnrnmpi()


def music_end ():
    if music_support:
        libnrnmpi.nrnmpi_terminate ()
