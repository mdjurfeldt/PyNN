# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id:__init__.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Rev: $"

from itertools import repeat
import sys, os, ctypes
import music
if music.predictRank () >= 0:
    # launched by MPI
    os.environ['NEURON_INIT_MPI'] = '1'
    music.supersedeArgv (['music'] + sys.argv)

from pyNN.random import *
from pyNN import common, core, space, errors, __doc__
from pyNN.standardmodels import StandardCellType
from pyNN.recording import get_io
from pyNN.space import Space
from pyNN.neuron import simulator
from pyNN.neuron.standardmodels.cells import *
from pyNN.neuron.connectors import *
from pyNN.neuron.standardmodels.synapses import *
from pyNN.neuron.standardmodels.electrodes import *
from pyNN.neuron.populations import Population, PopulationView, Assembly
from pyNN.neuron.projections import Projection
from pyNN.neuron.cells import NativeCellType
from .recording import Recorder  # tmp music
import numpy

import logging
from neuron import h

import neuronmusic as nmusic
try:
    import neuronmusic as nmusic
    music_support = True
except ImportError:
    music_support = False


logger = logging.getLogger("PyNN")

# ==============================================================================
#   Utility functions
# ==============================================================================

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values() if isinstance(obj, type) and issubclass(obj, StandardCellType)]

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.

    `extra_params` contains any keyword arguments that are required by a given
    simulator but not by others.

    NEURON specific extra_params:

    use_cvode - use the NEURON cvode solver. Defaults to False.
      Optional cvode Parameters:
      -> rtol - specify relative error tolerance
      -> atol - specify absolute error tolerance

    native_rng_baseseed - added to MPI.rank to form seed for SpikeSourcePoisson, etc.
    default_maxstep - TODO

    returns: MPI rank

    """
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.initializer.clear()
    simulator.state.clear()
    simulator.state.dt = timestep
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    if extra_params.has_key('use_cvode'):
        simulator.state.cvode.active(int(extra_params['use_cvode']))
        if extra_params.has_key('rtol'):
            simulator.state.cvode.rtol(float(extra_params['rtol']))
        if extra_params.has_key('atol'):
            simulator.state.cvode.atol(float(extra_params['atol']))
    if extra_params.has_key('native_rng_baseseed'):
        simulator.state.native_rng_baseseed = int(extra_params['native_rng_baseseed'])
    if extra_params.has_key('default_maxstep'):
        simulator.state.default_maxstep=float(extra_params['default_maxstep'])
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []
    #simulator.state.finalize()
    music_end() # not necessary once simulator.finalize is implemented

run = common.build_run(simulator)

reset = common.build_reset(simulator)

initialize = common.initialize

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time, get_time_step, get_min_delay, get_max_delay, \
            num_processes, rank = common.build_state_queries(simulator)

# ==============================================================================
#   MUSIC support
# ==============================================================================

# Need to preserve ports so that they don't get deallocated
ports = []

def music_export(population, port_name):
    """
    """
    port = nmusic.publishEventOutput (port_name)
    ports.append (port)

    channel = 0
    for pre in population:
        port.gid2index (pre, channel)
        channel += 1


from pyNN.neuron.cells import NativeCellType


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
        self.nc = state.parallel_context.gid_connect(int(self.presynaptic_cell), target_object)
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
                 receptor_type=None, space=Space(), label=None):
        """
        port - MUSIC event input port name
        width - port width (= size of remote population)
        postsynaptic_population - Population object.

        All other arguments are as for the standard Projection class.
        """
        params = [{"port_name": port, "music_channel": c} for c in xrange(width)]
        self.port = nmusic.publishEventInput (port)
        ports.append (self.port)
        pre_pop = IndexPopulation(width, PortIndex())
        Projection.__init__(self, pre_pop, postsynaptic_population,
                            connector, synapse_type, source=source,
                            receptor_type=receptor_type, space=space,
                            label=label)

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


if music_support:
    #libnrnmpi = ctypes.CDLL ("/usr/local/nrn/x86_64/lib/libnrnmpi.so")
    libnrnmpi = ctypes.CDLL ("/home/andrew/env/music/x86_64/lib/libnrnmpi.so")

def music_end ():
    if music_support:
        libnrnmpi.nrnmpi_terminate ()


# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)

set = common.set

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)

# ==============================================================================
