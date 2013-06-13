# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id:__init__.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Rev: $"

import music, sys, os, ctypes
if music.predictRank () >= 0:
    # launched by MPI
    os.environ['NEURON_INIT_MPI'] = '1'
    music.supersedeArgv (['music'] + sys.argv)

from pyNN.random import *
from pyNN import common, core, space, __doc__
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
                 label=None):
        __doc__ = common.Population.__doc__
        common.Population.__init__(self, size, cellclass, cellparams, structure, label)
        #simulator.initializer.register(self)

    def _create_cells(self, cellclass, cellparams, n):
        """
        Create cells in NEURON.
        
        `cellclass`  -- a PyNN standard cell or a native NEURON cell class that
                       implements an as-yet-undescribed interface.
        `cellparams` -- a dictionary of cell parameters.
        `n`          -- the number of cells to create
        """
        # this method should never be called more than once
        # perhaps should check for that
        assert n > 0, 'n must be a positive integer'
        celltype = cellclass(cellparams)
        #cell_model = celltype.model
        #cell_parameters = celltype.parameters
        #self.first_id = simulator.state.gid_counter
        self.first_id = 0
        #self.last_id = simulator.state.gid_counter + n - 1
        self.last_id = n - 1
        self.all_cells = numpy.array([id for id in range(self.first_id, self.last_id+1)], MusicID)
        # mask_local is used to extract those elements from arrays that apply to the cells on the current node
        #self._mask_local = self.all_cells%simulator.state.num_processes==simulator.state.mpi_rank # round-robin distribution of cells between nodes
        for i in self.all_cells:
            self.all_cells[i] = MusicID (i)
            self.all_cells[i].parent = self


class MusicConnection(simulator.Connection):
    """
    """
    def useSTDP(self, mechanism, parameters, ddf):
        raise NotImplementedError


class MusicProjection(Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    def __init__(self, port, width, postsynaptic_population,
                 method, source=None,
                 target=None, synapse_dynamics=None, label=None, rng=None):
        """
        port - MUSIC event input port name
        width - port width (= size of remote population)
        postsynaptic_population - Population object.

        source - string specifying which attribute of the presynaptic cell
                 signals action potentials

        target - string specifying which synapse on the postsynaptic cell to
                 connect to

        If source and/or target are not given, default values are used.

        method - a Connector object, encapsulating the algorithm to use for
                 connecting the neurons.

        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.

        rng - specify an RNG object to be used by the Connector.
        """
        print 1
        params = [{"port_name": port, "music_channel": c} for c in xrange(width)]
        print 2
        self.port = nmusic.publishEventInput (port)
        print 3
        ports.append (self.port)
        pre_pop = IndexPopulation(width, PortIndex)
        print 4
        Projection.__init__ (self, pre_pop, postsynaptic_population,
                             method, source=source,
                             target=target, synapse_dynamics=synapse_dynamics,
                             label=label, rng=rng)
        print 5

    def _divergent_connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.
        
        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        #if not isinstance(source, int) or source > simulator.state.gid_counter or source < 0:
            #errmsg = "Invalid source ID: %s (gid_counter=%d)" % (source, simulator.state.gid_counter)
            #raise errors.ConnectionError(errmsg)
        if not core.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise errors.ConnectionError("Invalid target ID: %s" % target)
              
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets), len(weights), len(delays))
        self._resolve_synapse_type()
        for target, weight, delay in zip(targets, weights, delays):
            if target.local:
                if "." in self.synapse_type: 
                    section, synapse_type = self.synapse_type.split(".") 
                    synapse_object = getattr(getattr(target._cell, section), synapse_type) 
                else: 
                    synapse_object = getattr(target._cell, self.synapse_type) 
                #nc = simulator.state.parallel_context.gid_connect(int(source), synapse_object)
                nc = self.port.index2target(source, synapse_object)
                nc.weight[0] = weight
                
                # if we have a mechanism (e.g. from 9ML) that includes multiple
                # synaptic channels, need to set nc.weight[1] here
                if nc.wcnt() > 1 and hasattr(target._cell, "type"):
                    nc.weight[1] = target._cell.type.synapse_types.index(self.synapse_type)
                nc.delay  = delay
                # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested
                self.connections.append(MusicConnection(source, target, nc))

    def _convergent_connect(self, sources, target, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.
        
        `sources`  -- a list/1D array of pre-synaptic cell IDs, or a single ID.
        `target` -- the ID of the post-synaptic cell.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        if not isinstance(target, int) or target > simulator.state.gid_counter or target < 0:
            errmsg = "Invalid target ID: %s (gid_counter=%d)" % (target, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)
        if not core.is_listlike(sources):
            sources = [sources]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(sources) > 0
        for source in sources:
            if not isinstance(source, common.IDMixin):
                raise errors.ConnectionError("Invalid source ID: %s" % source)
              
        assert len(sources) == len(weights) == len(delays), "%s %s %s" % (len(sources),len(weights),len(delays))
                
        if target.local:
            for source, weight, delay in zip(sources, weights, delays):
                if self.synapse_type is None:
                    self.synapse_type = weight >= 0 and 'excitatory' or 'inhibitory'
                if self.synapse_model == 'Tsodyks-Markram' and 'TM' not in self.synapse_type:
                    self.synapse_type += '_TM'
                synapse_object = getattr(target._cell, self.synapse_type)  
                #nc = simulator.state.parallel_context.gid_connect(int(source), synapse_object)
                nc = self.port.index2target(source, synapse_object)
                nc.weight[0] = weight
                nc.delay  = delay
                # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested
                self.connections.append(MusicConnection(source, target, nc))

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
