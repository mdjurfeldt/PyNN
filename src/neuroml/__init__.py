"""
Conversion between PyNN and NeuroML

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from pyNN import common
from pyNN.connectors import *
from . import simulator
from .standardmodels import *
from .populations import Population, PopulationView, Assembly
from .projections import Projection


logger = logging.getLogger("PyNN")


def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values() if isinstance(obj, type) and issubclass(obj, StandardCellType)]


def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    if 'rank' in extra_params:
        simulator.state.mpi_rank = extra_params['rank']
    if 'num_processes' in extra_params:
        simulator.state.num_processes = extra_params['num_processes']
    simulator.state.xmlfile = extra_params.get('file', "PyNN2NeuroML.nml")
    return rank()


def end():
    """Write XML to file."""
    simulator.state.export()


def run(simtime):
    """
    This does nothing, but the information that run() was called should be
    stored for outputting SED-ML."""
    pass

reset = common.build_reset(simulator)

initialize = common.initialize

get_current_time, get_time_step, get_min_delay, get_max_delay, \
                    num_processes, rank = common.build_state_queries(simulator)

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)

set = common.set

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)
