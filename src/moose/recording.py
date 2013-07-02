"""
MOOSE implementation of the PyNN API

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import moose
from pyNN import recording
from . import simulator
from .simulator import mV, uS

SCALE_FACTORS = {'v': 1/mV, 'gsyn_exc': 1/uS, 'gsyn_inh': 1/uS}
VARIABLE_MAP = {'v': 'Vm', 'gsyn_exc': 'esyn.Gk', 'gsyn_inh': 'isyn.Gk'}


class Recorder(recording.Recorder):
    _simulator = simulator

    def _record(self, variable, new_ids):
        moose_var = VARIABLE_MAP[variable]
        component = None
        if "." in moose_var:
            component, moose_var = moose_var.split('.')
        for id in new_ids:
            id._cell.tables[variable] = table = moose.Table("%s/%s" % (id._cell.path,
                                                                       variable))
            if component:
                source = getattr(id._cell, component)
            else:
                source = id._cell
            moose.connect(table, "requestData", source, "get_%s" % moose_var)
            moose.useClock(simulator.RECORDING_TICK, table.path, 'process')
            print "recording %s from %s to table %s" % (variable, id, table.path)

    def _get_all_signals(self, variable, ids, clear=False):
        scale_factor = SCALE_FACTORS[variable]
        if len(ids) > 0:
            return scale_factor * numpy.vstack((id._cell.tables[variable].vec for id in ids)).T
        else:
            return numpy.array([])
            
    @staticmethod
    def find_units(variable):
        if variable in recording.UNITS_MAP:
            return recording.UNITS_MAP[variable]
        else:
            raise Exception("units unknown")
