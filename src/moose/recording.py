import numpy
import moose
from pyNN import recording
from . import simulator
from .simulator import mV

SCALE_FACTORS = {'v': 1/mV, 'gsyn_exc': 1, 'gsyn_inh': 1}
VARIABLE_MAP = {'v': 'Vm', 'gsyn_exc': 'esyn.Gk', 'gsyn_inh': 'isyn.Gk'}


class Recorder(recording.Recorder):
    _simulator = simulator

    def _record(self, variable, new_ids):
        moose_var = VARIABLE_MAP[variable]
        component = None
        if "." in moose_var:
            component, moose_var = moose_var.split('.')
        for id in new_ids:
            id._cell.tables[variable] = table = moose.Table(moose_var, id._cell)
            if component:
                source = getattr(id._cell, component)
            else:
                source = id._cell
            moose.connect(table, "requestData", source, "get_%s" % moose_var)
            moose.useClock(simulator.RECORDING_CLOCK, table.path, 'process')

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
