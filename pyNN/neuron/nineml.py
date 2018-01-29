"""
Support cell types defined in 9ML with NEURON.

Requires the 9ml2nmodl script to be on the path.

Classes:
    NineMLCell       - a single neuron instance
    NineMLCellType   - base class for cell types, not used directly

Functions:
    nineml_cell_type - return a new NineMLCellType subclass

Constants:
    NMODL_DIR        - subdirectory to which NMODL mechanisms will be written

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from __future__ import absolute_import
import logging
import os
from datetime import datetime
import neuron
from pyNN.models import BaseCellType, BaseSynapseType
from pyNN.neuron.simulator import Connection, load_mechanisms
#from pyNN.nineml.cells import build_nineml_celltype


h = neuron.h
logger = logging.getLogger("PyNN")

NMODL_DIR = "nineml_mechanisms"


class NineMLCell(object):

    def __init__(self, **parameters):
        self.type = parameters.pop("type")
        self.source_section = h.Section()
        self.source = getattr(h, self.type.model_name)(0.5, sec=self.source_section)
        for param, value in parameters.items():
            setattr(self.source, param, value)
        # for recording
        self.rec = h.NetCon(self.source, None)
        self.spike_times = h.Vector(0)
        self.traces = {}
        self.recording_time = False

    def __getattr__(self, name):
        try:
            return self.__getattribute__(name)
        except AttributeError:
            if name in self.type.receptor_types:
                return self.source  # source is also target
            else:
                raise AttributeError("'NineMLCell' object has no attribute or receptor type '%s'" % name)

    def memb_init(self):
        # this is a bit of a hack
        for var in self.type.recordable:
            if hasattr(self, "%s_init" % var):
                initial_value = getattr(self, "%s_init" % var)
                logger.debug("Initialising %s to %g" % (var, initial_value))
                setattr(self.source, var, initial_value)


class NineMLCellType(BaseCellType):
    model = NineMLCell

    def __init__(self, **parameters):
        BaseCellType.__init__(self, **parameters)
        self.extra_parameters = {"type": self}  # self.__class__?


class build_nineml_celltype(type):
    """
    Metaclass for building NineMLCellType subclasses
    Called by nineml_celltype_from_model
    """
    
    def __new__(cls, name, bases, dct):

        import nineml.abstraction as al
        from nineml.abstraction.dynamics.utils import (
            flattener, xml, modifiers)

        # Extract Parameters Back out from Dict:
        combined_model = dct['combined_model']
        weight_vars = dct['weight_variables']

        # Flatten the model:
        assert isinstance(combined_model, al.ComponentClass)
        if combined_model.is_flat():
            flat_component = combined_model
        else:
            flat_component = flattener.flatten(combined_model, name)

        # Make the substitutions:
        flat_component.backsub_all()
        #flat_component.backsub_aliases()
        #flat_component.backsub_equations()

        # Close any open reduce ports:
        modifiers.DynamicPortModifier.close_all_reduce_ports(componentclass=flat_component)

        # New:
        dct["combined_model"] = flat_component
        dct["default_parameters"] = dict((param.name, 1.0) for param in flat_component.parameters)
        dct["default_initial_values"] = dict((statevar.name, 0.0) for statevar in chain(flat_component.state_variables))
        dct["receptor_types"] = list(weight_vars.keys())
        dct["standard_receptor_type"] = (dct["receptor_types"] == ('excitatory', 'inhibitory'))
        dct["injectable"] = False        # how to determine this? neuron component has a receive analog port with dimension current, that is not connected to a synapse port?
        dct["conductance_based"] = True  # how to determine this? synapse component has a receive analog port with dimension voltage?
        dct["model_name"] = name
        dct["units"] = dict((statevar.name, _default_units[statevar.dimension.name]) for statevar in chain(flat_component.state_variables))

        # Recording from bindings:
        dct["recordable"] = ([port.name for port in flat_component.analog_ports]
                             + ['spikes', 'regime']
                             + [alias.lhs for alias in flat_component.aliases]
                             + [statevar.name for statevar in flat_component.state_variables])

        logger.debug("Creating class '%s' with bases %s and dictionary %s" % (name, bases, dct))
        dct["builder"](flat_component, dct["weight_variables"], hierarchical_mode=True)

        return type.__new__(cls, name, bases, dct)


def nineml_cell_type(name, combined_model, weight_vars):
    """
    Return a new NineMLCellType subclass.
    """
    return build_nineml_celltype(name, (NineMLCellType,),
                                 {'combined_model': combined_model,
                                  'weight_variables': weight_vars})


class NineMLSynapseType(BaseSynapseType):

    connection_type = Connection
    presynaptic_type = None
    #postsynaptic_variable = "spikes"

    @property
    def native_parameters(self):
        return self.parameter_space

    def get_native_names(self, *names):
        return names

    def _get_minimum_delay(self):
        return state.min_delay


def nineml_synapse_type(name, nineml_model):
    """

    """
    return build_nineml_synapse_type(name, (NineMLSynapseType,),
                                     {'nineml_model': nineml_model})


class build_nineml_synapse_type(type):
    """
    Metaclass for building NineMLSynapseType subclasses.
    """

    def __new__(cls, name, bases, dct):

        import pype9
        from pype9.simulate.neuron import CodeGenerator
        from pype9.simulate.neuron.units import UnitHandler

        builder = CodeGenerator()

        # Extract attributes from dct:
        model = dct['nineml_model']
        
        # Calculate attributes of the new class
        dct["default_parameters"] = {
            "weight": 0.0, "delay": None,
            "dendritic_delay_fraction": 1.0
        }
        dct["default_parameters"].update((param.name, 1.0) for param in model.parameters)
        dct["default_initial_values"] = dict((statevar.name, 0.0) for statevar in model.state_variables)
        dct["postsynaptic_variable"] = "spikes"
        dct["model"] = name

        logger.debug("Creating class '%s' with bases %s and dictionary %s" % (name, bases, dct))

        # Now generate and compile the NMODL
        all_triggers = []
        for regime in model.regimes:
            for on_condition in regime.on_conditions:
                if on_condition.trigger.rhs not in all_triggers:
                    all_triggers.append(on_condition.trigger.rhs)

        context = {
            'component_class': model,
            'all_triggers': all_triggers,
            'component_name': model.name,
            'version': pype9.__version__, 
            'timestamp': datetime.now().strftime('%a %d %b %y %I:%M:%S%p'),
            'unit_handler': UnitHandler(model),
            'regime_varname': builder.REGIME_VARNAME,
            'code_gen': builder,
        }

        if not os.path.exists(NMODL_DIR):
            os.makedirs(NMODL_DIR)
        builder.render_to_file("weight_adapter.tmpl", 
                               context, 
                               model.name.lower() + '.mod',
                               NMODL_DIR)
        orig_dir = os.getcwd()
        builder.compile_source_files(NMODL_DIR, model.name)
        os.chdir(orig_dir)
        load_mechanisms(NMODL_DIR)

        return type.__new__(cls, name, bases, dct)
