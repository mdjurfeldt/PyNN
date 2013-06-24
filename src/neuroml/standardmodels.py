# encoding: utf-8
"""
Standard cells for the NeuroML module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from copy import deepcopy
import neuroml
from pyNN.standardmodels import cells, synapses, electrodes, build_translations, StandardCurrentSource
from .simulator import state
import logging

logger = logging.getLogger("PyNN")


def get_units(parameter_name):
    units_map = {
        "tau": "ms",
        "cm": "nF",
        "thresh": "mV",
        "reversal": "mV",
        "reset": "mV",
        "i_": "nA",
        "erev": "mV",
        "start": "ms",
        "rate": "per_s",
        "duration": "ms",
    }
    for fragment, units in units_map.items():
        if fragment in parameter_name:
            return units
    raise ValueError("Units for parameter %s cannot be determined" % parameter_name)


class NeuroMLCellTypeMixin(object):

    @property
    def cell_parameters(self):
        if not self.parameter_space.is_homogeneous:
            raise Exception("NeuroML backend does not yet support heterogeneous parameters.")
        parameters = self.native_parameters
        for name in ("tau_syn_E", "tau_syn_I", "e_rev_E", "e_rev_I"):
            if self.has_parameter(name):
                parameters.pop(self.get_native_names(name)[0])
        for name in parameters.keys():
            if "not_supported" in name:
                parameters.pop(name)
        parameters.shape = (1,)  # can do this, since homogeneous
        parameters.evaluate(simplify=True)
        parameters = parameters.as_dict()
        for name, value in parameters.items():
            parameters[name] = "%g %s" % (value, get_units(name))
        return parameters

    def receptor_parameters(self, receptor_type):
        parameters = self.native_parameters
        #for name in self.get_native_names("tau_m", "cm", "v_rest", "v_thresh", "tau_refrac", "i_offset"):
        #    parameters.pop(name)
        parameters.shape = (1,)  # can do this, since homogeneous
        parameters.evaluate(simplify=True)
        recep_parameters = {}
        for name, value in parameters.items():
            if receptor_type in name:
                stripped_name = name[len(receptor_type)+1:]
                recep_parameters[stripped_name] = "%g %s" % (value, get_units(name))
        return recep_parameters


#class IF_curr_alpha(cells.IF_curr_alpha):
#    __doc__ = cells.IF_curr_alpha.__doc__
#
#    translations = build_translations(  # should add some computed/scaled parameters
#        ('tau_m',      'TAU_M'),
#        ('cm',         'CM'),
#        ('v_rest',     'V_REST'),
#        ('v_thresh',   'V_THRESH'),
#        ('v_reset',    'V_RESET'),
#        ('tau_refrac', 'TAU_REFRAC'),
#        ('i_offset',   'I_OFFSET'),
#        ('tau_syn_E',  'TAU_SYN_E'),
#        ('tau_syn_I',  'TAU_SYN_I'),
#    )
#
#
#class IF_curr_exp(cells.IF_curr_exp):
#    __doc__ = cells.IF_curr_exp.__doc__
#
#    translations = build_translations(  # should add some computed/scaled parameters
#        ('tau_m',      'TAU_M'),
#        ('cm',         'CM'),
#        ('v_rest',     'V_REST'),
#        ('v_thresh',   'V_THRESH'),
#        ('v_reset',    'V_RESET'),
#        ('tau_refrac', 'T_REFRAC'),
#        ('i_offset',   'I_OFFSET'),
#        ('tau_syn_E',  'TAU_SYN_E'),
#        ('tau_syn_I',  'TAU_SYN_I'),
#    )
#
#
#class IF_cond_alpha(cells.IF_cond_alpha):
#    __doc__ = cells.IF_cond_alpha.__doc__
#
#    translations = build_translations(
#        ('tau_m',      'TAU_M'),
#        ('cm',         'CM'),
#        ('v_rest',     'V_REST'),
#        ('v_thresh',   'V_THRESH'),
#        ('v_reset',    'V_RESET'),
#        ('tau_refrac', 'TAU_REFRAC'),
#        ('i_offset',   'I_OFFSET'),
#        ('tau_syn_E',  'TAU_SYN_E'),
#        ('tau_syn_I',  'TAU_SYN_I'),
#        ('e_rev_E',    'E_REV_E'),
#        ('e_rev_I',    'E_REV_I')
#    )


class IF_cond_exp(cells.IF_cond_exp, NeuroMLCellTypeMixin):
    __doc__ = cells.IF_cond_exp.__doc__
    n = 0

    translations = build_translations(
        ('tau_m',      'tau'),
        ('cm',         'cm_not_supported'),
        ('v_rest',     'leak_reversal'),
        ('v_thresh',   'thresh'),
        ('v_reset',    'reset'),
        ('tau_refrac', 'tau_refrac_not_supported'),
        ('i_offset',   'i_offset_not_supported'),
        ('tau_syn_E',  'excitatory_tau_decay'),
        ('tau_syn_I',  'inhibitory_tau_decay'),
        ('e_rev_E',    'excitatory_erev'),
        ('e_rev_I',    'inhibitory_erev')
    )
    neuroml_cell_component = neuroml.IaFTauCell
    neuroml_receptor_component = neuroml.ExpOneSynapse  # should perhaps be a dict in the general case
    neuroml_cell_list = "iaf_tau_cells"

    def __init__(self, **parameters):
        super(self.__class__, self).__init__(**parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.__class__.n += 1


#class HH_cond_exp(cells.HH_cond_exp):
#    __doc__ = cells.HH_cond_exp.__doc__
#
#    translations = build_translations(
#        ('gbar_Na',    'GBAR_NA'),
#        ('gbar_K',     'GBAR_K'),
#        ('g_leak',     'G_LEAK'),
#        ('cm',         'CM'),
#        ('v_offset',   'V_OFFSET'),
#        ('e_rev_Na',   'E_REV_NA'),
#        ('e_rev_K',    'E_REV_K'),
#        ('e_rev_leak', 'E_REV_LEAK'),
#        ('e_rev_E',    'E_REV_E'),
#        ('e_rev_I',    'E_REV_I'),
#        ('tau_syn_E',  'TAU_SYN_E'),
#        ('tau_syn_I',  'TAU_SYN_I'),
#        ('i_offset',   'I_OFFSET'),
#    )


class SpikeSourcePoisson(cells.SpikeSourcePoisson, NeuroMLCellTypeMixin):
    __doc__ = cells.SpikeSourcePoisson.__doc__
    neuroml_cell_component = neuroml.SpikeGeneratorPoisson
    neuroml_cell_list = "spike_generator_poissons"

    n = 0
    
    translations = build_translations(
        ('start',    'not_supported'),
        ('rate',     'average_rate',),
        ('duration', 'not_supported'),
    )

    def __init__(self, **parameters):
        super(self.__class__, self).__init__(**parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.__class__.n += 1


#class SpikeSourceArray(cells.SpikeSourceArray):
#    __doc__ = cells.SpikeSourceArray.__doc__
#    #neuroml_cell_component = neuroml.SpikeArray
#
#    translations = build_translations(
#        ('spike_times', 'spikes'),
#    )
#
#    def __init__(self, **parameters):
#        super(self.__class__, self).__init__(**parameters)
#        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
#        self.__class__.n += 1


#class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
#    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__
#
#    translations = build_translations(
#        ('cm',         'CM'),
#        ('tau_refrac', 'TAU_REFRAC'),
#        ('v_spike',    'V_SPIKE'),
#        ('v_reset',    'V_RESET'),
#        ('v_rest',     'V_REST'),
#        ('tau_m',      'TAU_M'),
#        ('i_offset',   'I_OFFSET'),
#        ('a',          'A'),
#        ('b',          'B'),
#        ('delta_T',    'DELTA_T'),
#        ('tau_w',      'TAU_W'),
#        ('v_thresh',   'V_THRESH'),
#        ('e_rev_E',    'E_REV_E'),
#        ('tau_syn_E',  'TAU_SYN_E'),
#        ('e_rev_I',    'E_REV_I'),
#        ('tau_syn_I',  'TAU_SYN_I'),
#    )
#
#
#class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
#    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__
#
#    translations = build_translations(
#        ('cm',         'CM'),
#        ('tau_refrac', 'TAU_REFRAC'),
#        ('v_spike',    'V_SPIKE'),
#        ('v_reset',    'V_RESET'),
#        ('v_rest',     'V_REST'),
#        ('tau_m',      'TAU_M'),
#        ('i_offset',   'I_OFFSET'),
#        ('a',          'A'),
#        ('b',          'B'),
#        ('delta_T',    'DELTA_T'),
#        ('tau_w',      'TAU_W'),
#        ('v_thresh',   'V_THRESH'),
#        ('e_rev_E',    'E_REV_E'),
#        ('tau_syn_E',  'TAU_SYN_E'),
#        ('e_rev_I',    'E_REV_I'),
#        ('tau_syn_I',  'TAU_SYN_I'),
#    )


#class MockCurrentSource(object):
#    def inject_into(self, cells):
#        __doc__ = StandardCurrentSource.inject_into.__doc__
#        pass
#
#
#class DCSource(MockCurrentSource, electrodes.DCSource):
#    __doc__ = electrodes.DCSource.__doc__
#
#    translations = build_translations(
#        ('amplitude',  'amplitude'),
#        ('start',      'start'),
#        ('stop',       'stop')
#    )
#
#
#class StepCurrentSource(MockCurrentSource, electrodes.StepCurrentSource):
#    __doc__ = electrodes.StepCurrentSource.__doc__
#
#    translations = build_translations(
#        ('amplitudes',  'amplitudes'),
#        ('times',       'times')
#    )
#
#
#class ACSource(MockCurrentSource, electrodes.ACSource):
#    __doc__ = electrodes.ACSource.__doc__
#
#    translations = build_translations(
#        ('amplitude',  'amplitude'),
#        ('start',      'start'),
#        ('stop',       'stop'),
#        ('frequency',  'frequency'),
#        ('offset',     'offset'),
#        ('phase',      'phase')
#    )
#
#
#class NoisyCurrentSource(MockCurrentSource, electrodes.NoisyCurrentSource):
#
#    translations = build_translations(
#        ('mean',  'mean'),
#        ('start', 'start'),
#        ('stop',  'stop'),
#        ('stdev', 'stdev'),
#        ('dt',    'dt')
#    )


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
    )

    def _get_minimum_delay(self):
        return state.min_delay


#class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
#    __doc__ = synapses.TsodyksMarkramSynapse.__doc__
#
#    translations = build_translations(
#        ('weight', 'WEIGHT'),
#        ('delay', 'DELAY'),
#        ('U', 'UU'),
#        ('tau_rec', 'TAU_REC'),
#        ('tau_facil', 'TAU_FACIL'),
#        ('u0', 'U0'),
#        ('x0', 'X' ),
#        ('y0', 'Y')
#    )
#    
#    def _get_minimum_delay(self):
#        return state.min_delay
#
#
#class STDPMechanism(synapses.STDPMechanism):
#    __doc__ = synapses.STDPMechanism.__doc__
#
#    base_translations = build_translations(
#        ('weight', 'WEIGHT'),
#        ('delay', 'DELAY')
#    )
#
#    def _get_minimum_delay(self):
#        return state.min_delay
#    
#
#class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
#    __doc__ = synapses.AdditiveWeightDependence.__doc__
#
#    translations = build_translations(
#        ('w_max',     'wmax'),
#        ('w_min',     'wmin'),
#        ('A_plus',    'aLTP'),
#        ('A_minus',   'aLTD'),
#    )
#
#
#class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
#    __doc__ = synapses.MultiplicativeWeightDependence.__doc__
#
#    translations = build_translations(
#        ('w_max',     'wmax'),
#        ('w_min',     'wmin'),
#        ('A_plus',    'aLTP'),
#        ('A_minus',   'aLTD'),
#    )
#
#
#class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
#    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__
#
#    translations = build_translations(
#        ('w_max',     'wmax'),
#        ('w_min',     'wmin'),
#        ('A_plus',    'aLTP'),
#        ('A_minus',   'aLTD'),
#    )
#
#
#class GutigWeightDependence(synapses.GutigWeightDependence):
#    __doc__ = synapses.GutigWeightDependence.__doc__
#
#    translations = build_translations(
#        ('w_max',     'wmax'),
#        ('w_min',     'wmin'),
#        ('A_plus',    'aLTP'),
#        ('A_minus',   'aLTD'),
#        ('mu_plus',   'muLTP'),
#        ('mu_minus',  'muLTD'),
#    )
#
#
#class SpikePairRule(synapses.SpikePairRule):
#    __doc__ = synapses.SpikePairRule.__doc__
#
#    translations = build_translations(
#        ('tau_plus',  'tauLTP'),
#        ('tau_minus', 'tauLTD'),
#    )
