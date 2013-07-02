"""
Standard neuron, synapse and electrode models for pyNN.moose

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""


from pyNN.standardmodels import build_translations, cells, synapses
from .cells import StandardIF, SingleCompHH, PulseGenSpikeSource
from .simulator import state, mV, ms, nA, uS, nF

    


class IF_cond_exp(cells.IF_cond_exp):
    
    __doc__ = cells.IF_cond_exp.__doc__    
    
    translations = build_translations(
        ('tau_m',      'Rm',               '1e6*tau_m/cm',     '1e3*Rm*Cm'),
        ('cm',         'Cm',               nF),
        ('v_rest',     'Em',               mV),
        ('v_thresh',   'Vthreshold',       mV),
        ('v_reset',    'Vreset',           mV),
        ('tau_refrac', 'refractoryPeriod', ms),
        ('i_offset',   'inject',           nA),
        ('tau_syn_E',  'tau_e',            ms),
        ('tau_syn_I',  'tau_i',            ms),
        ('e_rev_E',    'e_e',              mV),
        ('e_rev_I',    'e_i',              mV)
    )
    model = StandardIF
    extra_parameters = {'syn_shape': 'exp'}


class IF_cond_alpha(cells.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""

    __doc__ = cells.IF_cond_alpha.__doc__        

    translations = IF_cond_exp.translations
    model = StandardIF
    extra_parameters = {'syn_shape': 'alpha'}


class HH_cond_exp(cells.HH_cond_exp):
    
    __doc__ = cells.HH_cond_exp.__doc__    

    translations = build_translations(
        ('gbar_Na',    'GbarNa',    uS),   
        ('gbar_K',     'GbarK',     uS),    
        ('g_leak',     'GLeak',     uS),    
        ('cm',         'Cm',        nF),  
        ('v_offset',   'Voff',      mV),
        ('e_rev_Na',   'ENa',       mV),
        ('e_rev_K',    'EK',        mV), 
        ('e_rev_leak', 'VLeak',     mV),
        ('e_rev_E',    'ESynE',     mV),
        ('e_rev_I',    'ESynI',     mV),
        ('tau_syn_E',  'tauE',      ms),
        ('tau_syn_I',  'tauI',      ms),
        ('i_offset',   'inject',    nA),
    )
    model = SingleCompHH



#class SpikeSourcePoisson(cells.SpikeSourcePoisson):
#    
#    __doc__ = cells.SpikeSourcePoisson.__doc__     
#
#    translations = build_translations(
#        ('start',    'start'),
#        ('rate',     'rate'),
#        ('duration', 'duration'),
#    )
#    model = RandomSpikeSource


class SpikeSourceArray(cells.SpikeSourceArray):
    
    __doc__ = cells.SpikeSourceArray.__doc__    

    translations = build_translations(
        ('spike_times', 'spike_times', ms),
    )
    model = PulseGenSpikeSource


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay', ms)
    )
    model = None

    def _get_minimum_delay(self):
        return state.min_delay




