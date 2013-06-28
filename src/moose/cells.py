"""
MOOSE implementation of the PyNN API

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy as np
import moose
from .simulator import state, INIT_CLOCK, INTEGRATION_CLOCK, RECORDING_CLOCK, mV, ms, nA, uS, nF


class StandardIF(moose.LeakyIaF):
    
    def __init__(self, path, syn_shape, Cm=1.0, Em=0.0, Rm=1.0, Vreset=0.0,
                 Vthreshold=1.0, refractoryPeriod=0.0, inject=0.0, tau_e=0.001,
                 tau_i=0.001, e_e=0.0, e_i=-0.07):
        moose.LeakyIaF.__init__(self, path)
        self.Cm = Cm
        self.Em = Em
        self.Rm = Rm
        self.Vreset = Vreset
        self.Vthreshold = Vthreshold
        if refractoryPeriod != 0:  # in MOOSE, the membrane potential can evolve
                                   # during the refractory period. For the
                                   # other PyNN backends it is clamped to the
                                   # reset potential.
            raise ValueError("pyNN.moose does not support refractory periods.")
        self.refractoryPeriod = refractoryPeriod
        self.inject = inject
        
        self.syn_shape = syn_shape
        self.esyn = moose.SynChan("%s/excitatory" % path)
        self.isyn = moose.SynChan("%s/inhibitory" % path)
        for syn in self.esyn, self.isyn:
            syn.tau2 = 1e-6 # instantaneous rise, for shape=='exp'
            syn.Gbar = 1*uS
            moose.connect(syn, 'IkOut', self, 'injectDest')
            moose.connect(self, 'VmOut', syn, 'Vm')        
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.e_e = e_e
        self.e_i = e_i
        
        #self.source = moose.SpikeGen("source", self)
        #self.source.thresh = 0.0
        #self.source.abs_refract = 2.0
        #self.connect("VmSrc", self.source, "Vm")
        moose.useClock(INIT_CLOCK, self.path, 'init')
        moose.useClock(INTEGRATION_CLOCK, self.path, 'process')
        moose.useClock(INTEGRATION_CLOCK, self.esyn.path, 'process')
        moose.useClock(INTEGRATION_CLOCK, self.isyn.path, 'process')
        #moose.useClock(INTEGRATION_CLOCK, self.source.path)

        self.tables = {}

    def _get_tau_e(self):
        return self.esyn.tau1
    def _set_tau_e(self, val):
        self.esyn.tau1 = val
        if self.syn_shape == 'alpha':
            self.esyn.tau2 = val
    tau_e = property(fget=_get_tau_e, fset=_set_tau_e)
    
    def _get_tau_i(self):
        return self.isyn.tau1
    def _set_tau_i(self, val):
        self.isyn.tau1 = val
        if self.syn_shape == 'alpha':
            self.isyn.tau2 = val
    tau_i = property(fget=_get_tau_i, fset=_set_tau_i)
    
    def _get_e_e(self):
        return self.esyn.Ek
    def _set_e_e(self, val):
        self.esyn.Ek = val
    e_e = property(fget=_get_e_e, fset=_set_e_e)
    
    def _get_e_i(self):
        return self.isyn.Ek
    def _set_e_i(self, val):
        self.isyn.Ek = val
    e_i = property(fget=_get_e_i, fset=_set_e_i)


class PulseGenSpikeSource(object):
    
    def __init__(self, path, spike_times):
        """
        spike_times is a Sequence containing times in seconds
        """
        self.source = moose.SpikeGen(path)
        self.source.threshold = 0.1
        self.pg = moose.PulseGen("%s/pulses" % path)
        times = spike_times.value
        delays = np.empty((times.size + 1,))
        delays[0] = times[0]
        delays[1:-1] = times[1:] - times[:-1]
        delays[-1] = 1e15
        self.pg.count = delays.size
        for i, deltat in enumerate(delays):
            print i, deltat
            self.pg.delay[i] = deltat
            self.pg.width[i] = 0.001
            self.pg.level[i] = 0.5
        moose.useClock(INTEGRATION_CLOCK, self.source.path, "process")
        moose.useClock(INTEGRATION_CLOCK, self.pg.path, "process")
        moose.connect(self.pg, 'outputOut', self.source, 'Vm')  


class TableSpikeSource(object):
    """
    There must be a more efficient, event-based way to implement this.
    """
    
    def __init__(self, path, spike_times):
        self.spike_table = moose.StimulusTable(path)
        self.spike_table.startTime = 0.0
        self.spike_table.stepSize = state.dt*ms
        self.spike_table.stopTime = spike_times.max()*ms
        indices = (np.round(spike_times.value)/state.dt).astype(int)
        self.spike_table.vec = np.zeros((spike_times.max()/state.dt + 1,))
        self.spike_table.vec[indices] = 1
        
        self.source = moose.SpikeGen('%s/spike' % path)
        self.source.threshold = 0.5
        moose.connect(self.spike_table, 'output', self.source, 'Vm')
        
        moose.useClock(INTEGRATION_CLOCK, self.spike_table.path, 'process')
        moose.useClock(INTEGRATION_CLOCK, self.source.path, 'process')
        
    #def _save_spikes(self, spike_times):
    #    ms = 1e-3
    #    self._spike_times = spike_times
    #    filename = self.filename or "%s/%s.spikes" % (temporary_directory,
    #                                                  uuid.uuid4().hex)
    #    numpy.savetxt(filename, spike_times*ms, "%g")
    #    self.filename = filename
    #    
    #def _get_spike_times(self):
    #    return self._spike_times
    #def _set_spike_times(self, spike_times):
    #    self._save_spikes(spike_times)
    #spike_times = property(fget=_get_spike_times, fset=_set_spike_times)
