"""
MOOSE implementation of the PyNN API

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy as np
import moose
from .simulator import (state, INIT_TICK, MEMBRANE_INTEGRATION_TICK,
                        SYNAPSE_INTEGRATION_TICK, RECORDING_TICK,
                        mV, ms, nA, uS, nF)


class StandardIF(moose.LeakyIaF):
    
    def __init__(self, path, syn_shape, Cm=1.0, Em=0.0, Rm=1.0, Vreset=0.0,
                 Vthreshold=1.0, refractoryPeriod=0.0, inject=0.0, tau_e=0.001,
                 tau_i=0.001, e_e=0.0, e_i=-0.07):  # should add units, as in SingleCompHH
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
        for syn in (self.esyn, self.isyn):
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
        moose.useClock(INIT_TICK, self.path, 'init')
        moose.useClock(INIT_TICK, self.esyn.path, 'init') ##
        moose.useClock(INIT_TICK, self.isyn.path, 'init') ##
        moose.useClock(MEMBRANE_INTEGRATION_TICK, self.path, 'process')
        moose.useClock(SYNAPSE_INTEGRATION_TICK, self.esyn.path, 'process')
        moose.useClock(SYNAPSE_INTEGRATION_TICK, self.isyn.path, 'process')
        #moose.useClock(MEMBRANE_INTEGRATION_TICK, self.source.path)

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


class SingleCompHH(moose.Compartment):
    
    def __init__(self, path, GbarNa=20*uS, GbarK=6*uS, GLeak=0.01*uS, Cm=0.2*nF,
                 ENa=40*mV, EK=-90*mV, VLeak=-65*mV, Voff=-63*mV, ESynE=0*mV,
                 ESynI=-70*mV, tauE=2*ms, tauI=5*ms, inject=0*nA, initVm=-65*mV):
        moose.Compartment.__init__(self, path)
        self.initVm = initVm
        self.Rm = 1/GLeak
        self.Cm = Cm
        self.Em = VLeak
        self.inject = inject
        
        vdiv = 150
        vmin = -100*mV
        vmax = 50*mV
        
        self.na = moose.HHChannel("%s/na" % path)
        self.na.Ek = ENa
        self.na.Gbar = GbarNa
        self.na.Xpower = 3
        self.na.Ypower = 1
        gate = moose.HHGate("%s/gateX" % self.na.path)
        gate.setupAlpha([3.2e5 * (13*mV+Voff), -3.2e5, -1, -(13*mV+Voff), -4*mV, # alpha
                         -2.8e5 * (40*mV+Voff),  2.8e5, -1, -(40*mV+Voff), 5*mV,  # beta
                        vdiv, vmin, vmax])
        gate = moose.HHGate("%s/gateY" % self.na.path)
        gate.setupAlpha([128,                   0,      0, -(17*mV+Voff), 18*mV, # alpha
                         4.0e3,                 0,      1, -(40*mV+Voff), -5*mV, # beta
                        vdiv, vmin, vmax])
        self.k = moose.HHChannel("%s/k" % path)
        self.k.Ek = EK
        self.k.Gbar = GbarK
        self.k.Xpower = 4
        gate = moose.HHGate("%s/gateX" % self.k.path)
        gate.setupAlpha([3.2e4 * (15*mV+Voff), -3.2e4, -1, -(15*mV+Voff), -5*mV,
                         500,                  0,       0, -(10*mV+Voff),  40*mV,
                         vdiv, vmin, vmax])
        self.na.X = 0  # these initial values should really be handled at the PyNN level
        self.na.Y = 1
        self.k.X = 0

        self.esyn = moose.SynChan("%s/excitatory" % path)
        self.isyn = moose.SynChan("%s/inhibitory" % path)
        for syn in (self.esyn, self.isyn):
            syn.tau2 = 1e-6 # instantaneous rise, for shape=='exp'
            syn.Gbar = 1*uS
            moose.connect(syn, "channel", self, "channel")
       
        self.tauE = tauE
        self.tauI = tauI
        self.ESynE = ESynE
        self.ESynI = ESynI

        moose.connect(self.na, "channel", self, "channel")
        moose.connect(self.k, "channel", self, "channel")

        for obj in (self, self.na, self.k):
            moose.useClock(INIT_TICK, obj.path, "init")
            moose.useClock(MEMBRANE_INTEGRATION_TICK, obj.path, "process")
        for obj in (self.esyn, self.isyn):
            moose.useClock(INIT_TICK, obj.path, 'init')
            moose.useClock(SYNAPSE_INTEGRATION_TICK, obj.path, 'process')
        
        #self.source = moose.SpikeGen("source", self)
        #self.source.thresh = 0.0
        #self.source.abs_refract = 2.0
        #self.connect("VmSrc", self.source, "Vm")

        self.tables = {}

    def _get_tau_e(self):
        return self.esyn.tau1
    def _set_tau_e(self, val):
        self.esyn.tau1 = val
    tauE = property(fget=_get_tau_e, fset=_set_tau_e)
    
    def _get_tau_i(self):
        return self.isyn.tau1
    def _set_tau_i(self, val):
        self.isyn.tau1 = val
    tauI = property(fget=_get_tau_i, fset=_set_tau_i)
    
    def _get_e_e(self):
        return self.esyn.Ek
    def _set_e_e(self, val):
        self.esyn.Ek = val
    ESynE = property(fget=_get_e_e, fset=_set_e_e)
    
    def _get_e_i(self):
        return self.isyn.Ek
    def _set_e_i(self, val):
        self.isyn.Ek = val
    ESynI = property(fget=_get_e_i, fset=_set_e_i)


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
            self.pg.delay[i] = deltat
            self.pg.width[i] = 2*state.clock.tick[MEMBRANE_INTEGRATION_TICK].dt  # to test: would a width of 1*dt work?
            self.pg.level[i] = 0.5
        moose.useClock(MEMBRANE_INTEGRATION_TICK, self.source.path, "process")
        moose.useClock(MEMBRANE_INTEGRATION_TICK, self.pg.path, "process")
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
        
        moose.useClock(MEMBRANE_INTEGRATION_TICK, self.spike_table.path, 'process')
        moose.useClock(MEMBRANE_INTEGRATION_TICK, self.source.path, 'process')
