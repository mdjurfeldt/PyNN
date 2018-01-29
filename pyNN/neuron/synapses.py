"""
Definition of NativeSynapseType class for NEURON

:copyright: Copyright 2006-2018 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import nest

from pyNN.models import BaseSynapseType
from .simulator import state


class NativeSynapseType(BaseSynapseType):

    @property
    def native_parameters(self):
        return self.parameter_space

    def get_native_names(self, *names):
        return names

    def _get_minimum_delay(self):
        return state.min_delay


