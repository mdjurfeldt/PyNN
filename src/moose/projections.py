"""
MOOSE implementation of the PyNN API

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from itertools import repeat, izip
import moose
from pyNN import common
from pyNN.core import ezip
from pyNN.parameters import ParameterSpace
from pyNN.space import Space
from . import simulator


class Connection(object):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, pre, post, synapse_object):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.synapse = synapse_object

#    def as_tuple(self, *attribute_names):
#        # should return indices, not IDs for source and target
#        return tuple([getattr(self, name) for name in attribute_names])


class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 space=Space(), label=None):
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)

        ## Create connections
        self.connections = []
        connector.connect(self)

    def __len__(self):
        return len(self.connections)

    def set(self, **attributes):
        parameter_space = ParameterSpace

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        for name, value in connection_parameters.items():
            if isinstance(value, float):
                connection_parameters[name] = repeat(value)
                
        target_cell = self.post[postsynaptic_index]._cell
        if self.receptor_type == "excitatory":
            receptor_object = target_cell.esyn
        elif self.receptor_type == "inhibitory":
            receptor_object =  target_cell.isyn
        else:
            receptor_object = getattr(target_cell, self.receptor_type)
        start_index = receptor_object.synapse.num
        receptor_object.synapse.num += presynaptic_indices.size
        for j, (pre_idx, values) in enumerate(
                                       ezip(presynaptic_indices,
                                            *connection_parameters.values())):
            attributes = dict(zip(connection_parameters.keys(), values))
            synapse_object = receptor_object.synapse[start_index + j]
            for name, value in attributes.items():
                setattr(synapse_object, name, value)
                print "Setting %s.%s = %s" % (synapse_object, name, value)
            presynaptic_cell = self.pre[pre_idx]._cell
            moose.connect(presynaptic_cell.source, 'event', synapse_object, 'addSpike')
            print "Connecting %s.event to %s.addSpike" % (presynaptic_cell.source, synapse_object)
            self.connections.append(
                Connection(pre_idx, postsynaptic_index, synapse_object)
            )