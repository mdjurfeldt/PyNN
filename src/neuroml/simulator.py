"""

"""

from pyNN import common
import neuroml
from neuroml.writers import NeuroMLWriter

name = "NeuroML"

class ID(int, common.IDMixin):
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


class State(common.control.BaseState):
    
    def __init__(self):
        common.control.BaseState.__init__(self)
        self.mpi_rank = 0
        self.num_processes = 1
        self.clear()
        self.dt = 0.1
        self.xmlfile = "PyNN2NeuroML"
    
    def run(self, simtime):
        self.t += simtime
        self.running = True
    
    def clear(self):
        self.recorders = set([])
        self.populations = []
        self.projections = []
        self.current_sources = []
        self.id_counter = 0
        self.segment_counter = -1
        self.reset()
    
    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1
    
    def export(self, label="PyNN_network"):
        """Export the model as NeuroML"""
        # and, in future, perhaps SED-ML as well
        doc = neuroml.NeuroMLDocument(id=label)
        net = neuroml.Network(id=label)
        doc.networks.append(net)
        for pp in self.populations:
            pp.to_neuroml(doc, net)
        for prj in self.projections:
            prj.to_neuroml(net)
        NeuroMLWriter.write(doc, self.xmlfile, validate=False)


state = State()
