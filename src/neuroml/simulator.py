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
        # for now, putting all the NeuroML code into this single method, later,
        # should consider either adding a "to_xml()" method to each relevant
        # class, or use the Visitor pattern.
        # The latter would potentially allow implementing neuroml, nineml and
        # nest_sli backends with minimal redundancy and duplicated code
        doc = neuroml.NeuroMLDocument(id=label)
        net = neuroml.Network(id=label)
        doc.networks.append(net)
        for pp in self.populations:
            cell_list = getattr(doc, pp.celltype.neuroml_cell_list)
            cell_list.append(
                pp.celltype.neuroml_cell_component(
                                    id=pp.celltype.label,
                                    **pp.celltype.cell_parameters))
            for rt in pp.celltype.receptor_types:
                doc.exp_one_synapses.append(
                    pp.celltype.neuroml_receptor_component(
                                    id="%s_%s" % (pp.celltype.label, rt),
                                    gbase="1 uS",
                                    **pp.celltype.receptor_parameters(rt)))
            net.populations.append(
                neuroml.Population(id=pp.label,
                                   component=pp.celltype.label,
                                   size=pp.size))
        for prj in self.projections:
            # there is a neuroml.Projection class, but I'm not sure how well supported it is,
            # so using neuroml.SynapticConnection for now.
            for c in prj.connections:
                net.synaptic_connections.append(
                    neuroml.SynapticConnection(
                        from_="%s[%i]" % (prj.pre.label, c.presynaptic_index),
                        synapse="%s_%s" % (prj.post.celltype.label, prj.receptor_type),
                        to="%s[%i]" % (prj.post.label, c.postsynaptic_index)))
        NeuroMLWriter.write(doc, self.xmlfile, validate=False)


state = State()
