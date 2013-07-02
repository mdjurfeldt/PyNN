# encoding: utf-8
"""
MOOSE implementation of the PyNN API

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
from pyNN import common
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace, simplify
from . import simulator
from .recording import Recorder



class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        #parameter_dict = {}
        #for name in names:
        #    value = self.parent._parameters[name]
        #    if isinstance(value, numpy.ndarray):
        #        value = value[self.mask]
        #    parameter_dict[name] = simplify(value)
        #return ParameterSpace(parameter_dict, shape=(self.size,)) # or local size?
        raise NotImplementedError()

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        ##ps = self.parent._get_parameters(*self.celltype.get_native_names())
        #for name, value in parameter_space.items():
        #    self.parent._parameters[name][self.mask] = value.evaluate(simplify=True)
        #    #ps[name][self.mask] = value.evaluate(simplify=True)
        ##ps.evaluate(simplify=True)
        ##self.parent._parameters = ps.as_dict()
        raise NotImplementedError()


    def _set_initial_value_array(self, variable, initial_values):
        raise NotImplementedError()

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)



class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def _create_cells(self):
        id_range = numpy.arange(simulator.state.gid_counter,
                                simulator.state.gid_counter + self.size)
        self.all_cells = numpy.array([simulator.ID(id) for id in id_range],
                                     dtype=simulator.ID)
        def is_local(id):
            return (id % simulator.state.num_processes) == simulator.state.mpi_rank
        self._mask_local = is_local(self.all_cells)
        
        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        parameter_space.evaluate(mask=None, simplify=False)
        
        for i, (id, is_local, params) in enumerate(zip(self.all_cells, self._mask_local, parameter_space)):
            self.all_cells[i].parent = self
            if is_local:
                if hasattr(self.celltype, "extra_parameters"):
                    params.update(self.celltype.extra_parameters)
                self.all_cells[i]._build_cell(self.celltype.model, params)
        simulator.state.gid_counter += self.size

    def _set_initial_value_array(self, variable, initial_values):
        initial_values = initial_values * simulator.mV  # tmp hack, should use SCALE_FACTORS
        if variable == 'v':  # temporary hack, need to handle other variables as well
            if initial_values.is_homogeneous:
                value = initial_values.evaluate(simplify=True)
                for cell in self:  # only on local node
                    setattr(cell._cell, "initVm", value)
            else:
                if isinstance(initial_values.base_value, RandomDistribution) and initial_values.base_value.rng.parallel_safe:
                    local_values = initial_values.evaluate()[self._mask_local]
                else:
                    local_values = initial_values[self._mask_local]            
                for cell, value in zip(self, local_values):
                    setattr(cell._cell, "initVm", value)
    
    #def _get_view(self, selector, label=None):
    #    return PopulationView(self, selector, label)
    #
    #def _get_parameters(self, *names):
    #    """
    #    return a ParameterSpace containing native parameters
    #    """
    #    parameter_dict = {}
    #    for name in names:
    #        parameter_dict[name] = simplify(self._parameters[name])
    #    return ParameterSpace(parameter_dict, shape=(self.local_size,))
    #
    #def _set_parameters(self, parameter_space):
    #    """parameter_space should contain native parameters"""
    #    #ps = self._get_parameters(*self.celltype.get_native_names())
    #    #ps.update(**parameter_space)
    #    #ps.evaluate(simplify=True)
    #    #self._parameters = ps.as_dict()
    #    parameter_space.evaluate(simplify=False, mask=self._mask_local)
    #    for name, value in parameter_space.items():
    #        self._parameters[name] = value
