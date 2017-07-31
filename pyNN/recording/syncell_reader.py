"""

"""


# todo: add support for rho_stim, rho_threshold and hypamp, since the NEST model changed
# todo: handle poisson_freq
# todo: check data types (e.g. i8 vs i4) in saved files

import sys
from collections import defaultdict
from pprint import pprint
import numpy as np
import h5py

from pyNN.connectors import Connector
from pyNN.parameters import Sequence, ParameterSpace
from pyNN.nest.standardmodels.cells import RoessertEtAl as RoessertEtAl_nest
import pyNN.nest as sim



class Network(object):

    @classmethod
    def from_syncell_files(cls, neuron_file_path, synapse_file_path):
        fn = h5py.File(neuron_file_path, "r")
        fs = h5py.File(synapse_file_path, "r")

        neuron_params = fn['neurons']['default']
        presyn_params = fs['presyn']['default']
        postsyn_params = fs['postreceptors']['default']
        gids = neuron_params['gid']
        # todo: figure out what hypamp, rho_stim and rho_threshold parameters are for

        receptor_ids = np.unique(presyn_params['receptor_ids'])
        receptor_parameters = {}
        for name in postsyn_params.keys():
            if name not in ('gid', 'n_receptors'):
                receptor_parameters[name] = {}
                for receptor_id in receptor_ids:
                    receptor_parameters[name][str(receptor_id)] = postsyn_params[name][:, receptor_id - 1]

        # === Create neurons =====
        neurons = sim.Population(
            gids.size,
            sim.RoessertEtAl(
                v_rest=neuron_params['E_L'].value,
                cm=0.001*neuron_params['C_m'].value,
                tau_m=neuron_params['C_m'].value/neuron_params['g_L'].value,
                tau_refrac=neuron_params['t_ref'].value,
                v_reset=neuron_params['V_reset'].value,
                i_offset=0,
                delta_v=neuron_params['Delta_V'].value,
                v_t_star=neuron_params['V_T_star'].value,
                lambda0=neuron_params['lambda_0'].value,
                tau_eta=[Sequence(x) for x in neuron_params['tau_stc']],
                tau_gamma=[Sequence(x) for x in neuron_params['tau_sfa']],
                a_eta=[Sequence(0.001*x) for x in neuron_params['q_stc']],
                a_gamma=[Sequence(x) for x in neuron_params['q_sfa']],
                e_syn_fast=receptor_parameters['E_rev_B'],
                e_syn_slow=receptor_parameters['E_rev'],
                tau_syn_fast_rise=receptor_parameters['tau_r_fast'],
                tau_syn_fast_decay=receptor_parameters['tau_d_fast'],
                tau_syn_slow_rise=receptor_parameters['tau_r_slow'],
                tau_syn_slow_decay=receptor_parameters['tau_d_slow'],
                ratio_slow_fast=receptor_parameters['ratio_slow'],
                mg_conc=receptor_parameters['mg'],
                tau_corr=receptor_parameters['tau_corr'],
                g_max=0.001
            ),
            #structure=?
            label="neurons")
        neurons.annotate(first_gid=gids[0])

        fn.close()
        fs.close()

        # === Create connectors =====

        connections = []
        offset = 0
        for receptor_id in receptor_ids:
            connections.append(
                sim.Projection(neurons, neurons,
                               SynCellFileConnector(synapse_file_path, offset=offset),
                               sim.TsodyksMarkramSynapseEM(),
                               receptor_type=str(receptor_id)))
            offset += connections[-1].size()


        obj = cls()
        obj.populations = [neurons]
        obj.projections = connections

        return obj

    @property
    def all_receptor_types(self):
        receptor_types = set(self.populations[0].celltype.receptor_types)
        for population in self.populations[1:]:
            receptor_types.update(population.celltype.receptor_types)
        return sorted(list(receptor_types))

    def get_receptor_index(self, receptor_type):
        return self.all_receptor_types.index(receptor_type)  # receptor id is receptor index + 1

    @property
    def connection_count(self):
        return sum(prj.size(gather=True) for prj in self.projections)

    def save_to_syncell_files(self, neuron_file_path, synapse_file_path):
        # only saves the first population and projection for now
        fn = h5py.File(neuron_file_path, "w")
        fs = h5py.File(synapse_file_path, "w")

        neuron_params = fn.create_group("neurons/default")  # use population.label instead of "default"?
        presyn_params =  fs.create_group("presyn/default")
        postsyn_params =  fs.create_group("postreceptors/default")

        # Write to "neurons" group
        assert len(self.populations) == 1, "Can't yet handle multiple populations"
        names = self.populations[0].celltype.get_parameter_names()
        # It appears the syncell format uses the NEST names,
        # so if using pyNN.nest we could take a shortcut and avoid
        # double translation.
        # Not doing this yet, for generality
        parameters = self.populations[0]._get_parameter_dict(names, gather=True, simplify=True)
        translated_parameters = RoessertEtAl_nest(**parameters).native_parameters
        translated_parameters.shape = (self.populations[0].size,)
        translated_parameters.evaluate(simplify=False)

        psr_names = ['tau_d_fast', 'tau_d_slow', 'tau_r_fast', 'tau_r_slow',
                     'E_rev', 'E_rev_B', 'ratio_slow', 'mg', 'tau_corr']
        excluded_names = psr_names + ['g_max']
        for name, value in translated_parameters.items():
            if name not in excluded_names:
                if isinstance(value[0], Sequence):
                    value = np.array([seq.value for seq in value])
                neuron_params.create_dataset(name, data=value)
        if 'first_gid' in self.populations[0].annotations:
            gid_offset = self.populations[0].annotations['first_gid'] - self.populations[0][0]
        else:
            gid_offset = 0
        neuron_params.create_dataset('gid', data=self.populations[0].all_cells.astype(int) + gid_offset)

        # Write to "presyn" and "postsyn" groups
        presyn_attribute_names = self.projections[0].synapse_type.get_parameter_names()
        # todo: assert that all projections have the same synapse_type

        presyn_params.create_dataset("pre_gid", (self.connection_count,), dtype='i4')
        presyn_params.create_dataset("post_gid", (self.connection_count,), dtype='i4')
        presyn_params.create_dataset("receptor_ids", (self.connection_count,), dtype='i4')
        for name in presyn_attribute_names:
            presyn_params.create_dataset(name, (self.connection_count,), dtype=float)

        n_post = self.projections[0].post.size
        n_receptors = len(self.projections)
        for name in psr_names:
            postsyn_params.create_dataset(name, (n_post, n_receptors), dtype=float)

        offset = 0
        for projection in self.projections:
            assert projection.pre == self.populations[0]
            assert projection.post == self.populations[0]
            projection_size = projection.size(gather=True)

            receptor_index = self.get_receptor_index(projection.receptor_type)

            presyn_params["receptor_ids"][offset:projection_size + offset] = receptor_index + 1  # receptor_ids count from 1
            for name in presyn_attribute_names:
                values = np.array(projection.get(name, format='list', gather=True, with_address=False))
                # todo: translate names and units where needed
                if name == "weight":
                    values *= 1000
                presyn_params[name][offset:projection_size + offset] = values

            presyn_idx, postsyn_idx, _ = np.array(
                projection.get('weight', format='list', gather=True, with_address=True)).T
            presyn_idx = presyn_idx.astype(int)
            postsyn_idx = postsyn_idx.astype(int)
            presyn_params["pre_gid"][offset:projection_size + offset] = index_to_gid(projection.pre, presyn_idx)
            presyn_params["post_gid"][offset:projection_size + offset] = index_to_gid(projection.pre, postsyn_idx)

            for name in psr_names:
                values = np.array([x.value[receptor_index] for x in translated_parameters[name]])
                postsyn_params[name][:, receptor_index] = values
            offset += projection_size

        postsyn_params.create_dataset("n_receptors", (n_post,), dtype='i4')  # todo: populate this dataset
        postsyn_params.create_dataset("gid", data=index_to_gid(projection.post, postsyn_idx), dtype='i4')

        fn.close()
        fs.close()



def gid_to_index(population, gid):
    # this relies on gids being sequential
    if "first_gid" in population.annotations:
        offset = population.annotations["first_gid"]
    else:
        offset = population.first_id
    return gid - offset


def index_to_gid(population, index):
    # this relies on gids being sequential
    if "first_gid" in population.annotations:
        offset = population.annotations["first_gid"]
    else:
        offset = population.first_id
    return index + offset


class SynCellFileConnector(Connector):
    """

    """

    def __init__(self, path, offset=0, safe=True, callback=None):
        f = h5py.File(path, "r")
        
        self.presyn_params = f['presyn']['default']
        self.postsyn_params = f['postreceptors']['default']

    def connect(self, projection):
        for post_index in projection.post._mask_local.nonzero()[0]:
            post_gid = index_to_gid(projection.post, post_index)
            # could process file by chunks, if memory becomes a problem for these masks
            mask1 = self.presyn_params["post_gid"].value == post_gid
            mask2 = self.presyn_params["receptor_ids"].value == int(projection.receptor_type)
            mask = mask1 * mask2

            pre_indices = gid_to_index(projection.pre, self.presyn_params["pre_gid"][mask])

            connection_parameters = {}
            for name in ('U', 'delay', 'tau_fac', 'tau_rec', 'weight'):
                # todo: handle w_corr
                if name == "tau_fac":   # hack - todo: proper translation
                    tname = "tau_facil"
                else:
                    tname = name
                connection_parameters[name] = self.presyn_params[tname][mask]
            # now translate names and units

            # create connections
            projection._convergent_connect(pre_indices, post_index, **connection_parameters)


class SynCellFileConnector2(SynCellFileConnector):
    """

    """

    def connect(self, projection):
        for i, post_gid in enumerate(self.presyn_params["post_gid"]):
            post_index = gid_to_index(projection.post, post_gid)
            if (str(self.postsyn_params["receptor_ids"][i]) == projection.receptor_type
                and projection.post._mask_local[post_index]):
                pre_index = gid_to_index(projection.pre, self.presyn_params["pre_gid"][i])
                ##print("i={} post_gid={} post_index={} receptor_type={} pre_index={}".format(i, post_gid, post_index, projection.receptor_type, pre_index))
                connection_parameters = {}
                for name in ('U', 'delay', 'tau_fac', 'tau_rec', 'weight'):
                    # todo: handle w_corr
                    if name == "tau_fac":  # hack - todo: proper translation
                        tname = "tau_facil"
                    else:
                        tname = name
                    connection_parameters[name] = self.presyn_params[tname][i]
                # now translate names and units

                # create connections
                projection._convergent_connect(np.array([pre_index]), post_index, **connection_parameters)
