"""

"""

# todo: handle pre_ids from other hypercolumns
# todo: check data types (e.g. i8 vs i4) in saved files

from itertools import chain
import numpy as np
import h5py

from pyNN.connectors import Connector
from pyNN.parameters import Sequence
from pyNN.nest.standardmodels.cells import RoessertEtAl as RoessertEtAl_nest
import pyNN.nest as sim

from . import get_mpi_comm, gather_dict

mpi_comm, mpi_flags = get_mpi_comm()


class Network(object):

    @classmethod
    def from_syncell_files(cls, neuron_file_path, synapse_file_path):
        fn = h5py.File(neuron_file_path, "r")
        fs = h5py.File(synapse_file_path, "r")

        neuron_params = fn['neurons']['default']
        presyn_params = fs['presyn']['default']
        postsyn_params = fs['postreceptors']['default']
        stim_params = fs['stimulation']['default']
        gids = neuron_params['gid']

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
                i_hyp=0.001*neuron_params['hypamp'].value,
                i_rho_thresh=0.001*neuron_params['rho_threshold'].value,
                f_rho_stim=0.01*stim_params['rho_stim'].value,
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

        # === Create external stimulation =====
        stim_gids = stim_params["stim_gids"]
        stim = sim.Population(
            stim_gids.size,
            sim.SpikeSourceArray(), # ?
            label="stim"
        )
        stim.annotate(first_gid=stim_gids[0])

        # === Create connections =====

        connections = []
        for receptor_id in receptor_ids:
            connections.append(
                sim.Projection(neurons, neurons,
                               SynCellFileConnector(synapse_file_path),
                               sim.TsodyksMarkramSynapseEM(),
                               receptor_type=str(receptor_id)))

        stim_connections = []
        for receptor_id in receptor_ids:
            stim_connections.append(
                sim.Projection(stim, neurons,
                               SynCellFileConnector(synapse_file_path),
                               sim.TsodyksMarkramSynapseEM(),
                               receptor_type=str(receptor_id)))

        # === Create populations and connections for spontaneous EPSP minis =====

        mepsp_populations = []
        mepsp_connections = []
        firing_rates = presyn_params['poisson_freq']
        offset = 0
        for projection in chain(connections, stim_connections):
            size = projection.size()
            if size > 0:
                mepsp_populations.append(
                    sim.Population(size,
                                   sim.SpikeSourcePoisson(rate=firing_rates[offset: size + offset]),
                                   label="Poisson source for {}".format(projection.label)
                                   )
                )
                connection_properties = np.array(projection.get(('weight', 'delay', 'U'), format='list'))
                connection_properties[:, 0] = np.arange(size)
                mepsp_connections.append(
                    sim.Projection(mepsp_populations[-1], projection.post,
                                   sim.FromListConnector(connection_properties, column_names=('weight', 'delay', 'U')),
                                   sim.TsodyksMarkramSynapseEM(),
                                   receptor_type=projection.receptor_type,
                                   label="spontaneous mEPSP connection for {}".format(projection.label))
                )
                mepsp_connections[-1]._size = size  # optimization
                offset += size

        # === Create Poisson populations for connections from outside the modelled region =====

        # any gids in presyn_params["pre_gid"] that are not in neuron_params["gid"] or stim_params["stim_gids"]
        # are from another hypercolumn, so we do not instantiate the neurons, but we do need to
        # add Poisson sources for the spontaneous EPSPs
        accounted_for = []
        pre_gids = presyn_params["pre_gid"].value
        for population in (neurons, stim):
            first_gid = population.annotations['first_gid']
            accounted_for.append(
                np.logical_and(pre_gids >= first_gid,
                               pre_gids < first_gid + population.size)

            )
        not_accounted_for = ~np.logical_or(*accounted_for)

        gids_other = pre_gids[not_accounted_for]
        weights = 0.001*presyn_params["weight"][not_accounted_for]
        delays = presyn_params["delay"][not_accounted_for]
        U = presyn_params["U"][not_accounted_for]
        receptor_ids = presyn_params["receptor_ids"][not_accounted_for]
        post_gids = presyn_params["post_gid"][not_accounted_for]
        firing_rates = presyn_params['poisson_freq'][not_accounted_for]

        other_population = sim.Population(gids_other.size,
                                          sim.SpikeSourcePoisson(rate=firing_rates),
                                          label="Poisson source for connections from other neurons")
        other_population.annotate(gids=gids_other)

        other_projections = []
        for receptor_id in np.unique(receptor_ids):
            mask = receptor_ids == receptor_id
            size = mask.sum()
            connection_properties = np.vstack((
                np.arange(size),
                gid_to_index(neurons, post_gids[mask])[0],
                weights[mask],
                delays[mask],
                U[mask]
            )).T
            other_projections.append(
                sim.Projection(other_population, neurons,
                               sim.FromListConnector(connection_properties, column_names=('weight', 'delay', 'U')),
                               sim.TsodyksMarkramSynapseEM(),
                               receptor_type=str(receptor_id)))
            other_projections[-1]._size = size  # optimization

        # === Construct the Network object =====

        obj = cls()
        obj.populations = [neurons]
        obj.stim_populations = [stim]
        obj.projections = connections
        obj.stim_projections = stim_connections
        obj.stim_spontaneous = mepsp_populations
        obj.spontaneous_projections = mepsp_connections
        obj.other_populations = [other_population]
        obj.other_projections = other_projections

        fn.close()
        fs.close()

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
        return sum(prj.size(gather=True) for prj in chain(self.projections, self.stim_projections, self.other_projections))

    def _get_local_sizes(self, population):
        size = {sim.rank(): population.local_size}
        return gather_dict(size, all=True)

    def _get_mpi_offset(self, population):
        local_sizes = self._get_local_sizes(population)
        mpi_offsets = [0]
        for i in range(1, sim.num_processes()):
            mpi_offsets.append(local_sizes[i - 1] + mpi_offsets[i - 1])
        return mpi_offsets[sim.rank()]

    def save_to_syncell_files(self, neuron_file_path, synapse_file_path, write_parallel=False):
        # only saves the first population and projection for now

        if write_parallel:
            fn = h5py.File(neuron_file_path, "w", driver = 'mpio', comm=mpi_comm)
            fs = h5py.File(synapse_file_path, "w", driver = 'mpio', comm=mpi_comm)
        else:
            fn = h5py.File(neuron_file_path, "w")
            fs = h5py.File(synapse_file_path, "w")
        gather = not write_parallel

        neuron_params = fn.create_group("neurons/default")  # use population.label instead of "default"?
        presyn_params =  fs.create_group("presyn/default")
        postsyn_params =  fs.create_group("postreceptors/default")
        stim_params = fs.create_group("stimulation/default")

        # Write to "neurons" group
        assert len(self.populations) == 1, "Can't yet handle multiple populations"
        names = self.populations[0].celltype.get_parameter_names()
        # It appears the syncell format uses the NEST names,
        # so if using pyNN.nest we could take a shortcut and avoid
        # double translation.
        # Not doing this yet, for generality
        parameters = self.populations[0]._get_parameter_dict(names, gather=gather, simplify=True)
        translated_parameters = RoessertEtAl_nest(**parameters).native_parameters
        translated_parameters.shape = (self.populations[0].local_size,)
        translated_parameters.evaluate(simplify=False)

        mpi_offset = self._get_mpi_offset(self.populations[0])

        psr_names = ['tau_d_fast', 'tau_d_slow', 'tau_r_fast', 'tau_r_slow',
                     'E_rev', 'E_rev_B', 'ratio_slow', 'mg', 'tau_corr']
        excluded_names = psr_names + ['g_max', 'rho_stim']
        for name, value in translated_parameters.items():
            if name not in excluded_names:
                if isinstance(value[0], Sequence):
                    value = np.array([seq.value for seq in value])
                neuron_params.create_dataset(name, shape=(self.populations[0].size,), dtype='f')  # check if this is the correct dtype
                neuron_params[name][mpi_offset:mpi_offset + value.size] = value

        def get_gids(population):
            if 'first_gid' in population.annotations:
                gid_offset = population.annotations['first_gid'] - population[0]
            else:
                gid_offset = 0
            gids = population.local_cells.astype(int) + gid_offset
            return gids

        gids = get_gids(self.populations[0])
        neuron_params.create_dataset('gid', shape=(self.populations[0].size,), dtype='i4')
        neuron_params['gid'][mpi_offset:mpi_offset + gids.size] = gids

        # Write to "stimulation" group
        mpi_offset = self._get_mpi_offset(self.stim_populations[0])
        stim_gids = get_gids(self.stim_populations[0])
        stim_params.create_dataset('gid', shape=(self.stim_populations[0].size,), dtype='i4')
        stim_params['gid'][mpi_offset:mpi_offset + gids.size] = stim_gids
        stim_params.create_dataset('rho_stim', shape=(self.stim_populations[0].size,), dtype='f')
        stim_params['rho_stim'][mpi_offset:mpi_offset + stim_gids.size] = translated_parameters['rho_stim']
        stim_params.create_dataset('stim_gids', shape=(self.stim_populations[0].size,), dtype='f')
        stim_params['stim_gids'][mpi_offset:mpi_offset + stim_gids.size] = stim_gids

        ### UPDATED FOR PARALLEL WRITING AS FAR AS HERE - TO FINISH

        # Write to "presyn" and "postsyn" groups
        presyn_attribute_names = self.projections[0].synapse_type.get_parameter_names()
        # todo: assert that all projections have the same synapse_type

        presyn_params.create_dataset("pre_gid", (self.connection_count,), dtype='i4')
        presyn_params.create_dataset("post_gid", (self.connection_count,), dtype='i4')
        presyn_params.create_dataset("receptor_ids", (self.connection_count,), dtype='i4')
        presyn_params.create_dataset("poisson_freq", (self.connection_count,), dtype=float)

        for name in presyn_attribute_names:
            presyn_params.create_dataset(name, (self.connection_count,), dtype=float)

        n_post = self.projections[0].post.size
        n_receptors = len(self.projections)
        for name in psr_names:
            postsyn_params.create_dataset(name, (n_post, n_receptors), dtype=float)

        offset = 0
        for projection in chain(self.projections, self.stim_projections, self.other_projections):
            projection_size = projection.size(gather=True)

            if projection_size > 0:

                receptor_index = self.get_receptor_index(projection.receptor_type)
                presyn_params["receptor_ids"][offset:projection_size + offset] = receptor_index + 1  # receptor_ids count from 1
                synapse_properties = np.array(projection.get(presyn_attribute_names, format='list', gather=True, with_address=True)).T

                for i, name in enumerate(presyn_attribute_names):
                    ##values = np.array(projection.get(name, format='list', gather=True, with_address=False))
                    values = synapse_properties[i + 2]
                    # todo: translate names and units where needed
                    if name == "weight":
                        values *= 1000
                    presyn_params[name][offset:projection_size + offset] = values

                ##presyn_idx, postsyn_idx, _ = np.array(
                ##    projection.get('weight', format='list', gather=True, with_address=True)).T
                presyn_idx, postsyn_idx = synapse_properties[0:2, :]
                presyn_idx = presyn_idx.astype(int)
                postsyn_idx = postsyn_idx.astype(int)
                presyn_params["pre_gid"][offset:projection_size + offset] = index_to_gid(projection.pre, presyn_idx)
                presyn_params["post_gid"][offset:projection_size + offset] = index_to_gid(projection.post, postsyn_idx)

                for name in psr_names:
                    values = np.array([x.value[receptor_index] for x in translated_parameters[name]])
                    postsyn_params[name][:, receptor_index] = values
                offset += projection_size
                #print(projection.label, presyn_idx, postsyn_idx, index_to_gid(projection.pre, postsyn_idx), index_to_gid(projection.post, postsyn_idx), offset)

        offset = 0
        for population in self.stim_spontaneous:
            presyn_params["poisson_freq"][offset:population.size + offset] = population.get("rate", gather=True)
            offset += population.size

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
    candidate_indices = gid - offset
    in_population = np.logical_and(0 <= candidate_indices, candidate_indices < population.size)
    return candidate_indices[in_population], in_population


def index_to_gid(population, index):
    if "gids" in population.annotations:
        return population.annotations["gids"][index]
    # the following rely on gids being sequential
    elif "first_gid" in population.annotations:
        offset = population.annotations["first_gid"]
    else:
        offset = population.first_id
    return index + offset


class SynCellFileConnector(Connector):
    """

    """
    parameter_names = ('path',)

    def __init__(self, path, safe=True, callback=None):
        self.path = path
        f = h5py.File(path, "r")
        
        self.presyn_params = f['presyn']['default']
        self.postsyn_params = f['postreceptors']['default']

    def connect(self, projection):
        mask2 = self.presyn_params["receptor_ids"].value == int(projection.receptor_type)
        n_connections = 0
        for post_index in projection.post._mask_local.nonzero()[0]:
            post_gid = index_to_gid(projection.post, post_index)
            # could process file by chunks, if memory becomes a problem for these masks
            mask1 = self.presyn_params["post_gid"].value == post_gid
            mask = mask1 * mask2
            pre_gids = self.presyn_params["pre_gid"][mask]
            pre_indices, in_population = gid_to_index(projection.pre, pre_gids)
            if pre_indices.size > 0:
                #print("****", post_index, post_gid, projection.pre.label, self.presyn_params["pre_gid"][mask], pre_indices)
                connection_parameters = {}
                for name in ('U', 'delay', 'tau_fac', 'tau_rec', 'weight'):
                    # todo: handle w_corr
                    if name == "tau_fac":   # hack - todo: proper translation
                        tname = "tau_facil"
                    else:
                        tname = name
                    connection_parameters[name] = self.presyn_params[tname][mask][in_population]
                # now translate names and units

                # create connections
                projection._convergent_connect(pre_indices, post_index, **connection_parameters)
                n_connections += pre_indices.size
        projection._size = n_connections  # optimization, to avoid calling GetConnections