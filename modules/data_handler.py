import os
import random
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from typing import List, Dict
from recordclass import recordclass

TrajectoryBatch = recordclass('TrajectoryBatch',
                        ('seq_lens', 'rec_mask', 'rec_lists', 'clicks', 'user_feat', 'serp_feat', 'inter_feat', 'relevances'))

class MyIter():
    def __init__(self, paths, data = None, device = "cpu"):
        self.paths = paths
        self.outer_iter = iter(self.paths)
        self.inner_iter = iter([])
        self.single_value = False
        self.device = device
        if data is not None:
            self.data = data
            self.single_value = True

    
    def __iter__(self):
        return self
    
    def load_data(self, path):
        if not self.single_value:
            self.data = torch.load(path, map_location = torch.device(self.device))
        self.inner_iter = iter(self.data.values())
    
    def __next__(self):
        try:
            sample = next(self.inner_iter)
        except:
            self.load_data(next(self.outer_iter))
            sample = next(self.inner_iter)
        return sample

class LoggedClickIterableDataset(torch.utils.data.IterableDataset):
    '''
        Pytorch dataset class for a dataset of logged interaction.
    '''
    def __init__(self, paths, device):
            '''
                Initialization

                Parameters :
                - paths : list(string)
                    List of paths to dataset_file
                
            '''
            self.paths = paths
            self.data = None
            self.device = device
            if len(paths) == 1:
                self.data = torch.load(self.paths[0], map_location = torch.device(device))

    def __iter__(self):
        '''
            Returns an iterator over the data
        '''
        return MyIter(self.paths, self.data, self.device)


def dynmod_collate_fn(batch, gpu = True):
    '''
        Collate function for Pytorch dataloader.

        Parameters : 
         - batch : list(Trajectory)
            Batch of trajectories
         - pad_sequences :  boolean
            If True, sequences are padded with zeros to all have same length and sequences are returned as one tensor 
            of size (max_seq_len, batch_size, *). If False, sequences are returned as a list(Tensor(size = (seq_len, *)), len = batch_size)
        
        Output : 
         - collated_batch : TrajectoryBatch
            Trajectory-like tuple where tensors are grouped in batches, ready to be used in pytorch models

    '''
    # Lengths of sequences (will be useful for packing)
    seq_lens = torch.LongTensor(list(map(lambda traj : len(traj["rec_lists"]), batch)))
    # Raw interactions
    rec_lists = [traj["rec_lists"] for traj in batch]
    rec_mask = [torch.clamp(rec_list.clone(), 0, 1) for rec_list in rec_lists]
    clicks = [traj["clicks"] for traj in batch]
    # Context
    rec_size = clicks[0].size()[1]
    if batch[0]["user_feat"] is None:
        user_features = [torch.FloatTensor(size = (seq_len, 0)) for seq_len in seq_lens]
    else :
        user_features = [traj["user_feat"] for traj in batch]
    if batch[0]["serp_feat"] is None:
        serp_features = [torch.FloatTensor(size = (seq_len, rec_size, 0)) for seq_len in seq_lens]
    else :
        serp_features = [traj["serp_feat"] for traj in batch]
    if batch[0]["inter_feat"] is None:
        inter_features = [torch.zeros(size = (seq_len, rec_size, 0)) for seq_len in seq_lens]
    else :
        inter_features = [traj["inter_feat"] for traj in batch]
    return TrajectoryBatch(seq_lens, rec_mask, rec_lists, clicks, user_features, serp_features, inter_features, None)


class ClickModelDataModule(pl.LightningDataModule):
    '''
        PyTorch Lightning data module for click model training and evaluation
    '''
    def __init__(self, train_paths : List[str], val_paths : List[str], test_paths : List[str], data_dir : str, 
                    batch_size : float, num_workers : int, device : str, debug: bool, val_check_interval : int,
                    test : bool, **kwargs):
        super().__init__()

        all_files = os.listdir(data_dir)
        if not debug and not test:
            if train_paths is None:
                train_paths = [data_dir + f for f in all_files if "train" in f]
                random.shuffle(train_paths)
            self.training_set = LoggedClickIterableDataset(train_paths, device)
            print("  -> Training dataset created.")
            if val_paths is None:
                val_paths = [data_dir + f for f in all_files if "val" in f]
            if val_check_interval > 1e6:
                self.validation_set = LoggedClickIterableDataset(["", ""], device)  # To gain time when we don't need validation checks
            self.validation_set = LoggedClickIterableDataset(val_paths, device)
            print("  -> Validation dataset created.")
        if not debug :
            if test_paths is None:
                test_paths = [data_dir + f for f in all_files if "test" in f]
        else:
            test_paths = [data_dir + f for f in all_files if "debug" in f]
        self.test_set = LoggedClickIterableDataset(test_paths, device)
        print("  -> Test dataset created.")

        self.batch_size = batch_size
        self.num_workers = num_workers

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_set, batch_size = self.batch_size, 
                                                collate_fn = dynmod_collate_fn,
                                                num_workers = self.num_workers, pin_memory = True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_set, batch_size = self.batch_size, 
                                                collate_fn = dynmod_collate_fn,
                                                num_workers = self.num_workers, pin_memory = True)
    

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size = self.batch_size, 
                                                collate_fn = dynmod_collate_fn,
                                                num_workers = self.num_workers, pin_memory = True)


def process_gt(path, filename, device, serp_feat_dim):
    '''
        Ground truth relevance labels loading and pre-processing

        Parameters :
         - path : str
            Path to directory where relevance labels are located
         - filename : str
            Filename of relevance labels
         - device : torch.device
            Device onto which we want to load the relevance labels
         - serp_feat_dim : int
            Dimension of SERP features (if applicable)
        
        Output :
         - shuffle_batches : list(TrajectoryBatch)
            Shuffled lists of documents for uncontextualized relevance estimation
         - batches : list(TrajectoryBatch)
            batches for NDCG2

    '''
    gt = torch.load(path + filename, map_location = device)  

    items = list(gt.values())
    n_items = len(items)
    random.shuffle(items)

    batches = []
    shuffle_batches = []
    for it in items:
        if it["clicks"] is None:
            it["clicks"] = torch.zeros_like(it["rec_lists"], device = device).long()
        if it["serp_feat"] is None:
            it["serp_feat"] = torch.zeros(len(it["rec_lists"]), len(it["rec_lists"][0]), serp_feat_dim, device = device).long()
        
        batches.append(TrajectoryBatch(torch.tensor([1]), [torch.ones(1, 10, device = device)], 
                                        **{ key : [val]  for key, val in it.items()}))

        randperm = torch.randperm(len(it["rec_lists"][0]))
        it["rec_lists"] = it["rec_lists"][:, randperm]
        it["relevances"] = it["relevances"][randperm]
        shuffle_batches.append(TrajectoryBatch(None, None, **{ key : [val]  for key, val in it.items()}))

    return shuffle_batches, batches

def get_file_name(cn : str, sl_cn : str, print_propensities : bool = False, discrete : bool = False, 
                    serp_feat_prop : bool = False, user_feat_prop : bool = False, all_serp_contexts : bool = False, 
                    relative : bool = False, kernel_size : int = 0, propensities_rel = None, 
                    stacked : bool = False, hidden_size : int = 0, absolute : bool = False,
                    propensities_abs = None, propensities = None, simplified : bool = False,
                    mode : str = "click", weighted : bool = False, normalized : bool = False,
                    smooth : float = 0.0, recurrent_type : str = "GRU", inner_state_dim : int = 0,
                    attention : bool = False, state_dim : int = 0, only_query : bool = False,
                    user_embedd_dim : List[int] = [0], embedd_dim : int = 0, seed : int = 0, 
                    serp_feat : bool = False, node2vec : bool = False, init_sdbn : bool = False, 
                    loss_per_rank : bool = False, rank_biased_loss : bool = False, non_causal : bool = False,
                    **kwargs):
    '''
        Get meaningful filename for saving model-specific outputs
    '''

    filename = cn
    if rank_biased_loss:
        filename += "_RBL"
    if cn in ['PBM', 'UBM', 'DBN', 'ARM']:
        if serp_feat_prop:
            filename += "_serpfeatprop"
        if user_feat_prop:
            filename += "_userfeatprop"
        if cn == 'ARM':
            if all_serp_contexts:
                filename += "_allserpcontexts"
            if relative:
                filename += "_relative_ker" + str(kernel_size)
                if print_propensities:
                    propensities = propensities_rel.weight
                    print("\n")
                    print("Relative propensities", propensities)
                    #torch.save(propensities, self.data_dir + "ARM_propensities_relative_" + str(self.relative) + ".pt")
            if stacked:
                filename += "_stacked" + str(hidden_size)
            if absolute:
                filename += "_absolute"
                if print_propensities:
                    propensities = propensities_abs.weight
                    print("\n")
                    print("Absolute propensities", propensities.squeeze())
                    #torch.save(propensities, self.data_dir + "ARM_propensities_absolute_" + str(self.absolute) + ".pt")
            if non_causal:
                filename += "_noncausal"
        else:
            if print_propensities:
                propensities = propensities.weight
                print("\n Propensities : ", torch.nn.Sigmoid()(propensities).squeeze())
                #torch.save(propensities, self.data_dir + "PBM_propensities_discrete_" + str(self.discrete) + ".pt")
            if cn == "DBN":
                if simplified:
                    filename += "_simplified"
                if init_sdbn:
                    filename += "_init-sdbn"
                if loss_per_rank:
                    filename += "rankweightedloss"
    elif cn == "TopPop":
        filename += "_" + mode
        if smooth:
            filename += "_smooth" + str(smooth)
        if weighted : 
            filename += "_weighted"
        if normalized :
            filename += "_normalized"
    elif cn in ["NCM", "AICM", "CACM"]:
        filename += "_" + recurrent_type + str(inner_state_dim)
        if attention:
            filename += "_attention"
        if serp_feat:
            filename += "_serpfeat"

    if sl_cn != "UserStateLearner":
        filename += "_" + sl_cn
    if sl_cn in ['ImmediateStateLearner', 'GRUStateLearner', 'CACMStateLearner']:
        if only_query:
            filename += "_onlyquery" + str(user_embedd_dim[0])
        else:
            filename += "_userstate" + str(state_dim)
    if sl_cn in ['GRUStateLearner', 'CACMStateLearner']:
        filename += "_GRU" + str(inner_state_dim)
        if attention:
            filename += "_attention"
    if sl_cn == "CACMStateLearner":
        if node2vec:
            filename += "_node2vec"

    if cn not in ["TopPop", "Random", "Oracle"] and sl_cn != "PairEmbeddingStateLearner":
        filename += "_itemembedd" + str(embedd_dim)
        
    filename += "_seed" + str(seed)

    return filename


def get_file_name_sim(sim_cn : str, pol_cn : str, distrib : str = None, cmp : Dict = None, epsilon : float = 0.0, 
                        T : float = 0.0, n_sess : int = 0, policy_name : str = None, sampled :bool = False, 
                        reverse : bool = False, frequencies : str = None, **kwargs):
    '''
        Get meaningful filename for saving the generated dataset
    '''
    filename = sim_cn

    if sim_cn == "SemiSyntheticSearchSim":
        filename += "_" + distrib
        if distrib == "fromfreq":
            filename += "_" + frequencies.split("/")[0]
        filename += "_" + cmp["type"] 
        if cmp["type"] == "DBN":
            filename += "alpha" + str(cmp["alpha"]) + "sigma" + str(cmp["sigma"]) + "gamma" + str(cmp["gamma"])
        elif cmp["type"] == "PBM":
            filename += "alpha" + str(cmp["alpha"]) + "gamma" + str(cmp["gamma"][1])
        elif cmp["type"] == "CoCM":
            filename += "_" + str(cmp["prob"][0]) + "_" + str(cmp["prob"][1]) + "_" + str(cmp["prob"][2])
        if sampled:
            filename += "_sampled"
    
    if sim_cn in ["SemiSyntheticSearchSim", "MultiModalSearchSim", "MultiModalRecSim"]:
        filename += ""
    
    filename += "_" + pol_cn[:-6]
    if pol_cn == "OraclePolicy":
        filename += "_epsilon" + str(epsilon)
    elif pol_cn in ["LoggedPolicy", "PlackettLuceLoggedPolicy"]:
        if len(policy_name) > 30:
            filename += "-" + policy_name.split("/")[-1][:15]
        else:
            filename += "-" + policy_name
        if pol_cn == "PlackettLuceLoggedPolicy":
            filename += "_temperature" + str(T)
        if reverse:
            filename += "_reverse"
    
    filename += "_" + str(n_sess)

    return filename
