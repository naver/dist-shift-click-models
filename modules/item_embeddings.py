import os

import pytorch_lightning as pl
import torch

from torch.nn import Embedding


class ItemEmbeddings(pl.LightningModule):
    '''
        Base Embedding class. It also serves as a one-hot embedding module.
    '''
    def __init__(self, num_items : int, embedd_dim : int, weights = None):
        super().__init__()

        self.embedd = Embedding(num_items, embedd_dim, _weight = weights)
    
    def forward(self, items):
        return self.embedd(items)

    @classmethod
    def from_pretrained(cls, data_dir, embedd_path):
        weights = torch.load(data_dir + embedd_path)
        num_items, embedd_dim = weights.size()
        return cls(num_items, embedd_dim, weights = weights)
    
    @classmethod
    def get_from_env(cls, env, data_dir : str, embedd_path :str):
        embedd_weights = env.get_item_embeddings()
        num_items, embedd_dim = embedd_weights.size()
        if not os.path.isfile(data_dir + embedd_path):
            torch.save(embedd_weights, data_dir + embedd_path)
        return cls(dimensions, weights = embedd_weights)
    
    @classmethod
    def from_scratch(cls, num_items : int, embedd_dim : int):
        return cls(num_items, embedd_dim)
    
    def clone_weights(self):
        return self.embedd.weight.clone()
