from typing import List
from recordclass import recordclass

import torch
import pytorch_lightning as pl
from torch.nn import Embedding, GRU, Linear, Softmax, LSTM, Identity

from modules.argument_parser import MyParser


TrajectoryBatch = recordclass('TrajectoryBatch',
                        ('seq_lens', 'rec_mask', 'rec_lists', 'clicks', 'user_feat', 'serp_feat', 'inter_feat'))

GTBatch = recordclass('GTBatch',
                        ('rec_lists', 'clicks', 'user_feat', 'serp_feat', 'inter_feat', 'relevances'))

class UserStateLearner(pl.LightningModule):
    '''
        Base class, identity funtion == no state learner. Returns only concatenated user features.
    '''
    def __init__(self, device : str, **kwargs):
        super().__init__()

        self.my_device = device
        self.only_query = True
        self.impression_based = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def forward(self, batch, h = None):
        '''
            Forward pass through state learner.

            Parameters : 
             - batch : TrajectoryBatch
                Batch of Trajectories

            Output :
             - s_u : torch.FloatTensor(size = (sum_seq_len, user_feat_dim))

        '''        
        return torch.cat(batch.user_feat, dim = 0)[:, 0]

class ImmediateStateLearner(UserStateLearner):
    '''  
        Simple Embedding-based immediate state learner (typically for query embeddings)
    '''
    def __init__(self, num_user_embedd : List[int], user_embedd_dim : List[int], state_dim : int, 
                    user_feat_dim : int, user_feat_type : str, only_query : bool, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.only_query = only_query
        self.user_feat_dim = user_feat_dim
        self.user_feat_type = user_feat_type
        self.num_user_embedd = num_user_embedd
        self.user_embedd_dim = user_embedd_dim

        if user_feat_dim == 0:
            raise ValueError("There are no user features in the dataset.")
        if user_feat_type == 'int':
            if only_query or user_feat_dim == 1:
                self.user_encoding = Embedding(num_user_embedd[0], state_dim)
            else:
                self.user_embeddings = [Embedding(num_user_embedd[i], user_embedd_dim[i]).to(
                                            self.my_device) for i in range(user_feat_dim)]
                self.lin_layer = Linear(torch.sum(torch.LongTensor(user_embedd_dim)), state_dim)
    
                self.user_encoding = lambda x : self.lin_layer(torch.cat([embeddings(x[:, i]) 
                                        for i,embeddings in enumerate(self.user_embeddings)], dim = 1))
        else:
            raise NotImplementedError("We only support categorical user features for now.")
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--user_embedd_dim', type=int, nargs='+', default = [64, 4, 8, 16])
        parser.add_argument('--only_query', type=parser.str2bool, default = False)
        return parser
    
    def forward(self, batch):
        '''
            Forward pass through state learner.
            Caution : requires batch.user_feat[k] to be a LongTensor of size (seq_len, *)

            Parameters : 
             - batch : TrajectoryBatch
                Batch of Trajectories

            Output :
             - s_u : torch.FloatTensor(size = (sum_seq_len, state_dim))

        '''     
        if self.only_query:
            user_inp = torch.cat(batch.user_feat, dim = 0)[:, 0]
        else:
            user_inp = torch.cat(batch.user_feat, dim = 0)
        return self.user_encoding(user_inp)

class PairEmbeddingStateLearner(UserStateLearner):
    '''  
        Simple Embedding-based immediate state learner (Only for Query/document embeddings)
        It requires to have pre-computed a dictionary {query, doc : embedding_index}
    '''
    def __init__(self, data_dir : str, **kwargs):
        super().__init__(**kwargs)

        self.pairs_in_data = torch.load(data_dir + "pair_embedding_dict.pt")

        self.user_embedd = Embedding(len(self.pairs_in_data) + 1, 1)
        self.impression_based = True
    
    def forward(self, batch, h = None):
        '''
            Forward pass through state learner.
            Caution : requires batch.user_feat[k] to be a LongTensor of size (seq_len, 1)

            Parameters : 
             - batch : TrajectoryBatch
                Batch of Trajectories

            Output :
             - s_u : torch.FloatTensor(size = (sum_seq_len, 1))

        '''       
        queries = torch.cat(batch.user_feat, dim = 0)[:, 0]
        items = torch.cat(batch.rec_lists, dim = 0)
        if self.my_device == "cpu":
            pair_idx = torch.LongTensor([[self.pairs_in_data[q.item() ,i.item()] for i in l]
                                                for (q,l) in zip(queries, items) ])
        else:
            pair_idx = torch.cuda.LongTensor([[self.pairs_in_data[q.item() ,i.item()] for i in l]
                                                for (q,l) in zip(queries, items) ])
        return self.user_embedd(pair_idx)


class RecurrentModule(torch.nn.Module):
    '''
        Generic recurrent module (contrary to click_models.RecurrentModule which is specially designed 
        for clik prediction head). That's basically a wrapper of torch.nn.GRU with advanced features.
    '''
    def __init__(self, recurrent_type : str, inner_state_dim : int, output : str, attention : bool,
                    num_layers_RNN : int, input_dim : int, device : torch.device, dropout_rate : float, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.device = device

        self.register_buffer("h0", torch.zeros(1, 1, inner_state_dim, device = device))
        self.recurrent_type = recurrent_type
        if recurrent_type == "LSTM":
            self.register_buffer("c0", torch.zeros(1, 1, inner_state_dim, device = device))
            self.rnn = LSTM(input_dim, hidden_size = inner_state_dim, num_layers = num_layers_RNN, batch_first = True)
            self.hidden_state = [self.h0, self.c0]
        elif recurrent_type == "GRU":
            self.rnn = GRU(input_dim, hidden_size = inner_state_dim, num_layers = num_layers_RNN, batch_first = True)
            self.hidden_state = self.h0
        else:
            raise NotImplementedError("This type of recurrent model has not been implemented yet.")
        
        self.output = output
        if output == "sigmoid":
            self.output_size = 1
            self.output_head = Sequential(Dropout(p=dropout_rate),
                                            Linear(inner_state_dim, 1),
                                                Sigmoid())
        elif output == "softmax":
            self.output_size = 2
            self.output_head = Sequential(Dropout(p=dropout_rate),
                                            Linear(inner_state_dim, 2),
                                                Softmax(dim = 1))
        elif output == "none":
            self.output_size = inner_state_dim
            self.output_head = Identity()
        else:
            raise NotImplementedError("This type of output for the recurrent module has not been implemented yet.")

        self.attention = attention 

        self.unpack = lambda packed_seq_obj: [tnsr[:ln] for tnsr, ln in zip(*torch.nn.utils.rnn.pad_packed_sequence(packed_seq_obj, batch_first=True))]

    def self_attention(self, h):
        '''
        h : torch.Tensor(seq_len, hidden_size)

        Output att : torch.tensor(seq_len, hidden_size)
        '''
        gram_matrices = torch.matmul(h, h.transpose(0,1)) # (seq_len, seq_len)
        alphas = Softmax(dim = 1)(gram_matrices) # (seq_len, seq_len)
        useful_alphas = torch.tril(alphas) # (seq_len, seq_len)
        expanded_h =  h.unsqueeze(0).expand(len(h), -1, -1)  # (seq_len, seq_len, hidden_size)
        ## We have expanded_h[*, i, :] = h_{i}
        unsqueezed = useful_alphas.unsqueeze(2) # (seq_len, seq_len, 1)
        expanded_h = unsqueezed * expanded_h # (seq_len, seq_len, hidden_size)
        att = torch.sum(expanded_h, dim = 1) # (seq_len, hidden_size) 

        return att

    def forward(self, inp, seq_lens):
        '''
            Forward pass through the recurrent module.

            Parameters :
             - inp : torch.FloatTensor(batch_size, max_seq_len, input_dim)
                Padded Input
             - seq_lens : torch.LongTensor(batch_size)
                Lengths of sequences
            
            Output :
             - out : torch.FloatTensor(batch_size, max_seq_len, output_size)
                Padded Output
        '''
        batch_size = len(inp)
        
        inp = torch.nn.utils.rnn.pack_padded_sequence(inp, lengths = seq_lens, batch_first = True, enforce_sorted = False)  # packed sequence

        if self.recurrent_type == "LSTM":
            hidden = tuple(map(lambda t : t.expand(-1, batch_size, -1).contiguous(), self.hidden_states))
        else:
            hidden = self.hidden_state.expand(-1, batch_size, -1).contiguous()

        out = self.unpack(self.rnn(inp, hidden)[0])    # List of tensors

        if self.attention:
            out = [self.self_attention(out_i) for out_i in out]
        
        return self.output_head(torch.cat(out, dim = 0))
        
class GRUStateLearner(ImmediateStateLearner):
    '''
        History-based state tracker using a Gated Recurrent Unit.
    '''
    def __init__(self, inner_state_dim : int, **kwargs):
        super().__init__(**kwargs)

        self.user_rnn = RecurrentModule("GRU", inner_state_dim = self.state_dim, output = "none", input_dim = self.state_dim, **kwargs)

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[ImmediateStateLearner.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--attention', type=parser.str2bool, default = True)
        parser.add_argument('--num_layers_RNN', type=int, default = 1)
        parser.add_argument('--inner_state_dim', type=int, default = 1)
        return parser

    def forward(self, batch):
        '''
            Forward pass through state learner.

            Parameters : 
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - h : None or torch.FloatTensor(size = (num_layers, batch_size, state_dim))
                Optional : Initial hidden state.

            Output :
             - s_u : torch.FloatTensor(size = (sum_seq_len, state_dim))

        '''
        try:
            batch.seq_lens
        except AttributeError:
            # i.e. we are extracting relevance
            new_batch = GTBatch(*batch)
            new_batch.user_feat = [batch.user_feat[0]]
            embedd = super().forward(new_batch).unsqueeze(0)
            seq_lens = torch.tensor([1])
        else:
            if self.only_query : 
                padded_seq = torch.nn.utils.rnn.pad_sequence(batch.user_feat, batch_first=True)[:, :, 0].unsqueeze(2)
            else:
                padded_seq = torch.nn.utils.rnn.pad_sequence(batch.user_feat, batch_first=True)
            padded_size = list(padded_seq.shape)[:2]
            new_batch = TrajectoryBatch(*batch)
            new_batch.user_feat = [padded_seq.flatten(end_dim = 1)]
            embedd = super().forward(new_batch).reshape(*padded_size, self.state_dim)
            seq_lens = batch.seq_lens
        
        return self.user_rnn.forward(embedd, seq_lens)

class CACMStateLearner(GRUStateLearner):
    '''
        Taken from Context-Aware Click Model [Chen et al., 2020]
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, node2vec : bool, num_items : int, embedd_dim : int, 
                    pos_embedd_dim : int, rec_size : int, num_serp_embedd : List[int], serp_embedd_dim : List[int], 
                    serp_feat_dim : int, serp_feat_type : str, click_embedd_dim : int, serp_state_dim : int, 
                    click_context_dim : int, inner_state_dim : int, **kwargs):
        super().__init__(inner_state_dim, **kwargs)

        self.impression_based = True

        ### Query Context Encoder
        self.node2vec = node2vec
        if node2vec :
            self.user_encoding = None # (à compléter)

        ### Click Context Encoder
        self.item_embedd_click = Embedding(num_items, embedd_dim, 
                                        _weight = item_embedd.clone_weights()).requires_grad_(True)

        self.pos_embedd_click = Embedding(rec_size, pos_embedd_dim)

        self.serp_state_dim = serp_state_dim
        self.serp_feat_dim = serp_feat_dim
        if serp_feat_dim == 0:
            raise ValueError("There are no serp features in the dataset.")
        if serp_feat_type == 'int':
            if serp_feat_dim == 1:
                self.serp_embedd_click = Embedding(num_serp_embedd[0], serp_state_dim).to(self.my_device)
            else:
                self.serp_embeddings = [Embedding(num_serp_embedd[i], serp_embedd_dim[i]).to(
                                            self.my_device) for i in range(serp_feat_dim)]
                self.lin_layer_serp = Linear(torch.sum(torch.LongTensor(serp_embedd_dim[:serp_feat_dim])), serp_state_dim)
    
                self.serp_embedd_click = lambda x : self.lin_layer_serp(torch.cat([embeddings(x[:, :, i]) 
                                        for i,embeddings in enumerate(self.serp_embeddings)], dim = 2))
        else:
            raise NotImplementedError("We only support categorical serp features for now.")

        self.click_embedd_click = Embedding(2, click_embedd_dim)

        self.click_context_model = RecurrentModule("GRU", click_context_dim, "none", 
                                        input_dim = embedd_dim + pos_embedd_dim + serp_state_dim + click_embedd_dim, **kwargs)
        self.click_context_dim = click_context_dim
 
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[GRUStateLearner.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--node2vec', type=parser.str2bool, default = False)
        parser.add_argument('--serp_state_dim', type=int, default = 32)
        parser.add_argument('--serp_embedd_dim', type=int, nargs = '+', default = [16, 16, 8])
        parser.add_argument('--click_embedd_dim', type=int, default = 4)
        parser.add_argument('--click_context_dim', type=int, default = 32)
        parser.add_argument('--pos_embedd_dim', type=int, default = 4)
        
        return parser

    def forward(self, batch):
        '''
            Forward pass through state learner. Be careful, this state learner return a user state after 
            each ~examination~, and not after each recommendation.

            Parameters : 
             - batch : TrajectoryBatch
                Batch of Trajectories.

            Output :
             - s_u : torch.FloatTensor(size = (sum_seq_len * rec_size, state_dim))

        '''
        rec_size = batch.rec_lists[0].size()[1]
        try :
            seq_lens = batch.seq_lens
            sum_seq_lens = torch.sum(batch.seq_lens)
            positions = torch.arange
            serp_feat = batch.serp_feat
            clicks = batch.clicks
        except AttributeError :
            # Relevance extraction
            seq_lens = torch.tensor([1])
            sum_seq_lens = 1
            positions = torch.zeros
            serp_feat = [torch.zeros(1, rec_size, self.serp_feat_dim, dtype = torch.long, device = self.device)]
            clicks = [torch.zeros(1, rec_size, dtype = torch.long, device = self.device)]
        
        ### 1 - Pass through Query encoder
        q_encoded = super().forward(batch)  # (sum_seq_lens, query_state_dim)

        ### 2 - Pass through Click Context Encoder
        item_embedd = self.item_embedd_click(torch.nn.utils.rnn.pad_sequence(batch.rec_lists, 
                                                batch_first=True).flatten(start_dim = 1, end_dim = 2))
        pos_embedd = self.pos_embedd_click(torch.nn.utils.rnn.pad_sequence([positions(rec_size, device = self.device, 
                                                    dtype = torch.long).unsqueeze(0).expand(len(sess), -1) for sess in batch.rec_lists], 
                                                batch_first=True).flatten(start_dim = 1, end_dim = 2))
        serp_embedd = self.serp_embedd_click(torch.nn.utils.rnn.pad_sequence(serp_feat, 
                                                batch_first=True).flatten(start_dim = 1, end_dim = 2)).squeeze(2)
        click_embedd = self.click_embedd_click(torch.nn.utils.rnn.pad_sequence(clicks, 
                                                batch_first=True).flatten(start_dim = 1, end_dim = 2))

        inp = torch.cat([item_embedd, pos_embedd, serp_embedd, click_embedd], dim = -1)
        

        c_encoded = self.click_context_model(inp, seq_lens * rec_size)   # (sum_seq_lens * rec_size, click_context_dim)
        q_encoded = q_encoded.unsqueeze(1).expand(-1, rec_size, -1).reshape(sum_seq_lens*rec_size, self.state_dim) 
                                                                    # (sum_seq_lens * rec_size, click_context_dim)

        
        return torch.cat([c_encoded, q_encoded], dim = 1)


