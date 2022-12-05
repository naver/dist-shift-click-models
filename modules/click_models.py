import random
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
from recordclass import recordclass

import torch
import pytorch_lightning as pl
from torchmetrics.functional.classification import auroc
from torch.nn import Sequential, GRU, LSTM, Linear, Dropout, ReLU, BCEWithLogitsLoss, Sigmoid, BCELoss, Tanh, Embedding, Softmax, MSELoss, Identity
from torch.nn.modules.sparse import Embedding
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.state_learners import UserStateLearner, ImmediateStateLearner, PairEmbeddingStateLearner, GRUStateLearner, CACMStateLearner
from modules.data_handler import process_gt, get_file_name
from modules.argument_parser import MyParser


TrajectoryBatch = recordclass('TrajectoryBatch',
                        ('seq_lens', 'rec_mask', 'rec_lists', 'clicks', 'user_feat', 'serp_feat', 'inter_feat'))



'''
    List of models in this file (Use CTRL-F):

    ClickModel      (Base class, simple feedforward click model)
    Random          (Outputs random click predictions and relevances)
    TopPop          (Popularity-based model, based on clicks, impressions or CTRs)
    PBM             (Position-Based Model, [Craswell et al., 2008])
    UBM             (User Browsing Model, [Dupret and Piwowarski, 2008])
    DBN             (Dynamic Bayesian Network, [Chapelle and Zhang, 2009])
    ARM             (Autoregressive Click Model)
    CACM            (Context-Aware Click Model, [Chen et al., 2020])
    CACM_minus           (Neural Examination-Hypothesis-Based Model)
    CoCM            (Complex Click Model, no training)
    NCM             (Neural Click Model, [Borisov et al., 2016])

'''


class ClickModel(pl.LightningModule):
    ''' 
        ClickModel base class. Implements a simple feedforward click model, where clicks 
        are predicted independantly for each rank and document.
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, embedd_dim : int, state_dim : int, rec_size : int, 
                    state_learner : str, lr : float, hidden_layers_cm : List[int], fine_tune_embedd : bool, 
                    data_dir : str, device : str, seed : int, exp_name: str, run_name: str, gen_ppl : bool, 
                    n_gen : int, ndcg_eval : bool, ndcg2_eval : bool, debug : bool, dropout_rate : float, 
                    weight_decay : float,serp_feat_dim : int, pair_dataset : bool, rank_biased_loss : bool, **kwargs):
        super().__init__()

        self.my_device = device
        self.data_dir = data_dir
        self.seed = seed
        self.exp_name = exp_name
        self.run_name = run_name
        self.gen_ppl = gen_ppl
        self.n_gen = n_gen
        self.debug = debug
        self.serp_feat_dim = serp_feat_dim
        self.pair_dataset = pair_dataset
        if pair_dataset:    ## This is for datasets where documents have a query-specific id
            self.pairs_in_data = torch.load(data_dir + "pair_embedding_dict.pt")
        self.rank_biased_loss = rank_biased_loss

        # General parameters
        self.embedd_dim = embedd_dim
        self.state_dim = state_dim
        self.rec_size = rec_size
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        # State learner
        if state_learner == "none" or state_learner == "UserStateLearner":
            self.state_learner = UserStateLearner(device = device, **kwargs)
        elif state_learner == "immediate" or state_learner == "ImmediateStateLearner":
            self.state_learner = ImmediateStateLearner(state_dim = state_dim, device = device, **kwargs)
        elif state_learner == "pair_embedding" or state_learner == "PairEmbeddingStateLearner":
            self.state_learner = PairEmbeddingStateLearner(device = device, data_dir = data_dir, **kwargs)
            self.state_dim = 1
        elif state_learner == "GRU" or state_learner == "GRUStateLearner":
            self.state_learner = GRUStateLearner(state_dim = state_dim, device = device, 
                                                    dropout_rate = dropout_rate, **kwargs)
        elif state_learner == "CACM" or state_learner == "CACMStateLearner":
            self.state_learner = CACMStateLearner(item_embedd, rec_size = rec_size, embedd_dim = embedd_dim, 
                                                    state_dim = state_dim, dropout_rate = dropout_rate, 
                                                    serp_feat_dim = serp_feat_dim, device = device, **kwargs)
            self.state_dim = state_dim + self.state_learner.click_context_dim
        else:
            raise NotImplementedError("The desired state tracker has not been implemented yet.")

        if not self.state_learner.only_query:
            self.ndcg_eval = False
        else:
            self.ndcg_eval = ndcg_eval
        self.ndcg2_eval = ndcg2_eval

        # Click model
        layers = []
        input_size = state_dim + embedd_dim + 1
        out_size = hidden_layers_cm[:]
        out_size.append(1)
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            if i != len(out_size) - 1:
                layers.append(ReLU())
        self.click_pred_head = Sequential(*layers)

        self.loss_fn = BCEWithLogitsLoss(reduction = 'none')
        if rank_biased_loss:
            phi = 0.92
            phi_pow_i_i = torch.tensor(phi, device = self.my_device).pow(torch.arange(self.rec_size, device = self.my_device)) / \
                                torch.arange(1, self.rec_size + 1, device = self.my_device)
            ref_phi = (1 - phi) / (1 - phi**self.rec_size)
            self.reduce_loss = lambda loss : ref_phi * torch.sum(phi_pow_i_i * torch.mean(loss, dim = 0))
        else:
            self.reduce_loss = torch.mean

        # Item embeddings
        self.item_embedd = item_embedd.requires_grad_(fine_tune_embedd)
        
    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        arguments = [action.option_strings[0] for action in parser._actions]
        parser.add_argument('--dropout_rate', type=float, default=0.2)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--fine_tune_embedd', type=parser.str2bool, default=True)
        parser.add_argument('--hidden_layers_cm', type=int, nargs = '+', default=[64, 16])
        parser.add_argument('--pair_dataset', type=parser.str2bool, default = False)
        parser.add_argument('--rank_biased_loss', type=parser.str2bool, default = False)
        # State learner
        if state_learner == "none" or state_learner == "UserStateLearner":
            parser = UserStateLearner.add_model_specific_args(parser)
        elif state_learner == "immediate" or state_learner == "ImmediateStateLearner":
            parser = ImmediateStateLearner.add_model_specific_args(parser)
        elif state_learner == "pair_embedding" or state_learner == "PairEmbeddingStateLearner":
           parser = PairEmbeddingStateLearner.add_model_specific_args(parser)
        elif state_learner == "GRU" or state_learner == "GRUStateLearner":
            parser = GRUStateLearner.add_model_specific_args(parser)
        elif state_learner == "CACM" or state_learner == "CACMStateLearner":
            parser = CACMStateLearner.add_model_specific_args(parser)
        else:
            raise NotImplementedError("The desired state learner has not been implemented yet.")
        return parser

    def forward(self, batch, h = None):
        '''
            Forward pass through the model.

            Parameters : 
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - h : None or torch.FloatTensor(size = (num_layers, batch_size, state_dim))
                Optional : Initial hidden state.

            Output :
             - click_pred : list(torch.LongTensor(size = (seq_len, rec_size)), len = batch_size)

        '''
        ### 1 - Pass through recurrent model
        s_u = self.state_learner.forward(batch, h = h)

        ### 2 - Pass through click prediction head
        click_pred = self.forward_click_pred_head(s_u, batch, return_loss = False)

        return click_pred
    
    def preprocess_input(self, s_u, batch, extend_su = True, collapse_seq = True):
        '''
            Batch preprocessing for click prediction. Not needed for pair_embedding state learner.

            Parameters :
             - s_u : torch.FloatTensor(sum_seq_lens, state_dim)
                Output of state learner
             - batch : TrajectoryBatch
                Current batch
             - extend_su : bool
                Whether we need to copy and paste s_u at each rank
             - collapse_seq : bool
                Whether the sequences of documents must be flattened into one batch
            
            Output :
             - s_u : torch.FloatTensor({sum_seq_lens or sum_seq_lens * rec_size}, state_dim)
                Pre-processed user state
             - rec_features : torch.FloatTensor({sum_seq_lens * rec_size or sum_seq_lens, rec_size}, state_dim)
                Pre-processed item embeddings

        '''
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)

        ### 1 - Pre-process input

        if self.pair_dataset:
            rec_lists = [torch.stack([torch.stack([torch.tensor(self.pairs_in_data[uf[t].item(), did.item()], device = self.device) for did in rl[t]]) 
                                        for t in range(len(rl))], dim = 0) for uf, rl in zip(batch.user_feat, batch.rec_lists)]
        else:
            rec_lists = batch.rec_lists

        if collapse_seq:
            # Get embedding for all recommended items and pad sequence
            # rec_features : list(LongTensor(seq_len, rec_size), len = batch_size) 
            #            --> LongTensor(sum_seq_len, rec_size)
            #            --> LongTensor(sum_seq_len * rec_size)
            #            --> FloatTensor(sum_seq_lens * rec_size, embedd_dim)
            rec_features = self.item_embedd(torch.cat([item_seq for item_seq in rec_lists], dim = 0).flatten())
        else:
            # Get embedding for all recommended items and pad sequence
            # rec_features : list(LongTensor(seq_len, rec_size), len = batch_size) 
            #            --> LongTensor(sum_seq_len, rec_size)
            #            --> FloatTensor(sum_seq_lens, rec_size, embedd_dim)
            rec_features = self.item_embedd(torch.cat([item_seq for item_seq in rec_lists], dim = 0))


        if extend_su:
            # Expand state
            # s_u : tensor(size = (sum_seq_lens, state_dim))
            #            --> tensor(size = (sum_seq_lens * rec_size, state_dim))
            s_u = s_u.unsqueeze(1).expand(-1, self.rec_size, -1).reshape(sum_seq_lens*self.rec_size, self.state_dim)

        return s_u, rec_features

    def preprocess_input_rels(self, s_u, batch):
        '''
            Batch preprocessing for relevance estimation. Not needed for pair_embedding state learner.

            Parameters :
             - s_u : torch.FloatTensor(sum_seq_lens, state_dim)
                Output of state learner
             - batch : TrajectoryBatch
                Current batch of one query to rank
            
            Output :
             - s_u : torch.FloatTensor(n_docs_in_query, state_dim)
                Pre-processed user state
             - rec_features : torch.FloatTensor(n_docs_in_query, embedd_dim)
                Pre-processed item embeddings

        '''
        if self.pair_dataset:
            rec_lists = [torch.stack([torch.stack([torch.tensor(self.pairs_in_data[uf[t].item(), did.item()], device = self.device) for did in rl[t]]) 
                                        for t in range(len(rl))], dim = 0) for uf, rl in zip(batch.user_feat, batch.rec_lists)]
        else:
            rec_lists = batch.rec_lists

        # Get embedding for all recommended items and pad sequence
        # rec_features : list(Longtensor(n_docs_in_query), len = 1) 
        #            --> LongTensor(n_docs_in_query)
        #            --> FloatTensor(n_docs_in_query, embedd_dim))
        rec_features = self.item_embedd(torch.cat([item_seq[-1] for item_seq in rec_lists]))

        s_u = s_u.unsqueeze(0).expand(len(rec_features), -1)

        return s_u, rec_features

    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        '''
            Forward pass through the click prediction head.

            Parameters : 
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Flattenned batch of user states.
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - return_loss : boolean
                If True, the click probability and the loss are returned, and if False, the sampled predicted clicks are returned

            
            Output : 
             - click_pred : list(torch.{Long or Float}Tensor(size = (seq_len, rec_size)), len = batch_size)
                Predicted clicks or click probability (see above)
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)

        ### 1 - Pre-process input
        s_u, rec_features = self.preprocess_input(s_u, batch)

        # Create rank variable : tensor(size = (sum_seq_lens * rec_size, 1))
        rank = torch.arange(self.rec_size, device=self.device).unsqueeze(1).repeat(sum_seq_lens, 1)

        # Concatenate to obtain input and convert to float
        inp = torch.cat([s_u, rec_features, rank], dim = 1).float()

        ### 2 - Pass through model
        click_pred = self.click_pred_head(inp).reshape(sum_seq_lens, self.rec_size)

        ### 3 - Reshape and return

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens * rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])
        click_pred_list = [Sigmoid()(click_pred[cum_lens[i]:cum_lens[i+1]]) for i in range(len(cum_lens) - 1)]

        if return_loss:   
            # Get targets         
            target_click = torch.cat(batch.clicks, dim = 0).float()
            #rec_mask = torch.cat(batch.rec_mask, dim = 0)
            
            # Compute loss
            #loss = self.loss_fn(click_pred * rec_mask, target_click) / torch.sum(rec_mask)
            loss = self.loss_fn(click_pred, target_click)

            return click_pred_list, loss

        else:
            # Bernoulli experiment for click generation
            return [torch.bernoulli(seq for seq in click_pred_list)]

    def training_step(self, batch, batch_idx):
        '''
            Training step.

            Parameters : 
             - batch : TrajectoryBatch
                Current batch of trajetories
             - batch_idx : int
                Batch index (unused)
            
            Output : 
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''
        ### 1 - Pass through recurrent model
        s_u = self.state_learner.forward(batch)

        ### 2 - Pass through click prediction head
        _, loss = self.forward_click_pred_head(s_u, batch, return_loss = True)
        
        mean_nll = self.reduce_loss(loss)
        self.log('train_loss', mean_nll)
        return mean_nll
    
    def training_epoch_end(self, training_step_outputs):
        mean_loss = torch.tensor([0]) 
        if self.automatic_optimization:
            mean_loss = torch.mean(torch.cat([op['loss'].unsqueeze(0) for op in training_step_outputs]))
        print("\n Epoch %d : mean_train_loss = %.3f" % (self.current_epoch, mean_loss.item()))
    
    def validation_epoch_end(self, validation_step_outputs):
        mean_loss = torch.mean(torch.cat([op.unsqueeze(0) for op in validation_step_outputs]))
        if not self.automatic_optimization:
            scheduler = self.lr_schedulers()
            scheduler.step(mean_loss)
        print("\n Val check : mean_val_loss = %.3f" % (mean_loss.item()))

    def validation_step(self, batch, batch_idx):
        '''
            Validation step.

            Parameters : 
             - batch : TrajectoryBatch
                Current batch of trajetories
             - batch_idx : int
                Batch index (unused)
            
            Output : 
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''
        ### 1 - Pass through recurrent model
        s_u = self.state_learner.forward(batch)

        ### 2 - Pass through click prediction head
        click_pred, loss = self.forward_click_pred_head(s_u, batch, return_loss = True)  
           
        mean_loss, sum_seq_lens = self.aggregate_nll(loss, batch.rec_mask)
        self.log('my_val_loss', mean_loss)
        return mean_loss

    def test_step(self, batch, batch_idx):
        '''
            Test step.

            Parameters : 
             - batch : TrajectoryBatch
                Current batch of trajetories
             - batch_idx : int
                Batch index (unused)
            
        '''
        if batch_idx < 10 or not self.debug:    ### In debug mode, we only test on the first 10 batches
            ### 1 - Pass through recurrent model
            s_u = self.state_learner.forward(batch)
            
            ### 2 - Pass through click prediction head
            click_pred, loss = self.forward_click_pred_head(s_u, batch, return_loss = True)
            click_pred_cat, click_true_cat = torch.cat(click_pred, dim=0), torch.cat(batch.clicks, dim=0)

            ### 3 - Metrics computation
            # Compute loss
            mean_loss, sum_seq_lens = self.aggregate_nll(loss, batch.rec_mask)
            # Compute perplexity
            perplexity, rec_mask_sum = self.aggregate_nll(loss / torch.log(2 * torch.ones_like(loss)), 
                                                                        batch.rec_mask, per_rank = True)

            if self.gen_ppl:    # For Reverse and Forward PPL (as in AICM)
                return [batch.rec_lists, click_pred, batch.user_feat, (mean_loss, perplexity, rec_mask_sum, sum_seq_lens)]
            else:
                return [(mean_loss, perplexity, rec_mask_sum, sum_seq_lens, click_pred_cat, click_true_cat)]
          
    def aggregate_nll(self, nll, rec_mask, per_rank = False):
        rec_mask_cat = torch.cat(rec_mask, dim = 0)
        sum_seq_lens = len(rec_mask_cat)
        rec_mask_sum = torch.sum(rec_mask_cat, dim = 0)
        
        masked_nll = nll * rec_mask_cat


        if per_rank:
            return torch.sum(masked_nll, dim = 0) / rec_mask_sum, rec_mask_sum
        else:
            return torch.sum(masked_nll) / torch.sum(rec_mask_sum), sum_seq_lens

    def ndcg(self, pred_rel, ranks = [1, 2, 3, 4, 5]):
        ''' 
            nDCG computation.

            Parameters :
             - pred_rel : torch.LongTensor(n_docs_in_query)
                True relevances of documents ordered according to the click model.
             - ranks : list(int)
                Cutoff ranks
        
        '''
        batch_size = len(pred_rel)
        sorted_rel, _ = torch.sort(pred_rel, descending = True)

        p = len(pred_rel)
        propensities = torch.log2(torch.arange(2, p + 2, dtype = torch.float, device = self.device))
        n_ranks = len(ranks)
        ndcg = torch.empty(n_ranks, device = self.device)
        dcg = (2 ** pred_rel - 1) / propensities
        idcg = (2 ** sorted_rel - 1) / propensities
        prev_r = 0
        dcg_r = torch.empty(n_ranks, device = self.device)
        idcg_r = torch.empty(n_ranks, device = self.device)
        for i,r in enumerate(ranks):
            if r == -1:
                dcg_r[i] = torch.sum(dcg[prev_r:])
                idcg_r[i] = torch.sum(idcg[prev_r:])
            else:
                dcg_r[i] = torch.sum(dcg[prev_r:r])
                idcg_r[i] = torch.sum(idcg[prev_r:r])
            prev_r = r
        
        ndcg = torch.cumsum(dcg_r, dim = 0) / torch.cumsum(idcg_r, dim = 0)
        
        return ndcg

    def extract_relevances(self, s_u, batch):
        '''
        Extracts uncontextualized relevances for NDCG evaluation.

        Parameters :
         - s_u : FloatTensor(n_docs_in_query, state_dim)
            Pre-processed user state
         - batch : TrajectoryBatch
            Current batch of one query to be ranked
        
        Output :
         - attr_pred_list : list(FloatTensor(n_docs_in_query), len = batch_size)

        '''
        ### 1 - Pre-process input
        s_u, rec_features = self.preprocess_input_rels(s_u, batch)

        rank = torch.zeros(len(rec_features), 1, device = self.device)

        ### 2 - Pass through attractiveness model
        inp = torch.cat([s_u, rec_features, rank], dim = 1).float()
        attr_pred = self.click_pred_head(inp).squeeze()  
        
        return attr_pred

    def test_epoch_end(self, outputs):
        '''
            Relevance estimation and logging of results.

            Parameters : 
             - outputs : None or list(test_outputs)
                List of outputs from test_step

        '''

        mean_ndcg = torch.empty(0, device = self.device)    # NDCG on uncontextualized relevance estimation
        mean_ndcg2 = torch.empty(0, device = self.device)   # NDCG2 is computed on contextual relevance estimation (only relevant on Tiangong-ST)
        if self.ndcg_eval:
            ranks = [1, 3, 5, 10] # k for NDCG@k

            # Loading the ground truth and organizing it into batches :
            batches, batches2 = process_gt(self.data_dir, "ground_truth_rerank.pt", self.device, self.serp_feat_dim)
                # batches of length 1
            
            ndcgs = torch.empty(len(batches), len(ranks), device = self.device)
            recalls = torch.empty(len(batches), len(ranks), device = self.device)
            if self.ndcg2_eval:
                ndcgs2 = torch.empty(len(batches), len(ranks), device = self.device)

            relevance_scores = {}
            for b, (batch, batch2) in enumerate(zip(batches, batches2)):
                if torch.max(batch.relevances[0]) > 0: ### to avoid idcg of 0
                    # Relevance computation :
                    s_u = self.state_learner.forward(batch) # FloatTensor(state_dim)

                    if self.state_learner.impression_based:
                        s_u = s_u[- batch.rec_lists[0].size()[1]:]
                    else:
                        s_u = s_u[-1]
                    query_pred = self.extract_relevances(s_u, batch) # FloatTensor(n_doc_in_query)

                    if self.ndcg2_eval:
                        if not self.state_learner.impression_based:
                            s_u = s_u.unsqueeze(0)
                        query_pred2 = self.forward_click_pred_head(s_u, batch2)[0][0].squeeze()
                    

                    # Order predicitions by decreasing relevance :
                    sorted_pred = torch.argsort(query_pred, descending = True)     # LongTensor(n_doc_in_query)
                    pred_rel = batch.relevances[0][sorted_pred]
                    # Compute NDCG :
                    ndcgs[b] = self.ndcg(pred_rel, ranks) # FloatTensor(n_doc_in_query)
                    # Compute recall :
                    binary = True
                    if binary:
                        rel_thresh = 0
                    else:
                        rel_thresh = 2
                    n_rel = torch.cumsum(torch.where(pred_rel>rel_thresh, torch.ones_like(pred_rel), torch.zeros_like(pred_rel)), dim = 0)
                    
                    if n_rel[-1] == 0:
                        recalls[b] = torch.ones(len(ranks), device = self.device)
                    else:
                        recalls[b] = (n_rel / n_rel[-1])[torch.clamp(torch.tensor(ranks, device = self.device) - 1 ,0, len(pred_rel) - 1)]
                    
                    if self.ndcg2_eval:
                        # Order predicitions by decreasing relevance :
                        sorted_pred2 = torch.argsort(query_pred2, descending = True)     # LongTensor(n_doc_in_query)
                        pred_rel2 = batch.relevances[0][sorted_pred2]
                        ndcgs2[b] = self.ndcg(pred_rel2, ranks)
                    
                else:
                    ndcgs[b] = torch.ones(len(ranks), device = self.device) # NDCG is 1 when IDCG is 0
                    recalls[b] = torch.ones(len(ranks), device = self.device)
                    if self.ndcg2_eval:
                        ndcgs2[b] = torch.ones(len(ranks), device = self.device)
            
            mean_ndcg = torch.mean(ndcgs, dim = 0)
            mean_recall = torch.mean(recalls, dim = 0)
            if self.ndcg2_eval:
                mean_ndcg2 = torch.mean(ndcgs2, dim = 0)
               

        # Save relevance and metric scores with meaningful filename :
        cn = self.__class__.__name__
        sl_cn = self.state_learner.__class__.__name__
        args_dict = {**self._modules, **self.__dict__, **self.state_learner.__dict__, **self.state_learner._modules}
        filename = get_file_name(cn = cn, sl_cn = sl_cn, print_propensities = False, **args_dict)

        # Save metrics
        res_path = self.data_dir + "results/"
        Path(res_path).mkdir(parents=True, exist_ok=True)
        res = {}
        sum_seq_lens = torch.tensor([op[-1][3] for op in outputs], device = self.device)
        perplexities = torch.stack([op[-1][1] for op in outputs])

        # Compute AUROC
        click_pred = torch.cat([op[-1][4] for op in outputs], dim=0)
        click_true = torch.cat([op[-1][5] for op in outputs], dim=0)
        auc = torch.stack([auroc(click_pred[:, i], click_true[:,i], pos_label = 1) for i in range(click_true.shape[1])])
        auc_whole = auroc(click_pred.flatten(), click_true.flatten(), pos_label = 1)

        rec_mask_sum = torch.stack([op[-1][2] for op in outputs])
        mean_perplexities = 2 ** (torch.sum(perplexities * rec_mask_sum, dim = 0) / torch.sum(rec_mask_sum, dim = 0))
        avg_perplex = torch.mean(mean_perplexities)
        perplex_ranks = [1, 2, 5, 10]
        for i, r in enumerate(perplex_ranks):
            res["Perplexity@" + str(r)] = mean_perplexities[r-1].item()
            self.log("test_perplexity@" + str(r), mean_perplexities[r-1])
            res["AUROC@" + str(r)] = auc[r-1].item()
            self.log("test_AUROC@" + str(r), auc[r-1])
        res["Perplexity"] = avg_perplex.item()
        self.log("test_perplexity", avg_perplex)
        res["AUROC"] = auc_whole.item()
        self.log("test_AUROC", auc_whole)

        loss = torch.tensor([op[-1][0] for op in outputs], device = self.device).unsqueeze(0)
        mean_loss = torch.sum(loss * sum_seq_lens) / torch.sum(sum_seq_lens)
        self.log("test_loss", mean_loss)
        res["NLL"] = mean_loss.item()

        if self.ndcg_eval:
            ndcg_ranks = [1,3,5,10]
            for i, r in enumerate(ndcg_ranks):
                res["NDCG@" + str(r)] = mean_ndcg[i].item()
                self.log("test_NDCG@" + str(r), mean_ndcg[i])
                res["Recall@" + str(r)] = mean_recall[i].item()
                self.log("test_recall@" + str(r), mean_recall[i])
                if self.ndcg2_eval:
                    res["NDCG*@" + str(r)] = mean_ndcg2[i].item()
                    self.log("test_NDCG*@" + str(r), mean_ndcg2[i])
        
        torch.save(res, res_path + filename + '.pt')

        # For Reverse and Forward PPL
        if self.gen_ppl:
            batch_size = len(outputs[0][0])
            wl_path = self.data_dir + "gen_data/"
            Path(wl_path).mkdir(parents=True, exist_ok=True)
            gen_dataset = []
            for k in range(self.n_gen + 1):
                gen_dataset.append({(k, i*batch_size + j) :  {"rec_lists" : op[0][j], 
                                            "clicks" : torch.bernoulli(op[1][j]),
                                            "user_feat" : op[2][j][:, 0].unsqueeze(1),
                                            "serp_feat" : None, 
                                            "inter_feat" : None} for i,op in enumerate(outputs) for j in range(len(op[0]))})
            gen_dataset = [it for d in gen_dataset for it in d.items()]
            n_data = len(gen_dataset)
            random.shuffle(gen_dataset)
            val_dataset = dict(gen_dataset[:n_data // 10])
            train_dataset = dict(gen_dataset[n_data // 10:])
            torch.save(val_dataset, wl_path + filename + '_val.pt')
            torch.save(train_dataset, wl_path + filename + '_train.pt')

        self.log(filename, 0, logger = False)    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(optimizer, factor = 0.5, patience = 2),
                'monitor': 'my_val_loss',
                }

    def get_policy(self, top_down = False):
        '''
            Returns the policy produced by the click model.

            Parameters :
             - top_down : bool
                If True, ranks documents by decreasing. If False, rank documents for reward maximization.

            Output :
             - policy : dict{ int : torch.LongTensor(self.rec_size)}
                Policy produced by the click model
             - docs_per_query : dict{ int : torch.LongTensor(n_docs_per_query)}
                All documents in the data
             - relevances : dict{ int : torch.LongTensor(n_docs_per_query)}
                Associated relevances

        '''
        if not self.pair_dataset:
            self.pairs_in_data = torch.load(self.data_dir + "pair_embedding_dict.pt")
        
        docs_per_query = {}
        for q, doc in self.pairs_in_data.keys():
            if q in docs_per_query:
                docs_per_query[q].append(doc)
            else:
                docs_per_query[q] = [doc]

        relevances = {}
        policy = {}
        expected_CTR = {}
        with torch.inference_mode():
            for q, doc_list in tqdm(docs_per_query.items(), total = len(docs_per_query)):
                docs_per_query[q] = torch.tensor(doc_list, device = self.device)[torch.randperm(len(doc_list))]
                batch = TrajectoryBatch(torch.tensor([1]), None, [docs_per_query[q].unsqueeze(0)], None, [torch.tensor([[q]], device = self.device)], None, None)
                
                s_u = self.state_learner.forward(batch) # FloatTensor(state_dim)
                if self.state_learner.impression_based:
                    s_u = s_u[- batch.rec_lists[0].size()[1]:]
                else:
                    s_u = s_u[-1]
                relevances[q] = self.extract_relevances(s_u, batch) # FloatTensor(n_doc_in_query)

                if top_down:
                    rels, doc_idx = torch.sort(relevances[q], descending = True)
                    rels = rels[:self.rec_size]
                    policy[q] = docs_per_query[q][doc_idx[:self.rec_size]]
                else:
                    policy[q], rels = self.max_reward_policy(q, docs_per_query[q], relevances[q])

                
                batch.rec_lists[0] = policy[q].unsqueeze(0)
                expected_CTR[q] = self.compute_reward(s_u, batch, rels) / self.rec_size
                # print(q)
                # print(policy[q])
                # print(rels)
                # print(expected_CTR[q])

        return policy, docs_per_query, relevances, expected_CTR
    
    def max_reward_policy(self, query, docs, rels):
        '''
            Returns a policy maximizing CTR, according to the click model

            Parameters :
             - query : int
                Current query
             - docs : torch.LongTensor(n_docs_in_q)
                List of documents associated with this query
             - rels : torch.LongTensor(n_docs_in_q)
                Relevances of documents in docs
            
            Output : 
             - policy : torch.LongTensor(self.rec_size)
                Maximum reward policy
        '''
        raise NotImplementedError("The MaxReward policy has not been implemented for Feedforward Click Model.")

class RandomClickModel(ClickModel):
    '''
        Click model that returns random clicks and relevances.
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, **kwargs):
        super().__init__(item_embedd, **kwargs)
        self.loss_fn = self.loss_fn = BCELoss(reduction = 'none')
    
    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        
        click_pred_list = [torch.rand(seq_len, len(batch.rec_lists[0][0]), device = self.device) for seq_len in batch.seq_lens]
        click_pred = torch.cat(click_pred_list, dim = 0)

        if return_loss:
            # Get targets       
            if batch.clicks is not None : 
                target_click = torch.cat(batch.clicks, dim = 0).float()
                loss = self.loss_fn(click_pred, target_click)
            # Compute loss
            return click_pred_list, loss

        else:
            return [torch.bernoulli(seq for seq in click_pred)]
    
    def extract_relevances(self, s_u, batch):
        return torch.rand(len(batch.rec_lists[0][-1]), device = self.device)

class TopPop(ClickModel):
    '''
        Popularity-based models, such as TopPop, TopObs, dCTR, drCTR, etc.
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, mode : str, smooth : bool, weighted : bool, 
                    normalized : bool, input_dir : str, **kwargs):
        super().__init__(item_embedd, **kwargs)
        self.loss_fn = self.loss_fn = BCELoss(reduction = 'none')
        self.mode = mode    # click, obs, CTR, product, or reverse_CTR
        self.smooth = smooth
        self.weighted = weighted
        self.normalized = normalized
        if input_dir is None:
            path = self.data_dir
        else:
            path = input_dir

        if self.weighted:   # rank-weighted models
            self.click_freq = torch.load(path + "click_freq_rank.pt")
            self.obs_freq = torch.load(path + "obs_freq_rank.pt")

            obs_by_rank = [[val for key, val in self.obs_freq.items() if key[2] == rank] 
                                                            for rank in range(self.rec_size)]
            n_obs_by_rank = torch.cat([torch.sum(torch.tensor(o_r, device = self.my_device)).unsqueeze(0) for o_r in obs_by_rank])

            clicks_by_rank = [[val for key, val in self.click_freq.items() if key[2] == rank] 
                                                        for rank in range(self.rec_size)]
            n_clicks_by_rank = torch.cat([torch.sum(torch.tensor(c_r, device = self.my_device)).unsqueeze(0) for c_r in clicks_by_rank])

            if self.mode == "reverse_CTR":
                self.weights = n_obs_by_rank / n_clicks_by_rank
            else:
                self.weights = n_clicks_by_rank / n_obs_by_rank
        else:
            self.click_freq = torch.load(path + "click_freq.pt")
            self.obs_freq = torch.load(path + "obs_freq.pt")



            self.weights = torch.ones(self.rec_size, device = self.my_device)
        
        if self.smooth:     # Prior computation for Bayesian smoothing
            if self.mode not in ["CTR", "reverse_CTR"]:
                raise ValueError("Mode must be CTR or reverse CTR to enable smoothing.")

            if self.weighted:
                self.prior = self.weights
            else:
                n_clicks = torch.sum(torch.tensor(list(self.click_freq.values())))
                n_obs = torch.sum(torch.tensor(list(self.obs_freq.values())))
                if self.mode == "reverse_CTR":
                    self.prior = n_obs / n_clicks * torch.ones(self.rec_size, device = self.my_device)
                else:
                    self.prior = n_clicks / n_obs * torch.ones(self.rec_size, device = self.my_device)
            
            print("Prior : ", self.prior)
        else:
            self.prior = torch.zeros(self.rec_size, device = self.my_device)
        
        if self.normalized and not self.weighted:
            raise ValueError("The model must be weighted to be normalized")

    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[ClickModel.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        parser.add_argument('--mode', type=str, choices=['obs', 'click', 'CTR', 'reverse_CTR', 'product'], default = 'click')
        parser.add_argument('--smooth', type=float, default = 0.0)
        parser.add_argument('--weighted', type=parser.str2bool, default = False)
        parser.add_argument('--normalized', type=parser.str2bool, default = False)
        parser.add_argument('--input_dir', type=str, default = None)
        return parser
    
    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        queries = torch.cat(batch.user_feat, dim = 0)[:, 0]
        items = torch.cat(batch.rec_lists, dim = 0)

        pop = torch.zeros_like(items, device = self.device).float()
        for k,q in enumerate(queries):
            q_id = q.item()
            for j, d in enumerate(items[k]):
                d_id = d.item()
                if self.weighted:
                    key = (q_id, d_id, j)
                else:
                    key = (q_id, d_id)
                if key in self.click_freq:
                    if self.mode == "CTR":
                        if self.smooth:
                            pop[k,j] = (self.click_freq[key] + self.smooth) / (self.obs_freq[key] + self.smooth / self.prior[j] )
                        else:
                            pop[k,j] = self.click_freq[key] / self.obs_freq[key]
                    elif self.mode == "click":
                        pop[k,j] = self.click_freq[key]
                    elif self.mode == "obs":
                        pop[k,j] = self.obs_freq[key]
                else:
                    pop[k,j] = self.prior[j]

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens * rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])
        click_pred_list = [pop[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]

        if return_loss:   
            # Get targets   
            pop = torch.clamp(pop, 1e-6, 1 - 1e-6)  
            loss = 0   
            if batch.clicks is not None : 
                target_click = torch.cat(batch.clicks, dim = 0).float()
                loss = self.loss_fn(pop, target_click)

            return click_pred_list, loss
        else:
            # Bernoulli experiment for click prediction
            return [torch.bernoulli(seq for seq in click_pred_list)]
    
    def extract_relevances(self, s_u, batch):

        q_id = batch.user_feat[0][-1, 0].item()
        docs = batch.rec_lists[0][-1]
        qp = torch.empty(len(docs))

        for i,d in enumerate(docs):
            d_id = d.item()
            if self.weighted:
                if self.mode == "obs":
                    quantity_r = torch.tensor([self.obs_freq[q_id, d_id, r]  if (q_id, d_id, r) in self.obs_freq else 0 
                                            for r in range(self.rec_size)], device = self.device)
                elif self.mode == "click":
                    quantity_r = torch.tensor([self.click_freq[q_id, d_id, r]  if (q_id, d_id, r) in self.click_freq else 0 
                                            for r in range(self.rec_size)], device = self.device)
                elif self.mode == "CTR":
                    if self.smooth > 0:
                        quantity_r = torch.tensor([(self.click_freq[q_id, d_id, r] + self.smooth) / \
                                                (self.obs_freq[q_id, d_id, r] + self.smooth / self.prior[r]) if (q_id, d_id, r) in self.click_freq else self.prior[r]
                                                for r in range(self.rec_size) ], device = self.device)
                    else:
                        quantity_r = torch.tensor([self.click_freq[q_id, d_id, r] / self.obs_freq[q_id, d_id, r] 
                                                if (q_id, d_id, r) in self.click_freq else 0
                                                for r in range(self.rec_size) ], device = self.device)
                elif self.mode == "reverse_CTR":
                    if self.smooth > 0:
                        quantity_r = torch.tensor([ - (self.obs_freq[q_id, d_id, r] + self.smooth) / \
                                                (self.click_freq[q_id, d_id, r] + self.smooth / self.prior[r]) if (q_id, d_id, r) in self.click_freq else - self.prior[r]
                                                for r in range(self.rec_size) ], device = self.device)
                    else:
                        quantity_r = torch.tensor([ - self.obs_freq[q_id, d_id, r] / self.click_freq[q_id, d_id, r] 
                                                if (q_id, d_id, r) in self.click_freq and self.click_freq[q_id, d_id, r] != 0 else -1e6
                                                for r in range(self.rec_size) ], device = self.device)
                else:
                    quantity_r = torch.tensor([self.obs_freq[q_id, d_id, r] * self.click_freq[q_id, d_id, r]  if (q_id, d_id, r) in self.obs_freq else 0 
                                            for r in range(self.rec_size)], device = self.device)

                    
                if torch.sum(quantity_r) == 0:
                    qp[i] = 0
                else:
                    if self.normalized:
                        quantity_r = quantity_r / torch.sum(quantity_r)

                    if self.mode == "obs":
                        qp[i] = torch.sum(quantity_r * self.weights)
                    else:
                        qp[i] = torch.sum(quantity_r / self.weights)

            else:
                if (q_id, d_id) in self.click_freq:
                    if self.mode == "CTR" :
                        if self.smooth > 0:
                            qp[i] = (self.click_freq[q_id, d_id] + self.smooth) / \
                                            (self.obs_freq[q_id, d_id] + self.smooth / self.prior[0] )
                        else:
                            qp[i] = self.click_freq[q_id, d_id] / self.obs_freq[q_id, d_id]
                    elif self.mode == "reverse_CTR" :
                        if self.smooth > 0:
                            qp[i] =  - (self.obs_freq[q_id, d_id] + self.smooth) / \
                                            (self.click_freq[q_id, d_id] + self.smooth / self.prior[0] )
                        else:
                            if self.click_freq[q_id, d_id] == 0:
                                qp[i] = - 1e6
                            else:
                                qp[i] = - self.obs_freq[q_id, d_id] / self.click_freq[q_id, d_id]
                    elif self.mode == "product":
                        qp[i] = self.click_freq[q_id, d_id] * self.obs_freq[q_id, d_id]
                    elif self.mode == "obs":
                        qp[i] = self.obs_freq[q_id, d_id]
                    else:
                        qp[i] = self.click_freq[q_id, d_id]
                else:
                    qp[i] = self.prior[0]
        
        return qp

    def compute_reward(self, s_u, batch, rels):
        ''' Only for dCTR'''
        return torch.sum(rels)

class PBM(ClickModel):
    '''
        Position-Based Model, making use of the examination hypothesis.
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, embedd_dim : int, state_dim : int, rec_size : int, 
                    serp_feat_prop : bool, num_serp_embedd : List[int], serp_embedd_dim : List[int], 
                    serp_feat_dim : int, serp_feat_type : str, serp_state_dim : int, hidden_layers_cm : List[int], 
                    user_feat_prop : bool, num_user_embedd : List[int], user_embedd_dim : List[int], 
                    user_feat_dim : int, user_feat_type : str, user_state_dim : int, hidden_layers_prop : List[int], 
                    state_learner : str, **kwargs):
        super().__init__(item_embedd, embedd_dim, state_dim, rec_size, state_learner, hidden_layers_cm = hidden_layers_cm, 
                        num_user_embedd = num_user_embedd, user_embedd_dim = user_embedd_dim, user_feat_dim = user_feat_dim, 
                        user_feat_type = user_feat_type, user_state_dim = user_state_dim, num_serp_embedd = num_serp_embedd, 
                        serp_embedd_dim = serp_embedd_dim, serp_feat_dim = serp_feat_dim, serp_feat_type = serp_feat_type, 
                        serp_state_dim = serp_state_dim, **kwargs)

        if state_learner == "pair_embedding" or state_learner == "PairEmbeddingStateLearner":
            self.discrete = True
        else:
            self.discrete = False
        self.serp_feat_prop = serp_feat_prop
        self.user_feat_prop = user_feat_prop
        self.num_serp_embedd = num_serp_embedd
        self.serp_feat_dim = serp_feat_dim
        self.user_feat_dim = user_feat_dim
        self.serp_feat_type = serp_feat_type
        self.user_feat_type = user_feat_type

        # Free unused memory space
        self.click_pred_head = None

        # Loss comes after Sigmoid in PBM
        self.loss_fn = BCELoss(reduction = 'none')

        self.propensities = Embedding(50, 1)

        if serp_feat_prop:
            self.serp_state_dim = serp_state_dim
            if serp_feat_dim == 0:
                raise ValueError("There are no serp features in the dataset. Please set serp_feat_prop = False.")
            if serp_feat_type == 'int':
                if serp_feat_dim == 1:
                    self.serp_encoding = Embedding(num_serp_embedd[0], serp_state_dim).to(self.my_device)
                else:
                    self.serp_embeddings = [Embedding(num_serp_embedd[i], serp_embedd_dim[i]).to(
                                                self.my_device) for i in range(serp_feat_dim)]
                    self.lin_layer_serp = Linear(torch.sum(torch.LongTensor(serp_embedd_dim)), serp_state_dim)
        
                    self.serp_encoding = lambda x : self.lin_layer_serp(torch.cat([embeddings(x[:, :, i]) 
                                            for i,embeddings in enumerate(self.serp_embeddings)], dim = 2))
            else:
                raise NotImplementedError("We only support categorical serp features for now.")
        else:
            self.serp_state_dim = 0

        if user_feat_prop :
            self.user_state_dim = user_state_dim
            if serp_feat_dim == 0:
                raise ValueError("There are no user features in the dataset. Please set user_feat_prop = False.")
            if user_feat_type == 'int':
                if user_feat_dim == 2:
                    self.user_encoding = Embedding(num_user_embedd[1], user_state_dim).to(self.my_device)
                else:
                    self.user_embeddings = [Embedding(num_user_embedd[i], user_embedd_dim[i]).to(
                                                self.my_device) for i in range(1, user_feat_dim)]
                    self.lin_layer_user = Linear(torch.sum(torch.LongTensor(user_embedd_dim[1:])), user_state_dim)
        
                    self.user_encoding = lambda x : self.lin_layer_user(torch.cat([embeddings(x[:, i]) 
                                            for i,embeddings in enumerate(self.user_embeddings)], dim = 1))
            else:
                raise NotImplementedError("We only support categorical user features for now.")
        else:
            self.user_state_dim = 0
            
        # Examination Network
        layers = []
        input_size = self.serp_state_dim + self.user_state_dim + 1
        out_size = hidden_layers_prop[:]
        if input_size != 1:
            out_size.append(1)
            for i, layer_size in enumerate(out_size):
                layers.append(Linear(input_size, layer_size))
                input_size = layer_size
                if i != len(out_size) - 1:
                    layers.append(ReLU())
        layers.append(Sigmoid())
        self.examination = Sequential(*layers)


        if self.discrete:
            self.item_embedd = None

            init_rCTR = False
            if init_rCTR : 
                click_freq_rank = torch.load(self.data_dir + "click_freq_rank.pt")
                obs_freq_rank = torch.load(self.data_dir + "obs_freq_rank.pt")

                obs_by_rank = [[val for key, val in obs_freq_rank.items() if key[2] == rank] 
                                                            for rank in range(self.rec_size)]
                n_obs_by_rank = torch.cat([torch.sum(torch.tensor(o_r, device = self.my_device)).unsqueeze(0) for o_r in obs_by_rank])

                clicks_by_rank = [[val for key, val in click_freq_rank.items() if key[2] == rank] 
                                                            for rank in range(self.rec_size)]
                n_clicks_by_rank = torch.cat([torch.sum(torch.tensor(c_r, device = self.my_device)).unsqueeze(0) for c_r in clicks_by_rank])
                
                new_prop = torch.zeros(50, 1)
                new_prop[:10, 0] = n_clicks_by_rank / n_obs_by_rank
                self.propensities.weight.data.copy_(torch.logit(new_prop / new_prop[0, 0]))

                print(new_prop[:10, 0])

                pairs_in_data = torch.load(self.data_dir + "pair_embedding_dict.pt")
                alphas = self.state_learner.user_embedd.weight.data.clone()
                for (qid, did), pair_num in pairs_in_data.items():
                    if (qid, did, 0) in obs_freq_rank:
                        alphas[pair_num, 0] = (click_freq_rank[qid, did, 0] + 0.5) / (obs_freq_rank[qid, did, 0] + 0.5 / new_prop[0, 0])
                
                self.state_learner.user_embedd.weight.data.copy_(alphas)
            
            ### Oracle
            # self.pairs_in_data = torch.load(self.data_dir + "pair_embedding_dict.pt")

            # attr_weights = self.state_learner.user_embedd.weight.data.clone().to(self.my_device)
            # rels = torch.load(self.data_dir + "../relevances_sampled.pt", map_location = self.my_device)
            # for qid, rel in rels.items():
            #     docs_in_d = torch.tensor([did for did in range(len(rel)) if (qid, did) in self.pairs_in_data])
            #     pairs = torch.tensor([self.pairs_in_data[qid, did] for did in range(len(rel)) 
            #             if (qid, did) in self.pairs_in_data])
            #     if len(docs_in_d) > 0:
            #         attr_weights[pairs, 0] = torch.logit(0.7 * rel[docs_in_d] / 4)
            # self.state_learner.user_embedd.weight.data.copy_(attr_weights)

            # gamma = torch.tensor([0.9**i for i in range(50)]).unsqueeze(1)
            # self.propensities.weight.data.copy_(torch.logit(gamma))

        else:
            # Relevance Network
            layers = []
            input_size = state_dim + embedd_dim
            out_size = hidden_layers_cm[:]
            out_size.append(1)
            for i, layer_size in enumerate(out_size):
                layers.append(Linear(input_size, layer_size))
                input_size = layer_size
                if i != len(out_size) - 1:
                    layers.append(ReLU())
                else:
                    layers.append(Sigmoid())
            self.attractiveness = Sequential(*layers)

    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[ClickModel.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        parser.add_argument('--user_feat_prop', type=parser.str2bool, default = False)
        parser.add_argument('--serp_feat_prop', type=parser.str2bool, default = False)
        parser.add_argument('--hidden_layers_prop', type=int, nargs='+', default = [])
        parser.add_argument('--user_state_dim', type=int, default = 64)

        arguments = [action.option_strings[0] for action in parser._actions]
        if '--serp_embedd_dim' not in arguments :
            parser.add_argument('--serp_state_dim', type=int, default = 32)
            parser.add_argument('--serp_embedd_dim', type=int, nargs = '+', default = [16, 16, 8])
        if '--user_embedd_dim' not in arguments :
            parser.add_argument('--user_embedd_dim', type=int, nargs = '+', default = [64, 4, 8, 16])
        
        return parser

    #@profile
    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        '''
            Forward pass through the click prediction head.

            Parameters : 
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Flattenned batch of user states.
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - return_loss : boolean
                If True, the click probability and the loss are returned, and if False, the sampled predicted clicks are returned

            
            Output : 
             - click_pred : list(torch.{Long or Float}Tensor(size = (seq_len, rec_size)), len = batch_size)
                Predicted clicks or click probability (see above)
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)
        rec_size = batch.rec_lists[0].size()[1]

        ### 1 - Pre-process input

        if self.discrete:
            attr_pred = Sigmoid()(s_u.flatten(end_dim = 1))
        else:
            s_u, rec_features = self.preprocess_input(s_u, batch)

            ### 2 - Pass through attractiveness model
            inp = torch.cat([s_u, rec_features], dim = 1).float()
            attr_pred = self.attractiveness(inp)


        ### 3 - Get examination probability
        rank = torch.arange(rec_size, device=self.device).repeat(sum_seq_lens)
        propensities = self.propensities(rank)

        if self.serp_feat_prop:
            serp_feat = self.serp_encoding(torch.cat(batch.serp_feat, dim = 0)).flatten(end_dim = 1)
        else:
            serp_feat = torch.empty(0, device = self.device)

        if self.user_feat_prop:
            if len(batch.user_feat[0][0]) == 1 and self.user_feat_dim != 1:
                # That means there is no user features other than query in the ground truth
                batch.user_feat = [torch.cat([queries, torch.zeros(len(queries), self.user_feat_dim - 1, 
                                                                    dtype = torch.long, device = self.device)], 
                                                dim = 1) for queries in batch.user_feat]
            user_feat = self.user_encoding(torch.cat(batch.user_feat, dim = 0)[:, 1:]).unsqueeze(1).expand(-1, 
                                                                                rec_size, -1).flatten(end_dim = 1)
        else:
            user_feat = torch.empty(0, device = self.device)

        exam_prob = self.examination(torch.cat([propensities, serp_feat, user_feat], dim = 1))

        ### 4 - Get click probability
        click_pred = attr_pred * exam_prob
        click_pred = click_pred.reshape(sum_seq_lens, rec_size)

        ### 5 - Reshape and return

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens * rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])
        click_pred_list = [click_pred[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]

        if return_loss:
            # Get targets         
            target_click = torch.cat(batch.clicks, dim = 0).float()
            
            # Compute loss
            loss = self.loss_fn(click_pred, target_click)
            return click_pred_list, loss

        else:
            return [torch.bernoulli(seq for seq in click_pred_list)]

    def extract_relevances(self, s_u, batch):
        '''
        Extracts uncontextualized relevances for NDCG evaluation.

        Parameters :
         - s_u : FloatTensor(size = sum_docs_per_query, state_dim)
            User state, obtained without enriched user features.
         - batch : TrajectoryBatch
            Batch of queries and documents -> No user, serp and interaction features, no clicks
        
        Output :
         - attr_pred_list : list(FloatTensor(n_doc_per_query), len = batch_size)

        '''

        if self.discrete:
            attr_pred = Sigmoid()(s_u).squeeze()
        else:
            s_u, rec_features = self.preprocess_input_rels(s_u, batch)

            ### 2 - Pass through attractiveness model
            inp = torch.cat([s_u, rec_features], dim = 1).float()
            attr_pred = self.attractiveness(inp).squeeze() # (n_docs_per_query)   

        return attr_pred

    def max_reward_policy(self, query, docs, rels):
        '''
            Returns a policy maximizing CTR, according to the click model

            Parameters :
             - query : int
                Current query
             - docs : torch.LongTensor(n_docs_in_q)
                List of documents associated with this query
             - rels : torch.LongTensor(n_docs_in_q)
                Relevances of documents in docs
            
            Output : 
             - policy : torch.LongTensor(self.rec_size)
                Maximum reward policy
        '''
        gammas = self.propensities.weight.data[:self.rec_size, 0]

        ordered_ranks = torch.arange(self.rec_size)[torch.argsort(gammas, descending = True)]
        top_rels, top_doc_idx = torch.sort(rels, descending = True)
        top_docs = docs[top_doc_idx]
        policy = top_docs[ordered_ranks]

        return policy, top_rels[ordered_ranks]

    def compute_reward(self, s_u, batch, rels):
        ''' Only for dCTR'''
        gammas = self.examination(self.propensities.weight.data[:self.rec_size, 0])
        return torch.sum(rels * gammas)

class UBM(PBM):
    '''
        User Browsing Model (Dupret and Piwowarski, 2008).
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, rec_size : int, **kwargs):
        super().__init__(item_embedd, rec_size = rec_size, **kwargs)

        self.propensities = Embedding(rec_size * (rec_size + 1), 1)
    
    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[PBM.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        return parser

    def forward_click_pred_head(self, s_u, batch, return_loss = True):  
        '''
            Forward pass through the click prediction head.

            Parameters : 
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Flattenned batch of user states.
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - return_loss : boolean
                If True, the click probability and the loss are returned, and if False, the sampled predicted clicks are returned

            
            Output : 
             - click_pred : list(torch.{Long or Float}Tensor(size = (seq_len, rec_size)), len = batch_size)
                Predicted clicks or click probability (see above)
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)
        rec_size = batch.rec_lists[0].size()[1]

        ### 1 - Pre-process input
        if self.discrete:
            attr_pred = Sigmoid()(s_u).squeeze(dim = 2)
        else:
            s_u, rec_features = self.preprocess_input(s_u, batch)

            ### 2 - Pass through attractiveness model
            inp = torch.cat([s_u, rec_features], dim = 1).float()
            attr_pred = self.attractiveness(inp).reshape(sum_seq_lens, rec_size)


        if self.serp_feat_prop:
            serp_feat = torch.cat(batch.serp_feat, dim = 0).reshape(sum_seq_lens * self.rec_size, self.serp_feat_dim)
            if self.serp_feat_type == 'int':
                apply = lambda l, t : [el(t[:, i]) for i,el in enumerate(l)]
                serp_feat = torch.cat(apply(self.serp_embeddings, serp_feat), dim = 1)
        else:
            serp_feat = torch.empty(0, rec_size, device = self.device)

        if self.user_feat_prop:
            user_feat = torch.cat(batch.user_feat, dim = 0)[:, 1:].unsqueeze(1).expand(-1, self.rec_size, -1).reshape(
                                                sum_seq_lens*self.rec_size, self.user_feat_dim_prop - 1)
            if self.user_feat_type_prop == 'int':
                apply = lambda l, t : [el(t[:, i]) for i,el in enumerate(l)]
                user_feat = torch.cat(apply(self.user_embeddings_prop, user_feat), dim = 1)
        else:
            user_feat = torch.empty(0, rec_size, device = self.device)

        context = torch.cat([serp_feat, user_feat], dim = 0)
        
        ### 3 - Get examination and click probability
        
        latest_clicked_items = torch.zeros(sum_seq_lens, dtype = torch.long, device = self.device)
        click_pred = torch.zeros(sum_seq_lens, self.rec_size, dtype = torch.long, device = self.device) ## (sum_seq_lens, rec_size)
        if return_loss :
            target_clicks = torch.cat(batch.clicks, dim = 0).float() ## (sum_seq_lens, rec_size)
            click_pred = click_pred.float() 

        rank = torch.zeros(sum_seq_lens, dtype = torch.long, device = self.device)
        for r in range(self.rec_size):   ## 0 to rec_size-1, not 1 to rec_size
            rank += 1
            exam_prob = self.examination(torch.cat([self.propensities(latest_clicked_items + r * (self.rec_size + 1)), 
                                                    context[:, r]], dim = 1))   # (sum_seq_len, 1)
            # print(exam_prob.size())
            # print(attr_pred.size())
            clicks_prob = exam_prob.squeeze() * attr_pred[:, r] # len = sum_seq_len
            if not return_loss:
                clicks = torch.bernoulli(clicks_prob) # len = sum_seq_len
                latest_clicked_items = torch.where(clicks == 1, rank, latest_clicked_items) # len = sum_seq_len
                click_pred[:, r] = clicks
            else: 
                latest_clicked_items = torch.where(target_clicks[:,r] == 1, rank, latest_clicked_items)
                click_pred[:, r] = clicks_prob
        
        if return_loss:
            loss = self.loss_fn(click_pred, target_clicks) # Conditional log-likelihood : log( P(C_r = c_{obs,r} | c{obs,<r}) )

        ### 4 - Reshape and return

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens, rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])
        click_pred = [click_pred[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]
            
        if return_loss:
            return click_pred, loss
        else:
            return click_pred

    def compute_reward(self, s_u, batch, rels):
            # temp[j] = prod_{k = j+1}^{rank - 1} ( 1 - alpha_dk_q gamma_k_j )
            embedd_idx = lambda k, j : j + (k-1) * (self.rec_size + 1)
            temp = torch.ones(1)
            click_prob = rels[0] * Sigmoid()(self.propensities(torch.tensor(0))) 
            for r in range(2, self.rec_size + 1):
                # 1 - Update temp
                gamma_rd_minus_1_j = Sigmoid()(self.propensities(embedd_idx(r - 1, torch.arange(r - 1))).squeeze()) # (r - 1)
                alpha_d_rd_minus_one = rels[r-2]

                temp *= (1 - alpha_d_rd_minus_one * gamma_rd_minus_1_j) # (r - 1)

                # 2 - For each remaining doc, compute click prob
                gamma_rd_j = Sigmoid()(self.propensities(embedd_idx(r, torch.arange(r))).squeeze()) # (r)
                alpha_d = rels[r - 1]
                cp_rank = torch.cat([torch.tensor([1.0]), click_prob[:-1]]) # (r - 1)
                cp = torch.sum(temp * cp_rank * gamma_rd_j[:-1]) * alpha_d + \
                                        alpha_d * gamma_rd_j[-1] * click_prob[-1]
                click_prob = torch.cat([click_prob, cp.unsqueeze(0)]) # r
                temp = torch.cat([temp, torch.ones(1)]) # r
            
            return torch.sum(click_prob)

    def max_reward_policy(self, query, docs, rels):
        '''
            Returns a policy maximizing CTR, according to the click model

            Parameters :
                - query : int
                Current query
                - docs : torch.LongTensor(n_docs_in_q)
                List of documents associated with this query
                - rels : torch.LongTensor(n_docs_in_q)
                Relevances of documents in docs
            
            Output : 
                - policy : torch.LongTensor(self.rec_size)
                Maximum reward policy
        '''
        if self.discrete:
            raise NotImplementedError("The MaxReward policy has only been implemented for UBM Pair.")

        ### Keep only top 10 most relevant documents
        top_rels, top_doc_idx = torch.sort(rels, descending = True)
        top_rels = top_rels[:self.rec_size]
        top_doc_idx = top_doc_idx[:self.rec_size]        
        
        all_rankings = torch.load("/home/rdeffaye/workspace/playground/perms.pt", map_location = self.device) 
                # All permutations of rankings (saved once to avoid recomputing it each time)
        # Sample
        sampling_rate = 900
        n = len(all_rankings)
        all_rankings = all_rankings[torch.randperm(n)[: n // sampling_rate]]


        traj_reward = torch.zeros(len(all_rankings))
        for k, rl in enumerate(all_rankings):
            rels_perm = top_rels[rl] 
            traj_reward[k] = self.compute_reward(None, None, rels_perm)

        best_traj_idx = torch.argmax(traj_reward) 
        best_traj = docs[top_doc_idx[all_rankings[best_traj_idx]]]
        
        return best_traj, top_rels[all_rankings[best_traj_idx]] 

class DBN(PBM):
    '''
        Dynamic bayesian Network, [Chapelle and Zhang, 2009].
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, state_dim : int, embedd_dim : int, oracle : bool, rel_path : str,
                    simplified : bool, hidden_layers_cm : List[int], init_sdbn : bool, loss_per_rank : bool, **kwargs):
        super().__init__(item_embedd, state_dim = state_dim, embedd_dim = embedd_dim, 
                            hidden_layers_cm = hidden_layers_cm, **kwargs)

        self.simplified = simplified
        self.init_sdbn = init_sdbn
        self.loss_per_rank = loss_per_rank

        # Click model
        if self.discrete:
            self.pairs_in_data = torch.load(self.data_dir + "pair_embedding_dict.pt")
            self.relevance = Embedding(len(self.pairs_in_data) + 1, 1).to(self.my_device)
                
            if init_sdbn:
                obs_freq_before_last = torch.load(self.data_dir + "obs_freq_before_last.pt")
                click_freq_before_last = torch.load(self.data_dir + "click_freq_before_last.pt")
                click_freq_last = torch.load(self.data_dir + "click_freq_last.pt")

                rel_weights = self.relevance.weight.data.clone()
                attr_weights = self.state_learner.user_embedd.weight.data.clone()
                for pair in obs_freq_before_last.keys():
                    if obs_freq_before_last[pair] > 5:
                        alpha = torch.logit(torch.clamp(torch.tensor(click_freq_before_last[pair] / obs_freq_before_last[pair]), 1e-3, 1 - 1e-3))
                        attr_weights[self.pairs_in_data[pair], 0] = alpha
                        if click_freq_before_last[pair] != 0:
                            sigma = torch.logit(torch.tensor(click_freq_last[pair] / click_freq_before_last[pair]))
                        else:
                            sigma = torch.logit(torch.tensor(0))
                        rel_weights[self.pairs_in_data[pair], 0] = sigma
                    else:
                        attr_weights[self.pairs_in_data[pair], 0] = torch.logit(torch.tensor(1e-2))

                self.relevance.weight.data.copy_(rel_weights)
                self.state_learner.user_embedd.weight.data.copy_(attr_weights)

            ### This is for DBN Oracle
            if oracle:
                rel_weights = self.relevance.weight.data.clone()
                attr_weights = self.state_learner.user_embedd.weight.data.clone().to(self.my_device)
                rels = torch.load(self.data_dir + rel_path, map_location = self.my_device)
                for qid, rel in rels.items():
                    docs_in_d = torch.tensor([did for did in range(len(rel)) if (qid, did) in self.pairs_in_data])
                    pairs = torch.tensor([self.pairs_in_data[qid, did] for did in range(len(rel)) 
                            if (qid, did) in self.pairs_in_data])
                    if len(docs_in_d) > 0:
                        attr_weights[pairs, 0] = torch.logit(torch.maximum(0.02 * torch.ones_like(rel[docs_in_d]), 0.95 * rel[docs_in_d]))
                        rel_weights[pairs, 0] = torch.logit(0.5 * rel[docs_in_d])
                self.relevance.weight.data.copy_(rel_weights)
                self.state_learner.user_embedd.weight.data.copy_(attr_weights)
            

        else:
            layers = []
            input_size = state_dim + embedd_dim
            out_size = hidden_layers_cm[:]
            out_size.append(1)
            for i, layer_size in enumerate(out_size):
                layers.append(Linear(input_size, layer_size))
                input_size = layer_size
                if i != len(out_size) - 1:
                    layers.append(ReLU())
            else:
                layers.append(Sigmoid())
            self.relevance = Sequential(*layers)

        self.gamma = Embedding(1,1).requires_grad_(not simplified)
        if simplified or init_sdbn:
            self.gamma.weight.data.copy_(torch.logit(torch.tensor([[1.0]])))
        else:
            self.gamma.weight.data.copy_(torch.logit(torch.tensor([[0.9]])))    # We initialize at 0.9 as it is a realistic value

    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[PBM.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        parser.add_argument('--simplified', type=parser.str2bool, default = False)
        parser.add_argument('--init_sdbn', type=parser.str2bool, default = False)
        parser.add_argument('--loss_per_rank', type=parser.str2bool, default = False)
        parser.add_argument('--oracle', type=parser.str2bool, default = False)
        parser.add_argument('--rel_path', type=str, default = "../relevances_sampled.pt")
        return parser

    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        '''
            Forward pass through the click prediction head.

            Parameters : 
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Flattenned batch of user states.
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - return_loss : boolean
                If True, the click probability and the loss are returned, and if False, the sampled predicted clicks are returned

            
            Output : 
             - click_pred : list(torch.{Long or Float}Tensor(size = (seq_len, rec_size)), len = batch_size)
                Predicted clicks or click probability (see above)
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)
        rec_size = batch.rec_lists[0].size()[1]

        ### 1 - Pre-process input
        if self.discrete:
            attr_pred = Sigmoid()(s_u).squeeze(dim = 2)

            queries = torch.cat(batch.user_feat, dim = 0)[:, 0]
            items = torch.cat(batch.rec_lists, dim = 0)
            if self.my_device == "cpu":
                pair_idx = torch.LongTensor([[self.pairs_in_data[q.item() ,i.item()] for i in l]
                                                    for (q,l) in zip(queries, items) ])
            else:
                pair_idx = torch.cuda.LongTensor([[self.pairs_in_data[q.item() ,i.item()] for i in l]
                                                    for (q,l) in zip(queries, items) ])
            rel_pred = Sigmoid()(self.relevance(pair_idx)).squeeze(dim = 2)

            '''
            print("\n Batch : ", batch)

            print("\n attr_pred : ", attr_pred)
            print(attr_pred.size())

            print("\n rel_pred : ", rel_pred)
            print(rel_pred.size())

            [1, 2] * [1, 2, 3]
            '''


            # attr_pred = torch.stack([self.attr[q.item()][rl.squeeze()] for qs, rls in zip(batch.user_feat, batch.rec_lists) 
            #                                                                     for q, rl in zip(qs, rls)])
            # rel_pred = torch.stack([self.rel[q.item()][rl.squeeze()] for qs, rls in zip(batch.user_feat, batch.rec_lists) 
            #                                                                         for q, rl in zip(qs, rls)])
        else:
            s_u, rec_features = self.preprocess_input(s_u, batch)

            ### 2 - Pass through attractiveness model
            inp = torch.cat([s_u, rec_features], dim = 1).float()
            attr_pred = self.attractiveness(inp).reshape(sum_seq_lens, self.rec_size)

            rel_pred = self.relevance(inp).reshape(sum_seq_lens, self.rec_size)

        ### Get context

        if self.serp_feat_prop:
            serp_feat = self.serp_encoding(torch.cat(batch.serp_feat, dim = 0)).flatten(end_dim = 1)
        else:
            serp_feat = torch.empty(0, device = self.device)

        if self.user_feat_prop:
            if len(batch.user_feat[0][0]) == 1 and self.user_feat_dim != 1:
                # That means there is no user features other than query in the ground truth
                batch.user_feat = [torch.cat([queries, torch.zeros(len(queries), self.user_feat_dim - 1, 
                                                                    dtype = torch.long, device = self.device)], 
                                                dim = 1) for queries in batch.user_feat]
            user_feat = self.user_encoding(torch.cat(batch.user_feat, dim = 0)[:, 1:]).unsqueeze(1).expand(-1, 
                                                                                rec_size, -1).flatten(end_dim = 1)
        else:
            user_feat = torch.empty(0, device = self.device)

        context = torch.cat([serp_feat, user_feat], dim = 1)


        ### 3 - Get examination and click probability
        exam_prob = torch.ones(sum_seq_lens, dtype = torch.float, device = self.device)
        click_pred = torch.zeros(sum_seq_lens, rec_size, dtype = torch.long, device = self.device) ## (sum_seq_lens, rec_size)
        if return_loss :
            target_clicks = torch.cat(batch.clicks, dim = 0).float() ## (sum_seq_lens, rec_size)
            click_pred = click_pred.float() 
            loss = 0
            if self.loss_per_rank:
                loss_factor = rec_size - torch.arange(rec_size, device = self.device)
            else:
                loss_factor = torch.ones(rec_size, device = self.device)
        
        for r in range(rec_size):   ## 0 to rec_size-1, not 1 to rec_size
            click_prob = attr_pred[:, r] * exam_prob
            if return_loss:
                click_pred[:, r] = click_prob
                satisfactions = rel_pred[:, r]
                exam_nonclick = Sigmoid()(self.gamma(torch.zeros(1, dtype = torch.long, device =self.device)).squeeze()) * exam_prob * \
                                     (1 - attr_pred[:, r]) / (1 - attr_pred[:, r] * exam_prob)  # See 3.9 of Clicks Models for Web Search
                exam_click = (1 - satisfactions) * Sigmoid()(self.gamma(torch.zeros(1, dtype = torch.long, device =self.device)).squeeze())
                exam_prob = target_clicks[:, r] * exam_click + (1 - target_clicks[:,r]) * exam_nonclick
            else:
                clicks = torch.bernoulli(click_prob)
                click_pred[:, r] = clicks
                satisfactions = clicks * torch.bernoulli(rel_pred[:, r])
                exam_prob = (1 - satisfactions) * Sigmoid()(self.gamma(torch.zeros(1, dtype = torch.long, device =self.device)).squeeze())
        
        if return_loss:
            loss = loss_factor * rec_size / torch.sum(loss_factor) * self.loss_fn(click_pred, target_clicks) # Conditional log-likelihood : log( P(C_r = c_{obs,r} | c{obs,r-1}) )
        ### 4 - Reshape and return

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens, rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])
        click_pred = [click_pred[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]
            
        if return_loss:
            return click_pred, loss
        else:
            return click_pred

    def extract_relevances(self, s_u, batch):

        attr_pred = super().extract_relevances(s_u, batch)

        if self.discrete:
            queries = torch.cat(batch.user_feat, dim = 0)[:, 0]
            items = torch.cat(batch.rec_lists, dim = 0)
            if self.my_device == "cpu":
                pair_idx = torch.LongTensor([[self.pairs_in_data[q.item() ,i.item()] for i in l]
                                                    for (q,l) in zip(queries, items) ])
            else:
                pair_idx = torch.cuda.LongTensor([[self.pairs_in_data[q.item() ,i.item()] for i in l]
                                                    for (q,l) in zip(queries, items) ])
            satisfaction_pred = Sigmoid()(self.relevance(pair_idx)).squeeze()
        else:
            s_u, rec_features = self.preprocess_input_rels(s_u, batch)

            ### 2 - Pass through attractiveness model
            inp = torch.cat([s_u, rec_features], dim = 1).float()
            satisfaction_pred = self.relevance(inp).squeeze() # (n_docs_per_query) 
        
        rel_pred = attr_pred * satisfaction_pred

        return rel_pred

    def compute_reward(self, s_u, batch, rels):
        '''only for immediate state learner'''
        s_u, rec_features = self.preprocess_input(s_u.unsqueeze(0), batch)

        ### 2 - Pass through attractiveness model
        inp = torch.cat([s_u, rec_features], dim = 1).float().squeeze()
        attr_pred = self.attractiveness(inp).squeeze()

        rel_pred = self.relevance(inp).squeeze()

        gamma = Sigmoid()(self.gamma(torch.zeros(1, dtype = torch.long, device =self.device)).squeeze())

        examination = torch.ones(self.rec_size)
        for r in range(self.rec_size - 1):
            examination[r + 1] = examination[r] * gamma * (1 - rel_pred[r] * attr_pred[r])

        return torch.sum(attr_pred * examination)

    def max_reward_policy(self, query, docs, rels):
        '''
            Returns a policy maximizing CTR, according to the click model

            Parameters :
             - query : int
                Current query
             - docs : torch.LongTensor(n_docs_in_q)
                List of documents associated with this query
             - rels : torch.LongTensor(n_docs_in_q)
                Relevances of documents in docs
            
            Output : 
             - policy : torch.LongTensor(self.rec_size)
                Maximum reward policy
        '''
        rels_perm, docs_perms = torch.sort(rels, descending = True)
        return docs[docs_perms[:self.rec_size]], rels_perm[:self.rec_size]

class ARM(PBM):
    '''
        AutoRegressive click Model
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, serp_feat_dim : List[int], 
                    serp_feat_type : str, num_serp_embedd : List[int], serp_embedd_dim : List[int], 
                    kernel_size : int, relative : bool, absolute : bool, stacked : bool, 
                    hidden_size : int, all_serp_contexts : bool, non_causal : bool, **kwargs):
        super().__init__(item_embedd, serp_feat_dim = serp_feat_dim, serp_embedd_dim = serp_embedd_dim, 
                            serp_feat_type = serp_feat_type, num_serp_embedd = num_serp_embedd,
                            kernel_size = kernel_size, **kwargs)

        self.relative = relative
        self.absolute = absolute
        self.stacked = stacked
        self.all_serp_contexts = all_serp_contexts
        self.non_causal = non_causal
        self.propensities = None

        if stacked:
            self.hidden_size = hidden_size
            self.output_layer = Sequential(Linear(hidden_size, 1),
                                            Sigmoid())
        else:
            self.hidden_size = 1

        if relative:
            self.kernel_size = kernel_size
            self.propensities_rel = torch.nn.Conv1d(1, self.hidden_size, kernel_size, padding = kernel_size)
        if absolute:
            self.propensities_abs = Embedding(self.rec_size, self.hidden_size)
        if not relative and not absolute:
            raise ValueError("ARM should be relative, absolute, or both.")
        
        self.reduce = Linear(self.hidden_size * (relative + absolute), self.hidden_size)

        if serp_feat_type == 'float':
            if serp_feat_dim > 0:
                self.serp_encoding = Linear(serp_feat_dim, self.hidden_size)    ### Pour commencer
            else : 
                self.serp_encoding = lambda x : torch.zeros(len(x), self.hidden_size)
        elif serp_feat_type == 'int':
            self.lin_layer = Linear(torch.sum(torch.LongTensor(serp_embedd_dim)), self.hidden_size)
            self.serp_embeddings = [Embedding(num_serp_embedd[i], serp_embedd_dim[i]).to(self.my_device) for i in range(serp_feat_dim)]
            self.serp_encoding = lambda x : self.lin_layer(torch.cat([embeddings(x[:, :, i]) for i,embeddings in enumerate(self.serp_embeddings)], dim = 2))
        else:
            raise ValueError("Wrong type for SERP features.")
    
    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[PBM.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        parser.add_argument('--kernel_size', type=int, default = 5)
        parser.add_argument('--relative', type=parser.str2bool, default = False)
        parser.add_argument('--absolute', type=parser.str2bool, default = False)
        parser.add_argument('--stacked', type=parser.str2bool, default = False)
        parser.add_argument('--hidden_size', type=int, default = 8)
        parser.add_argument('--all_serp_contexts', type=parser.str2bool, default = False)
        parser.add_argument('--non_causal', type = parser.str2bool, default = False)
        return parser        

    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        '''
            Forward pass through the click prediction head.

            Parameters : 
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Flattenned batch of user states.
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - return_loss : boolean
                If True, the click probability and the loss are returned, and if False, the sampled predicted clicks are returned

            
            Output : 
             - click_pred : list(torch.{Long or Float}Tensor(size = (seq_len, rec_size)), len = batch_size)
                Predicted clicks or click probability (see above)
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''

        rec_size = batch.rec_lists[0].size()[1]
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)

        if self.discrete:
            attr_pred = Sigmoid()(s_u.flatten(end_dim = 1))
        else:
            s_u, rec_features = self.preprocess_input(s_u, batch)

            ### 2 - Pass through attractiveness model
            inp = torch.cat([s_u, rec_features], dim = 1).float()
            attr_pred = self.attractiveness(inp).reshape(sum_seq_lens, rec_size)

        ### 3 - Get examination probability
        if return_loss:
            target_clicks = torch.cat(batch.clicks, dim = 0).float()    # (sum_seq_len, rec_size) 
        
        exam_prob_rel = torch.empty(sum_seq_lens, rec_size, 0, device = self.device)
        if self.relative:
            exam_prob_rel = self.propensities_rel(target_clicks.unsqueeze(1))[:, :, :-self.kernel_size - 1] # (sum_seq_len, hidden_size, rec_size)
            exam_prob_rel = exam_prob_rel.transpose(1, 2)                           #(sum_seq_len, rec_size, hidden_size)
            exam_prob_abs = torch.empty(sum_seq_lens, rec_size, 0, device = self.device)
        if self.absolute:
            if self.non_causal:
                previous_clicks = target_clicks.unsqueeze(2) # --> ARM NC
            else:
                previous_clicks = torch.cat([torch.ones(sum_seq_lens, 1, device = self.device), 
                                                target_clicks[:, :-1]], dim = 1).unsqueeze(2)    # (sum_seq_len, rec_size, 1) 
            params = self.propensities_abs(torch.arange(rec_size, device = self.device))    # (rec_size, hidden_size)
            exam_prob_abs = torch.cumsum(params * previous_clicks, dim = 1)  #(sum_seq_len, rec_size, hidden_size)
        
        exam_prob = torch.cat([exam_prob_rel, exam_prob_abs], dim = 2)  #(sum_seq_len, rec_size, hidden_size * (relative + absolute))
        exam_prob = self.reduce(exam_prob.flatten(end_dim = 1))         #(sum_seq_len * rec_size, hidden_size)
        exam_prob = exam_prob.reshape(sum_seq_lens, rec_size, self.hidden_size)  #(sum_seq_len, rec_size, hidden_size) 

        if self.serp_feat_prop:
            serp_feat = torch.cat(batch.serp_feat, dim = 0)     # (sum_seq_len, rec_size)
            context = self.serp_encoding(serp_feat)             # (sum_seq_len, rec_size, hidden_size)

            if self.all_serp_contexts :
                context = torch.cumsum(context, dim = 1) #(sum_seq_len, rec_size, hidden_size)
            
            exam_prob += context
        
        exam_prob = Sigmoid()(exam_prob) #(sum_seq_len, rec_size, hidden_size)
        
        if self.stacked:
            exam_prob = self.output_layer(exam_prob.flatten(end_dim = 1)) #(sum_seq_len * rec_size, 1)
            exam_prob = exam_prob.reshape(sum_seq_lens, rec_size) #(sum_seq_len, rec_size)
        exam_prob = exam_prob.squeeze() #(sum_seq_len, rec_size)


        ### 4 - Get click probability
        click_pred = attr_pred * exam_prob  #(sum_seq_len, rec_size)


        ### 5 - Reshape and return

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens,  rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])
        click_pred_list = [click_pred[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]

        if return_loss:
            # Get targets         
            target_click = torch.cat(batch.clicks, dim = 0).float()
            
            # Compute loss
            loss = self.loss_fn(click_pred, target_click)

            return click_pred_list, loss

        else:
            return [torch.bernoulli(seq for seq in click_pred_list)]

    def compute_reward(self, s_u, batch, rels, ep = None, apc = None):
        if ep is None:  # For MaxReward we pre-compute everything not ranking-dependent, so no need to do that
            def get_all_binary_sequences(size, seq = torch.LongTensor([])):
                '''
                    This returns all possible sequences of clicks
                '''
                if len(seq) == size:
                    return seq.unsqueeze(0)

                return torch.cat([get_all_binary_sequences(size, seq = torch.cat([seq, torch.zeros(1, dtype = torch.long)])),
                                    get_all_binary_sequences(size, seq = torch.cat([seq, torch.ones(1, dtype = torch.long)]))], dim = 0)

            #### Pass all possible sequences to model
            apc = torch.cat([torch.zeros(2 ** (self.rec_size - 1), 1, dtype = torch.long), get_all_binary_sequences(self.rec_size - 1)], dim = 1)
            # apc is (2 ** (self.rec_size - 1), rec_size)
            params = self.propensities_abs(torch.arange(self.rec_size, device = self.device)).squeeze()    # rec_size
            exam_prob_abs = torch.cumsum(apc * params, dim = 1) # (2 ** (self.rec_size - 1), rec_size)
            ep = self.reduce(exam_prob_abs.flatten().unsqueeze(1)).reshape(2 ** (self.rec_size - 1), self.rec_size)   # (2 ** (self.rec_size - 1), rec_size)
            ep = Sigmoid()(ep)

        def build_prev_seq_prob(all_prev_clicks, exam_probs, rels):
            '''
                This gives P(C_<r = c_<r) for all 1 <= r <= self.rec_size and (c_1, ..., c_(self.rec_size -1)) for the given ranking
            '''
            prev_seq_prob = torch.ones(len(all_prev_clicks)).unsqueeze(1)
            for r in range(1, self.rec_size):
                new_prob = (1 - all_prev_clicks[:, r] + (2 * all_prev_clicks[:, r] - 1) * rels[r - 1] * exam_probs[:, r-1]) * prev_seq_prob[:, -1] 
                prev_seq_prob = torch.cat([prev_seq_prob, new_prob.unsqueeze(1)], dim = 1)
            return prev_seq_prob    # Should be of size (2 ** (self.rec_size - 1), self.rec_size)
        
        all_seqs = lambda length : torch.arange(2 ** (length - 1)) * 2 ** (self.rec_size - length)
        def get_marginal_click_prob(prev_seq_prob, exam_probs, rels):
            '''
                This gives the list [P(C_1 = 1), ... , P(C_self.rec_size = 1)] for the given ranking
            '''
            seq_idx = [all_seqs(k) for k in range(1, self.rec_size + 1)]
            marginal_exam_prob = [torch.sum(exam_probs[seq_idx[k], k] * prev_seq_prob[seq_idx[k], k]) for k in range(len(seq_idx))]

            return rels * torch.tensor(marginal_exam_prob)
        
        # P(C_r = 1) = P(A_r = 1) * sum_prevclickseqs(P(E_r = 1 |prevclickseq) * P(prevclick_seq))
        prev_seq_prob = build_prev_seq_prob(apc, ep, rels)
        click_probs = get_marginal_click_prob(prev_seq_prob, ep, rels)
        return torch.sum(click_probs)

    def max_reward_policy(self, query, docs, rels):
        '''
            Returns a policy maximizing CTR, according to the click model

            Parameters :
             - query : int
                Current query
             - docs : torch.LongTensor(n_docs_in_q)
                List of documents associated with this query
             - rels : torch.LongTensor(n_docs_in_q)
                Relevances of documents in docs
            
            Output : 
             - policy : torch.LongTensor(self.rec_size)
                Maximum reward policy
        '''
        ### Keep only top 10 most relevant documents
        top_rels, top_doc_idx = torch.sort(rels, descending = True)
        top_rels = top_rels[:self.rec_size]
        top_doc_idx = top_doc_idx[:self.rec_size]        
        
        all_rankings = torch.load("/home/rdeffaye/workspace/playground/perms.pt", map_location = self.device) 
                # All permutations of rankings (saved once to avoid recomputing it each time)
        # Sample
        sampling_rate = 900
        n = len(all_rankings)
        all_rankings = all_rankings[torch.randperm(n)[: n // sampling_rate]]

        def get_all_binary_sequences(size, seq = torch.LongTensor([])):
            '''
                This gives the list [P(C_1 = 1), ... , P(C_self.rec_size = 1)] for the given ranking
            '''
            if len(seq) == size:
                return seq.unsqueeze(0)

            return torch.cat([get_all_binary_sequences(size, seq = torch.cat([seq, torch.zeros(1, dtype = torch.long)])),
                                get_all_binary_sequences(size, seq = torch.cat([seq, torch.ones(1, dtype = torch.long)]))], dim = 0)

        #### Pass all possible sequences to model
        all_prev_clicks = torch.cat([torch.zeros(2 ** (self.rec_size - 1), 1, dtype = torch.long), get_all_binary_sequences(self.rec_size - 1)], dim = 1)
        # all_prev_clicks is (2 ** (self.rec_size - 1), rec_size)
        params = self.propensities_abs(torch.arange(self.rec_size, device = self.device)).squeeze()    # rec_size
        exam_prob_abs = torch.cumsum(all_prev_clicks * params, dim = 1) # (2 ** (self.rec_size - 1), rec_size)
        exam_prob = self.reduce(exam_prob_abs.flatten().unsqueeze(1)).reshape(2 ** (self.rec_size - 1), self.rec_size)   # (2 ** (self.rec_size - 1), rec_size)
        exam_prob = Sigmoid()(exam_prob)

        traj_reward = torch.zeros(len(all_rankings))
        for k, rl in enumerate(all_rankings):
            rels_perm = top_rels[all_rankings[k]]
            traj_reward[k] = self.compute_reward(None, None, rels_perm, ep = exam_prob, apc = all_prev_clicks)

        best_traj = docs[top_doc_idx[all_rankings[torch.argmax(traj_reward)]]]

        return best_traj, top_rels[all_rankings[torch.argmax(traj_reward)]]

class CACM(PBM):
    '''
        Derived from Context-Aware Click Model [Chen et al., 2020]
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, inner_state_dim : int, serp_embedd_dim : List[int], 
                    hidden_layers_cm : List[int], pos_embedd_dim : int, click_embedd_dim : int, doc_embedd_size : int, 
                    time_embedd_dim : int, serp_feat_prop : bool, **kwargs):
        super().__init__(item_embedd, hidden_layers_cm = hidden_layers_cm, serp_feat_prop = True, 
                            pos_embedd_dim = pos_embedd_dim, click_embedd_dim = click_embedd_dim, 
                            serp_embedd_dim = serp_embedd_dim, inner_state_dim = inner_state_dim, **kwargs)

        #### Document Encoder Embeddings
        self.pos_embedd_doc = Embedding(self.rec_size, pos_embedd_dim)

        if self.serp_feat_type == 'int':
            self.serp_embedd_doc = [Embedding(self.num_serp_embedd[i], serp_embedd_dim[i]).to(self.my_device) 
                                            for i in range(self.serp_feat_dim)]
            self.serp_embedd_dim_doc = torch.sum(torch.LongTensor(serp_embedd_dim[:self.serp_feat_dim])).item()

        self.time_embedd_doc = Embedding(100, time_embedd_dim) # A changer eventuellement

        #### Document Encoder
        self.doc_encoder = Sequential(Linear(self.embedd_dim + pos_embedd_dim + self.serp_embedd_dim_doc + time_embedd_dim, 
                                                    doc_embedd_size), 
                                        Tanh())

        
        #### Relevance Predictor
        hidden_layer_size = hidden_layers_cm[0]
        self.attractiveness = Sequential(Linear(self.state_dim + doc_embedd_size, hidden_layer_size),
                                            Tanh(),
                                            Linear(hidden_layer_size, 1),
                                            Sigmoid())
        

        #### Examination Embeddings
        self.pos_embedd_exam = Embedding(self.rec_size, pos_embedd_dim)

        if self.serp_feat_type == 'int':
            self.serp_embedd_exam = [Embedding(self.num_serp_embedd[i], serp_embedd_dim[i]).to(self.my_device) 
                                            for i in range(self.serp_feat_dim)]
            self.serp_embedd_dim_exam = torch.sum(torch.LongTensor(serp_embedd_dim[:self.serp_feat_dim])).item()

        self.click_embedd_exam = Embedding(2, click_embedd_dim)


        #### Examination predictor
        self.propensities = GRU(pos_embedd_dim + self.serp_embedd_dim_exam + click_embedd_dim, 
                                        hidden_size = inner_state_dim, num_layers = 1, batch_first = True)
        self.fully_connected_exam = Sequential(Linear(inner_state_dim, 1),
                                                Sigmoid())
        self.register_buffer("h0_exam", torch.zeros(1, 1, inner_state_dim)) ## How to initialize it ?        

    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[PBM.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        
        parser.add_argument('--time_embedd_dim', type=int, default = 4)
        parser.add_argument('--doc_embedd_size', type=int, default = 10)

        arguments = [action.option_strings[0] for action in parser._actions]
        if '--pos_embedd_dim' not in arguments :
            parser.add_argument('--pos_embedd_dim', type=int, default = 4)
        if '--click_embedd_dim' not in arguments :
            parser.add_argument('--click_embedd_dim', type=int, default = 4)
        if '--inner_state_dim' not in arguments :
            parser.add_argument('--inner_state_dim', type=int, default = 64)
        return parser 

    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        '''
            Forward pass through the click prediction head (batch input).

            Parameters : 
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Batch of user states (or queries).
             - rec_items : list(torch.Tensor(long, size = (seq_len, rec_size)), len = batch_size))
                Batch of sequences of recommended items.
             - seq_lens : torch.Tensor(long, size = (batch_size,))
                Length of all sequences in the batch.

            
            Output : 
             - click_pred : list(torch.Tensor(float, size = (seq_len, rec_size)), len = batch_size)
                Predicted clicks (before sigmoid because of BCEWithLogitsLoss)
            
        '''
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)
        rec_size = batch.rec_lists[0].size()[1]

        ### 1 - Pass through Document Encoder

        s_u, url_embedd = self.preprocess_input(s_u, batch)
        positions = torch.arange(rec_size, device = self.device).unsqueeze(0).expand(sum_seq_lens, -1) # (sum_seq_len, rec_size)
        serp_feat = torch.cat([sf for sf in batch.serp_feat], dim = 0)  # (sum_seq_len, rec_size, serp_feat_dim)
        timesteps = torch.cat([torch.arange(sl, device = self.device).unsqueeze(1).expand(-1, rec_size) 
                                for sl in batch.seq_lens]) # (sum_seq_len, rec_size)

        pos_embedd = self.pos_embedd_doc(positions.flatten())   # (sum_seq_len * rec_size, pos_embedd_dim)
        if self.serp_feat_type == 'int':
            apply = lambda l, t : [el(t[:, i]) for i,el in enumerate(l)]
            serp_embedd = torch.cat(apply(self.serp_embedd_doc, serp_feat.reshape(sum_seq_lens * rec_size, self.serp_feat_dim)), 
                                        dim = 1) # (sum_seq_len * rec_size, serp_embed_dim)
        time_embedd = self.time_embedd_doc(timesteps.flatten())           # (sum_seq_len * rec_size, time_embedd_dim)

        doc_embedd = self.doc_encoder(torch.cat([url_embedd, pos_embedd, serp_embedd, time_embedd], dim = 1))

        ### 2 - Pass through relevance estimator        
        relevance_pred = self.attractiveness(torch.cat([s_u, doc_embedd], dim = 1)).squeeze()       # sum_seq_len * rec_size


        ### 3 - Pass through examination predictor
        ## Careful here : we obtain tensors of size (sum_seq_len, rec_size, *), but sum_seq_len corresponds to batch size and rec_size to seq_len
        ## of the examination GRU
        pos_embedd = self.pos_embedd_exam(positions)   # (sum_seq_len, rec_size, pos_embedd_dim)
        if self.serp_feat_type == 'int':
            apply = lambda l, t : [el(t[:, :, i]) for i,el in enumerate(l)]
            serp_embedd = torch.cat(apply(self.serp_embedd_exam, serp_feat), 
                                        dim = 2) # (sum_seq_len, rec_size, serp_embed_dim)
        prev_clicks = torch.cat([torch.zeros(sum_seq_lens, 1, dtype = torch.long, device = self.device), 
                                    torch.cat(batch.clicks, dim = 0)[:, :-1]], dim = 1) # (sum_seq_len, rec_size)
        click_embedd = self.click_embedd_exam(prev_clicks)           # (sum_seq_len, rec_size, click_embedd_dim)

        exam_inp = torch.cat([pos_embedd, serp_embedd, click_embedd], dim = 2)

        h = self.h0_exam.expand(-1, sum_seq_lens, -1).contiguous()
        h, _ = self.propensities(exam_inp, h)   # (sum_seq_lens, rec_size, inner_state_dim)
        exam_prob = self.fully_connected_exam(h.flatten(end_dim = 1)).squeeze() # sum_seq_lens * rec_size

        ### 4 - Combine Relevance and estimation
        #### If we want to extend to different combination functions, we need to do it here
        click_pred = relevance_pred * exam_prob
        click_pred = click_pred.reshape(sum_seq_lens, rec_size)

        ### 5 - Reshape and return

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens * rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])
        click_pred_list = [click_pred[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]

        if return_loss:
            # Get targets         
            target_click = torch.cat(batch.clicks, dim = 0).float()
            
            # Compute loss
            loss = self.loss_fn(click_pred, target_click)
            return click_pred_list, loss

        else:
            return [torch.bernoulli(seq for seq in click_pred_list)]

    def extract_relevances(self, s_u, batch):
        '''
        Extracts uncontextualized relevances for NDCG evaluation.

        Parameters :
         - s_u : FloatTensor(size = sum_docs_per_query, state_dim)
            User state, obtained without enriched user features.
         - batch : TrajectoryBatch
            Batch of queries and documents -> No user, serp and interaction features, no clicks
        
        Output :
         - attr_pred_list : list(FloatTensor(n_doc_per_query), len = batch_size)

        '''
        n_docs = len(batch.rec_lists[0][-1])

        ### 1 - Pass through Document Encoder

        s_u, url_emebdd = self.preprocess_input_rels(s_u, batch)
        positions = torch.zeros(n_docs, dtype = torch.long, device = self.device)
        serp_feat = torch.zeros(n_docs, self.serp_feat_dim, dtype = torch.long, device = self.device)
        timesteps = len(batch.rec_lists[0]) * torch.ones(n_docs, dtype = torch.long, device = self.device)

        pos_embedd = self.pos_embedd_doc(positions)   # (n_docs, pos_embedd_dim)
        if self.serp_feat_type == 'int':
            apply = lambda l, t : [el(t[:, i]) for i,el in enumerate(l)]
            serp_embedd = torch.cat(apply(self.serp_embedd_doc, serp_feat), 
                                        dim = 1) # (n_docs, serp_embed_dim)
        time_embedd = self.time_embedd_doc(timesteps)           # (n_docs, time_embedd_dim)

        doc_embedd = self.doc_encoder(torch.cat([url_embedd, pos_embedd, serp_embedd, time_embedd], dim = 1))   # (n_docs, doc_embedd_size)

        ### 2 - Pass through relevance estimator        
        relevance_pred = self.attractiveness(torch.cat([s_u, doc_embedd], dim = 1)).squeeze()       # n_docs

        return relevance_pred

class CACM_minus(PBM):
    '''
        Neural Examination-Hypothesis-Based Model, based off CACM.
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, pos_embedd_dim : int, click_embedd_dim : int, 
                    inner_state_dim : int, non_causal : bool, dropout_rate : float, **kwargs):
        super().__init__(item_embedd, dropout_rate = dropout_rate, **kwargs)

        self.propensities = None
        self.non_causal = non_causal
        self.inner_state_dim = inner_state_dim

        self.pos_embedd = Embedding(self.rec_size, pos_embedd_dim)
        self.click_embedd = Embedding(2, click_embedd_dim)


        #### Examination predictor
        self.examination = GRU(pos_embedd_dim + click_embedd_dim, # + self.serp_embedd_dim
                                hidden_size = inner_state_dim, num_layers = 1, batch_first = True, bidirectional = non_causal)
        self.fully_connected = Sequential(Linear((1 + non_causal) * inner_state_dim, 1),
                                            Sigmoid())
        self.register_buffer("h0", torch.zeros(1 + non_causal, 1, inner_state_dim)) ## How to initialize it ? 
    
    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[PBM.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        parser.add_argument('--pos_embedd_dim', type=int, default = 4)
        parser.add_argument('--click_embedd_dim', type=int, default = 4)
        parser.add_argument('--inner_state_dim', type=int, default = 64)
        parser.add_argument('--non_causal', type=parser.str2bool, default = False)
        return parser

    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        '''
            Forward pass through the click prediction head.

            Parameters : 
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Flattenned batch of user states.
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - return_loss : boolean
                If True, the click probability and the loss are returned, and if False, the sampled predicted clicks are returned

            
            Output : 
             - click_pred : list(torch.{Long or Float}Tensor(size = (seq_len, rec_size)), len = batch_size)
                Predicted clicks or click probability (see above)
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)
        rec_size = batch.rec_lists[0].size()[1]

        ### 1 - Pre-process input and compute attractiveness

        if self.discrete:
            attr_pred = Sigmoid()(s_u.flatten(end_dim = 1))
        else:
            
            s_u, rec_features = self.preprocess_input(s_u, batch)

            ### 2 - Pass through attractiveness model
            inp = torch.cat([s_u, rec_features], dim = 1).float()
            attr_pred = self.attractiveness(inp)
        
        ### 2 - Compute examination
        positions = torch.arange(rec_size, device = self.device).unsqueeze(0).expand(sum_seq_lens, -1) # (sum_seq_len, rec_size)
        serp_feat = torch.cat([sf for sf in batch.serp_feat], dim = 0)  # (sum_seq_len, rec_size, serp_feat_dim)

        pos_embedd = self.pos_embedd(positions)   # (sum_seq_len, rec_size, pos_embedd_dim)
        # if self.serp_feat_type == 'int':
        #     apply = lambda l, t : [el(t[:, :, i]) for i,el in enumerate(l)]
        #     serp_embedd = torch.cat(apply(self.serp_embedd, serp_feat), 
        #                                 dim = 2) # (sum_seq_len, rec_size, serp_embed_dim)

        if self.non_causal :
            prev_clicks = torch.cat(batch.clicks, dim = 0) # (sum_seq_len, rec_size)
        else:
            prev_clicks = torch.cat([torch.zeros(sum_seq_lens, 1, dtype = torch.long, device = self.device), 
                                        torch.cat(batch.clicks, dim = 0)[:, :-1]], dim = 1) # (sum_seq_len, rec_size)
        click_embedd = self.click_embedd(prev_clicks)           # (sum_seq_len, rec_size, click_embedd_dim)

        exam_inp = torch.cat([pos_embedd, click_embedd], dim = 2)

        h = self.h0.expand(-1, sum_seq_lens, -1).contiguous()
        h, _ = self.examination(exam_inp, h)   # (sum_seq_lens, rec_size, inner_state_dim)
        exam_prob = self.fully_connected(h.flatten(end_dim = 1)) # sum_seq_lens * rec_size

        ### 4 - Get click probability
        click_pred = attr_pred * exam_prob
        click_pred = click_pred.reshape(sum_seq_lens, rec_size)

        ### 5 - Reshape and return

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens * rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])
        click_pred_list = [click_pred[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]

        if return_loss:
            # Get targets         
            target_click = torch.cat(batch.clicks, dim = 0).float()
            
            # Compute loss
            loss = self.loss_fn(click_pred, target_click)
            return click_pred_list, loss

        else:
            return [torch.bernoulli(seq for seq in click_pred_list)]

    def compute_reward(self, s_u, batch, rels, ep = None, apc = None):
        if ep is None:  # For MaxReward we pre-compute everything not ranking-dependent, so no need to do that
            def get_all_binary_sequences(size, seq = torch.LongTensor([])):
                '''
                    This returns all possible sequences of clicks
                '''
                if len(seq) == size:
                    return seq.unsqueeze(0)

                return torch.cat([get_all_binary_sequences(size, seq = torch.cat([seq, torch.zeros(1, dtype = torch.long)])),
                                    get_all_binary_sequences(size, seq = torch.cat([seq, torch.ones(1, dtype = torch.long)]))], dim = 0)

            #### Pass RNN through all possible lists
            apc = torch.cat([torch.zeros(2 ** (self.rec_size - 1), 1, dtype = torch.long), get_all_binary_sequences(self.rec_size - 1)], dim = 1)
            pos_embedd = self.pos_embedd(torch.arange(self.rec_size)).unsqueeze(0).expand(2 ** (self.rec_size - 1), -1, -1) # (2 ** (rec_size - 1), rec_size, pos_embedd_dim)
            click_embedd = self.click_embedd(apc) # (2 ** (rec_size - 1), rec_size, pos_embedd_dim)
            
            exam_inp = torch.cat([pos_embedd, click_embedd], dim = 2)   # (2 ** (rec_size - 1), rec_size, pos_embedd_dim + click_embedd_dim)

            h = self.h0.expand(-1, 2 ** (self.rec_size - 1), -1).contiguous()
            h, _ = self.examination(exam_inp, h)   # (2 ** (rec_size - 1), rec_size, inner_state_dim)
            ep = self.fully_connected(h.flatten(end_dim = 1)).reshape(2 ** (self.rec_size - 1), self.rec_size) # (2 ** (rec_size - 1), rec_size)
            ### That's P(E=1 | C_<r)


        def build_prev_seq_prob(all_prev_clicks, exam_probs, rels):
            '''
                This gives P(C_<r = c_<r) for all 1 <= r <= self.rec_size and (c_1, ..., c_(self.rec_size -1)) for the given ranking
            '''
            prev_seq_prob = torch.ones(len(all_prev_clicks)).unsqueeze(1)
            for r in range(1, self.rec_size):
                new_prob = (1 - all_prev_clicks[:, r] + (2 * all_prev_clicks[:, r] - 1) * rels[r - 1] * exam_probs[:, r-1]) * prev_seq_prob[:, -1] 
                prev_seq_prob = torch.cat([prev_seq_prob, new_prob.unsqueeze(1)], dim = 1)
            return prev_seq_prob    # Should be of size (2 ** (self.rec_size - 1), self.rec_size)
        
        all_seqs = lambda length : torch.arange(2 ** (length - 1)) * 2 ** (self.rec_size - length)
        def get_marginal_click_prob(prev_seq_prob, exam_probs, rels):
            '''
                This gives the list [P(C_1 = 1), ... , P(C_self.rec_size = 1)] for the given ranking
            '''
            seq_idx = [all_seqs(k) for k in range(1, self.rec_size + 1)]
            marginal_exam_prob = [torch.sum(exam_probs[seq_idx[k], k] * prev_seq_prob[seq_idx[k], k]) for k in range(len(seq_idx))]

            return rels * torch.tensor(marginal_exam_prob)
        
        # P(C_r = 1) = P(A_r = 1) * sum_prevclickseqs(P(E_r = 1 |prevclickseq) * P(prevclick_seq))
        prev_seq_prob = build_prev_seq_prob(apc, ep, rels)
        click_probs = get_marginal_click_prob(prev_seq_prob, ep, rels)
        return torch.sum(click_probs)

    def max_reward_policy(self, query, docs, rels):
        '''
            Returns a policy maximizing CTR, according to the click model

            Parameters :
             - query : int
                Current query
             - docs : torch.LongTensor(n_docs_in_q)
                List of documents associated with this query
             - rels : torch.LongTensor(n_docs_in_q)
                Relevances of documents in docs
            
            Output : 
             - policy : torch.LongTensor(self.rec_size)
                Maximum reward policy
        '''
        ### Keep only top 10 most relevant documents
        top_rels, top_doc_idx = torch.sort(rels, descending = True)
        top_rels = top_rels[:self.rec_size]
        top_doc_idx = top_doc_idx[:self.rec_size]        
        
        all_rankings = torch.load("/home/rdeffaye/workspace/playground/perms.pt", map_location = self.device) 
                # All permutations of rankings (saved once to avoid recomputing it each time)
        # Sample
        sampling_rate = 900
        n = len(all_rankings)
        all_rankings = all_rankings[torch.randperm(n)[: n // sampling_rate]]

        def get_all_binary_sequences(size, seq = torch.LongTensor([])):
            '''
                This gives the list [P(C_1 = 1), ... , P(C_self.rec_size = 1)] for the given ranking
            '''
            if len(seq) == size:
                return seq.unsqueeze(0)

            return torch.cat([get_all_binary_sequences(size, seq = torch.cat([seq, torch.zeros(1, dtype = torch.long)])),
                                get_all_binary_sequences(size, seq = torch.cat([seq, torch.ones(1, dtype = torch.long)]))], dim = 0)

        
        all_prev_clicks = torch.cat([torch.zeros(2 ** (self.rec_size - 1), 1, dtype = torch.long), get_all_binary_sequences(self.rec_size - 1)], dim = 1)
        pos_embedd = self.pos_embedd(torch.arange(self.rec_size)).unsqueeze(0).expand(2 ** (self.rec_size - 1), -1, -1) # (2 ** (rec_size - 1), rec_size, pos_embedd_dim)
        click_embedd = self.click_embedd(all_prev_clicks) # (2 ** (rec_size - 1), rec_size, pos_embedd_dim)
        
        exam_inp = torch.cat([pos_embedd, click_embedd], dim = 2)   # (2 ** (rec_size - 1), rec_size, pos_embedd_dim + click_embedd_dim)

        h = self.h0.expand(-1, 2 ** (self.rec_size - 1), -1).contiguous()
        h, _ = self.examination(exam_inp, h)   # (2 ** (rec_size - 1), rec_size, inner_state_dim)
        exam_prob = self.fully_connected(h.flatten(end_dim = 1)).reshape(2 ** (self.rec_size - 1), self.rec_size) # (2 ** (rec_size - 1), rec_size)
        ### That's P(E=1 | C_<r)

        traj_reward = torch.zeros(len(all_rankings))
        for k, rl in enumerate(all_rankings):
            rels_perm = top_rels[all_rankings[k]]
            traj_reward[k] = self.compute_reward(None, None, rels_perm, ep = exam_prob, apc = all_prev_clicks)

        best_traj = docs[top_doc_idx[all_rankings[torch.argmax(traj_reward)]]]

        return best_traj, top_rels[all_rankings[torch.argmax(traj_reward)]]

class CoCM(PBM):
    '''
        Complex Click model (used solely for click prediction with oracle initialization.)
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, mismatch : bool, rel_path : str, **kwargs):
        super().__init__(item_embedd, **kwargs)

        if not self.discrete:
            raise NotImplementedError("CoCM requires pair_embedding state learner.")
        
        self.pairs_in_data = torch.load(self.data_dir + "pair_embedding_dict.pt")
        self.satisfaction = Embedding(len(self.pairs_in_data) + 1, 1).to(self.my_device)

        sat_weights = self.satisfaction.weight.data.clone()
        attr_weights = self.state_learner.user_embedd.weight.data.clone().to(self.my_device)
        rels = torch.load(self.data_dir + rel_path, map_location = self.my_device)
        for qid, rel in rels.items():
            docs_in_d = torch.tensor([did for did in range(len(rel)) if (qid, did) in self.pairs_in_data])
            pairs = torch.tensor([self.pairs_in_data[qid, did] for did in range(len(rel)) 
                    if (qid, did) in self.pairs_in_data])
            if len(docs_in_d) > 0:
                attr_weights[pairs, 0] = torch.maximum(0.02 * torch.ones_like(rel[docs_in_d]), 1.0 * rel[docs_in_d])
                sat_weights[pairs, 0] = 0.7
        self.satisfaction.weight.data.copy_(sat_weights)
        self.state_learner.user_embedd.weight.data.copy_(attr_weights)

        self.gamma = 0.9
        self.epsilons = torch.tensor([0.2 * 0.9**i for i in range(10)])

        if mismatch:
            self.mode_probs = [0.1, 0.2, 0.7]
        else:
            self.mode_probs = [0.1, 0.6, 0.3]
    
    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[PBM.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        parser.add_argument('--mismatch', type=parser.str2bool, default = False)
        parser.add_argument('--rel_path', type=str, default = "../relevances_sampled.pt")
        return parser
        
    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        
        batch_size = len(batch.seq_lens)
        sum_seq_lens = torch.sum(batch.seq_lens)
        rec_size = batch.rec_lists[0].size()[1]

        sigma = 0.7


        attractiveness = torch.maximum(0.02 * torch.ones(sum_seq_lens, rec_size), s_u.squeeze(dim = 2))
        # print("------------")
        # print(attractiveness[2])

        queries = torch.cat(batch.user_feat, dim = 0)[:, 0]
        items = torch.cat(batch.rec_lists, dim = 0)
        if self.my_device == "cpu":
            pair_idx = torch.LongTensor([[self.pairs_in_data[q.item() ,i.item()] for i in l]
                                                for (q,l) in zip(queries, items) ])
        else:
            pair_idx = torch.cuda.LongTensor([[self.pairs_in_data[q.item() ,i.item()] for i in l]
                                                for (q,l) in zip(queries, items) ])
        #satisfactions = self.satisfaction(pair_idx).squeeze(dim = 2)
        
        # no_look:
        prob_no_look = self.epsilons.unsqueeze(0).expand(sum_seq_lens, -1)

        # top-down:
        target_clicks = torch.cat(batch.clicks, dim = 0)
        prev_clicks_td = torch.cat([torch.zeros(sum_seq_lens, 1, device = self.device, dtype = torch.long), 
                                                target_clicks[:, :-1]], dim = 1)

        next_attractiveness_td = torch.cat([attractiveness[:, 1:], torch.ones(sum_seq_lens, 1, device = self.device, dtype = torch.long)], dim = 1)

        attractiveness_prod_td = attractiveness * (1 - 0.5 * next_attractiveness_td)
        attractiveness_prod_td[:, 0] = attractiveness[:, 0] * (1 - next_attractiveness_td[:, 0])

        exam_prob_td = torch.ones(sum_seq_lens, 1)
        exam_prob_td_minus_r = torch.ones(sum_seq_lens, 1, rec_size)
        prev_clicks_td_reverse = prev_clicks_td.clone().unsqueeze(2).expand(-1, -1, rec_size).clone()
        prev_clicks_td_reverse[:, torch.arange(1, rec_size), torch.arange(1, rec_size)] = 1 - prev_clicks_td[:, 1:]
        for r in range(1, rec_size):
            ep = ( prev_clicks_td[:, r] * (1 - sigma) * self.gamma + (1 - prev_clicks_td[:, r]) ) * self.gamma * exam_prob_td[:, -1]
            exam_prob_td = torch.cat([exam_prob_td, ep.unsqueeze(1)], dim = 1)

            ep_minus_r = ( prev_clicks_td[:, r].unsqueeze(1) * (1 - sigma)  * self.gamma + (1 - prev_clicks_td[:, r].unsqueeze(1)) ) * self.gamma * exam_prob_td_minus_r[:, -1, :]
            exam_prob_td_minus_r = torch.cat([exam_prob_td_minus_r, ep_minus_r.unsqueeze(1)], dim = 1)
            ep_reverse = ( (1 - prev_clicks_td[:, r]) * (1 - sigma)  * self.gamma + (prev_clicks_td_reverse[:, r, r]) ) * self.gamma * exam_prob_td_minus_r[:, -1, r]
            exam_prob_td_minus_r[:, -1, r - 1] = ep_reverse
        
        
        prob_top_down = (prev_clicks_td == 0) * attractiveness_prod_td * exam_prob_td
        prob_top_down[:, -1] = (prev_clicks_td[:, -1] == 1)

        prob_top_down_reverse = torch.stack([(prev_clicks_td_reverse[:, :, r] == 0) * attractiveness_prod_td * exam_prob_td_minus_r[:, :, r] for r in range(rec_size)], dim = 2)
        prob_top_down_reverse[:, -1] = (prev_clicks_td_reverse[:, -1] == 1)
        
        #print(prob_top_down_reverse[2])

        # bottom-up:
        prev_clicks_bu = torch.cat([target_clicks[:, 1:], torch.zeros(sum_seq_lens, 1, device = self.device, dtype = torch.long)], dim = 1)

        next_attractiveness_bu = torch.cat([torch.ones(sum_seq_lens, 1, device = self.device, dtype = torch.long), attractiveness[:, 1:]], dim = 1)

        attractiveness_prod_bu = attractiveness * (1 - 0.5 * next_attractiveness_bu)
        attractiveness_prod_bu[:, -1] = attractiveness[:, -1] * (1 - next_attractiveness_bu[:, -2])

        exam_prob_bu = torch.ones(sum_seq_lens, 1)
        exam_prob_bu_minus_r = torch.ones(sum_seq_lens, 1, rec_size)
        prev_clicks_bu_reverse = prev_clicks_bu.clone().unsqueeze(2).expand(-1, -1, rec_size).clone()
        prev_clicks_bu_reverse[:, torch.arange(1, rec_size), torch.arange(1, rec_size)] = 1 - prev_clicks_bu[:, 1:]
        for r in range(2, rec_size + 1):
            ep = ( prev_clicks_bu[:, rec_size - r] * (1 - sigma) * self.gamma + (1 - prev_clicks_bu[:, rec_size - r]) ) * self.gamma * exam_prob_bu[:, 0]
            exam_prob_bu = torch.cat([ep.unsqueeze(1), exam_prob_bu], dim = 1)

            ep_minus_r = ( prev_clicks_bu[:, rec_size - r].unsqueeze(1) * (1 - sigma) * self.gamma + (1 - prev_clicks_bu[:, rec_size - r].unsqueeze(1)) ) * self.gamma * exam_prob_bu_minus_r[:, 0, :]
            exam_prob_bu_minus_r = torch.cat([ep_minus_r.unsqueeze(1), exam_prob_bu_minus_r], dim = 1)
            ep_reverse = ( (1 - prev_clicks_bu[:, rec_size - r]) * (1 - sigma) * self.gamma + (prev_clicks_bu[:, rec_size - r]) ) * self.gamma * exam_prob_bu_minus_r[:, 0, rec_size - r]
            exam_prob_bu_minus_r[:, 0, rec_size - r + 1] = ep_reverse
        
        prob_bottom_up = (prev_clicks_bu == 0) * attractiveness_prod_bu * exam_prob_bu
        prob_bottom_up[:, 0] = (prev_clicks_bu[:, 0] == 1)

        prob_bottom_up_reverse = torch.stack([(prev_clicks_bu_reverse[:, :, r] == 0) * attractiveness_prod_bu * exam_prob_bu_minus_r[:, :, r] for r in range(rec_size)], dim = 2)
        prob_bottom_up_reverse[:, 0] = (prev_clicks_bu_reverse[:, 0] == 1)

        prior_mode = self.mode_probs
        
        # Let's compute the posterior probability of each mode given the page except the current page
        target_clicks = target_clicks.float()
        target_clicks_reverse = target_clicks.clone().unsqueeze(2).expand(-1, -1, rec_size).clone()
        target_clicks_reverse[:, torch.arange(rec_size), torch.arange(rec_size)] = 1 - target_clicks


        ll_rank_no_look = target_clicks * prob_no_look + (1 - target_clicks) * (1 - prob_no_look)
        likelihood_no_look = torch.stack([torch.exp(torch.sum(torch.log(torch.cat([ll_rank_no_look[:, :r], ll_rank_no_look[:, r+1:]], dim = 1)), 
                                                                dim = 1)) for r in range(0, rec_size)], 
                                            dim = 1)        # (sum_seq_lens, rec_size)
        ll_rank_td = target_clicks * prob_top_down + (1 - target_clicks) * (1 - prob_top_down)
        likelihood_td = torch.stack([torch.exp(torch.sum(torch.log(ll_rank_td), 
                                                                dim = 1)) for r in range(0, rec_size)], 
                                            dim = 1)        # (sum_seq_lens, rec_size)
        ll_rank_td_reverse = target_clicks_reverse * prob_top_down_reverse + (1 - target_clicks_reverse) * (1 - prob_top_down_reverse)
        likelihood_td_reverse = torch.exp(torch.sum(torch.log(ll_rank_td_reverse), 
                                                                dim = 1))

        likelihood_td = target_clicks * (prob_top_down * likelihood_td + (1 - prob_top_down) * likelihood_td_reverse) + \
                            (1 - target_clicks) * (prob_top_down * likelihood_td_reverse + (1 - prob_top_down) * likelihood_td)


        ll_rank_bu = target_clicks * prob_bottom_up + (1 - target_clicks) * (1 - prob_bottom_up)
        likelihood_bu = torch.stack([torch.exp(torch.sum(torch.log(ll_rank_bu), 
                                                                dim = 1)) for r in range(0, rec_size)], 
                                            dim = 1)        # (sum_seq_lens, rec_size)
        ll_rank_bu_reverse = target_clicks_reverse * prob_bottom_up_reverse + (1 - target_clicks_reverse) * (1 - prob_bottom_up_reverse)
        likelihood_bu_reverse = torch.exp(torch.sum(torch.log(ll_rank_bu_reverse), 
                                                                dim = 1))

        likelihood_bu = target_clicks * (prob_bottom_up * likelihood_bu + (1 - prob_bottom_up) * likelihood_bu_reverse) + \
                            (1 - target_clicks) * (prob_bottom_up * likelihood_bu_reverse + (1 - prob_bottom_up) * likelihood_bu)
        
        
        posterior_no_look = prior_mode[0] * likelihood_no_look
        posterior_top_down = prior_mode[1] * likelihood_td
        posterior_bottom_up = prior_mode[2] * likelihood_bu
        posteriors = torch.stack([posterior_no_look, posterior_top_down, posterior_bottom_up], dim = 2)
        posterior_no_look /= torch.sum(posteriors, dim = 2)
        posterior_top_down /= torch.sum(posteriors, dim = 2)
        posterior_bottom_up /= torch.sum(posteriors, dim = 2)

        click_prob = posterior_no_look * prob_no_look + posterior_top_down * prob_top_down + posterior_bottom_up * prob_bottom_up

        #print(click_prob[2])

        if return_loss:
            loss = self.loss_fn(click_prob, target_clicks)
            return click_prob, loss

        else:
            return [torch.bernoulli(seq for seq in click_prob)]

    def extract_relevances(self, s_u, batch):
        return super().extract_relevances(s_u, batch)

class RecurrentModule(torch.nn.Module):
    '''
        Recurrent module, used in NCM and AICM.
    '''
    def __init__(self, recurrent_type : str, inner_state_dim : int, output : str, attention : bool,
                    num_layers_RNN : int, embedd_dim : int, state_dim : int, serp_feat : bool, 
                    num_serp_embedd : List[int], serp_embedd_dim : List[int], serp_feat_type : str, 
                    serp_feat_dim : int, serp_state_dim : int, device : torch.device, dropout_rate : float,
                    inter_state_dim : int = 1, **kwargs):
        super().__init__()
        self.embedd_dim = embedd_dim
        self.state_dim = state_dim
        self.device = device
        self.serp_feat = serp_feat
        self.inter_state_dim = inter_state_dim


        if serp_feat:
            self.serp_state_dim = serp_state_dim
            if serp_feat_dim == 0:
                raise ValueError("There are no serp features in the dataset. Please set serp_feat = False.")
            if serp_feat_type == 'int':
                if serp_feat_dim == 1:
                    self.serp_encoding = Embedding(num_serp_embedd[0], serp_state_dim).to(device)
                else:
                    self.serp_embeddings = [Embedding(num_serp_embedd[i], serp_embedd_dim[i]).to(
                                                device) for i in range(serp_feat_dim)]
                    self.lin_layer = Linear(torch.sum(torch.LongTensor(serp_embedd_dim)), serp_state_dim)
        
                    self.serp_encoding = lambda x : self.lin_layer(torch.cat([embeddings(x[:, :, i]) 
                                            for i,embeddings in enumerate(self.serp_embeddings)], dim = 2))
            else:
                raise NotImplementedError("We only support categorical serp features for now.")
        else:
            self.serp_state_dim = 0

        self.register_buffer("h0", torch.zeros(1, 1, inner_state_dim, device = device))
        self.recurrent_type = recurrent_type
        if recurrent_type == "LSTM":
            self.register_buffer("c0", torch.zeros(1, 1, inner_state_dim, device = device))
            self.rnn = LSTM(state_dim + embedd_dim + self.serp_state_dim + self.inter_state_dim,
                                                    hidden_size = inner_state_dim, num_layers = num_layers_RNN, batch_first = True)
            self.hidden_states = [self.h0, self.c0]
        elif recurrent_type == "GRU":
            self.rnn = GRU(state_dim + embedd_dim + self.serp_state_dim + self.inter_state_dim,
                                                    hidden_size = inner_state_dim, num_layers = num_layers_RNN, batch_first = True)
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
    
    def self_attention(self, h):
        '''
        h : torch.Tensor(batch_size, seq_len, hidden_size)

        Output att : torch.tensor(batch_size, seq_len, hidden_size)
        '''
        gram_matrices = torch.matmul(h, h.transpose(1,2)) # (batch_size, seq_len, seq_len)
        alphas = Softmax(dim = 2)(gram_matrices) # (batch_size, seq_len, seq_len)
        useful_alphas = torch.stack([torch.tril(alphas[batch_idx]) for batch_idx in range(len(h))], dim = 0)
        expanded_h =  h.unsqueeze(1).expand(-1, h.size()[1], -1, -1)  # (batch_size, seq_len, seq_len, hidden_size)
        ## We have expanded_h[batch_idx, *, i, :] = h_{i,batch_idx}
        unsqueezed = useful_alphas.unsqueeze(3)
        expanded_h = unsqueezed * expanded_h # (batch_size, seq_len, seq_len, hidden_size)
        att = torch.sum(expanded_h, dim = 2) # (batch_size, seq_len, hidden_size) 

        return att

    def sample(self, click_prob):
        if self.output == "softmax":
            sampled_clicks = torch.distributions.Categorical(click_prob).sample().unsqueeze(1)
        elif self.output == "sigmoid":
            sampled_clicks = torch.bernoulli(click_prob)
        else:
            raise ValueError("self.output is unknown.")
        return sampled_clicks

    def forward(self, s_u, rec_features, batch, input_clicks = "previous_target", h = None):
        '''
            Forward pass through the recurrent model. Constructs input and passes it through the model
            either conditonnaly to previous predicted click or previous observed click.

            Parameters:
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Flattenned batch of user states.
             - rec_features : torch.FloatTensor(sum_seq_lens, rec_size, embedd_dim)
                Flattenned batch of item embeddings.
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - input_clicks : str, choices = ["previous_target", "previous_pred", "current_target"]
                Defines what should go into the click-related part of the input.
            
            Output:
             - click_probs : torch.FloatTensor(sum_seq_lens, rec_size, output_size)
                Predicted click probabilities.
             - click_seq_pred : torch.LongTensor(sum_seq_lens, rec_size)
                Predicted click sequences (only for input_clicks = "previous_pred")

        '''        
        sum_seq_lens = len(rec_features)
        rec_size = rec_features.size()[1]

        inp = torch.cat([torch.zeros(sum_seq_lens, rec_size, self.state_dim, device = self.device), 
                            rec_features], dim = 2).float()
        if self.serp_feat:
            serp_state = self.serp_encoding(torch.cat(batch.serp_feat, dim = 0).squeeze())
            if len(serp_state.size()) == 2: # When we evaluate relevance
                serp_state = serp_state.unsqueeze(1)
            inp = torch.cat([inp, serp_state], dim = 2)

        ### 2 - Pass through recurrent model
        click_probs = torch.empty(sum_seq_lens, rec_size, self.output_size, 
                                    dtype = torch.float, device = self.device) ## (sum_seq_lens, rec_size)
        if input_clicks == "current_target":
            target_clicks = torch.cat(batch.clicks, dim = 0).float().unsqueeze(2) ## (sum_seq_lens, rec_size, 1)
            inp_click = target_clicks[:, 0]
        else : 
            if input_clicks == "previous_target" : 
                target_clicks = torch.cat(batch.clicks, dim = 0).float().unsqueeze(2) ## (sum_seq_lens, rec_size, 1)
            elif input_clicks == "previous_pred" : 
                click_seq_pred = torch.empty(sum_seq_lens, rec_size, dtype = torch.long, device = self.device)
            else:
                raise NotImplementedError("This type of input clicks is not supported yet.")

            inp_click = torch.zeros(sum_seq_lens, 1, device = self.device)            


        if h is None:
            if self.recurrent_type == "LSTM":
                hidden = tuple(map(lambda t : t.expand(-1, sum_seq_lens, -1).contiguous(), self.hidden_states))
            else:
                hidden = self.hidden_state.expand(-1, sum_seq_lens, -1).contiguous()
            outp, hidden = self.rnn(torch.cat([s_u, 
                                    torch.zeros(sum_seq_lens, 1, self.embedd_dim + self.serp_state_dim + 1, 
                                                    device = self.device)], dim = 2), hidden)
        else:
            if self.recurrent_type == "LSTM":
                hidden = [h, h]
            else:
                hidden = h
        for r in range(rec_size):
            outp, hidden = self.rnn(torch.cat([inp[:, r, :], inp_click], dim = 1).unsqueeze(1), hidden)
            if self.attention:
                outp = self.self_attention(outp)
            click_prob = self.output_head(outp.squeeze())   #sum_seq_len, output_size
            if input_clicks == "current_target":
                inp_click = target_clicks[:, min(r+1, rec_size - 1)] #sum_seq_len, 1
            elif input_clicks == "previous_pred":
                inp_click = self.sample(click_prob.detach())    #sum_seq_len, 1
                click_seq_pred[:, r] = inp_click.squeeze()
            elif  input_clicks == "previous_target":
                if target_clicks.size()[1] != 0:
                    inp_click = target_clicks[:, r] # (sum_seq_len, 1)
            click_probs[:, r] = click_prob

        if input_clicks == "previous_pred":
            return click_seq_pred, click_probs, hidden
        else:
            return None, click_probs.squeeze(2), hidden

class NCM(ClickModel):
    '''
        Neural Click Model, [Borisov et al., 2016]
    '''
    def __init__(self, item_embedd : torch.nn.Embedding, recurrent_type : str, inner_state_dim : int, state_dim : int, 
                    serp_feat : bool, non_causal : bool, **kwargs):
        super().__init__(item_embedd, state_dim = state_dim, **kwargs)

        self.loss_fn = BCELoss(reduction = 'none')

        # For logging
        self.recurrent_type = recurrent_type
        self.inner_state_dim = inner_state_dim
        self.non_causal = non_causal

        self.recurrent_module = RecurrentModule(output = "sigmoid", recurrent_type = recurrent_type, 
                                                inner_state_dim = inner_state_dim, state_dim = self.state_dim, 
                                                serp_feat = serp_feat, **kwargs)      
    
    @staticmethod
    def add_model_specific_args(parent_parser, state_learner):
        parser = MyParser(parents=[ClickModel.add_model_specific_args(parent_parser, state_learner)], add_help=False)
        parser.add_argument('--recurrent_type', type=str, default = "GRU")
        parser.add_argument('--attention', type=parser.str2bool, default = False)
        parser.add_argument('--inner_state_dim', type=int, default = 64)
        parser.add_argument('--serp_feat', type=parser.str2bool, default = False)
        parser.add_argument('--non_causal', type=parser.str2bool, default = False)

        arguments = [action.option_strings[0] for action in parser._actions]
        if '--num_layers_RNN' not in arguments:
            parser.add_argument('--num_layers_RNN', type=int, default = 1)
        if '--serp_embedd_dim' not in arguments :
            parser.add_argument('--serp_state_dim', type=int, default = 32)
            parser.add_argument('--serp_embedd_dim', type=int, nargs = '+', default = [16, 16, 8])

        return parser 

    def forward_click_pred_head(self, s_u, batch, return_loss = True):
        '''
            Forward pass through the click prediction head.

            Parameters : 
             - s_u : torch.Tensor(float, size = (sum_seq_lens, state_dim)))
                Flattenned batch of user states.
             - batch : TrajectoryBatch
                Batch of Trajectories.
             - return_loss : boolean
                If True, the click probability and the loss are returned, and if False, the sampled predicted clicks are returned

            
            Output : 
             - click_pred : list(torch.{Long or Float}Tensor(size = (seq_len, rec_size)), len = batch_size)
                Predicted clicks or click probability (see above)
             - loss : torch.Tensor(float, size = (,))
                Sum of loss function on the batch
            
        '''
        ### 1 - Pre-process input
        s_u, rec_features = self.preprocess_input(s_u, batch, extend_su = False, collapse_seq = False)

        ### 2 - Pass through model
        click_seq_pred, click_probs, _ = self.recurrent_module(s_u.unsqueeze(1), rec_features, batch, 
                                        input_clicks = "previous_target" if return_loss else "previous_pred")
        

        ### 3 - Compute loss
        if return_loss:
            target_clicks = torch.cat(batch.clicks, dim = 0).float()
            loss = self.loss_fn(click_probs, target_clicks)
        
        
        ### 4 - Reshape and return

        # Reshape prediction to get sequences back
        # click_pred : tensor(size = (sum_seq_lens, rec_size)))
        #            --> list(tensor(size = (seq_len, rec_size)), len = batch_size)
        cum_lens = torch.cat([torch.LongTensor([0]), torch.cumsum(batch.seq_lens, dim = 0)])

        if return_loss:
            click_probs = [click_probs[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]
            return click_probs, loss
        else:
            click_seq_pred = [click_seq_pred[cum_lens[i]:cum_lens[i+1]] for i in range(len(cum_lens) - 1)]
            return click_seq_pred

    def extract_relevances(self, s_u, batch):
        '''
        Extracts uncontextualized relevances for NDCG evaluation.

        Parameters :
         - s_u : FloatTensor(size = sum_docs_per_query, state_dim)
            User state, obtained without enriched user features.
         - batch : TrajectoryBatch
            Batch of queries and documents -> No user, serp and interaction features, no clicks
        
        Output :
         - attr_pred_list : list(FloatTensor(n_doc_per_query), len = batch_size)

        '''
        if not self.state_learner.only_query:
            raise ValueError("Please disable user features for uncontextualized NDCG evaluation.")

        s_u, rec_features = self.preprocess_input_rels(s_u, batch)

        ### 2 - Pass through model
        _, click_probs, _ = self.recurrent_module(s_u.unsqueeze(1), rec_features.unsqueeze(1), batch, input_clicks = "previous_pred")

        return click_probs.squeeze()

    def compute_reward(self, s_u, batch, rels, rf = None, apc = None):
        if rf is None:
            s_u, rf = self.preprocess_input(s_u.unsqueeze(0).unsqueeze(1), batch, extend_su = False, collapse_seq = False)
            def get_all_binary_sequences(size, seq = torch.LongTensor([])):
                '''
                    This returns all possible sequences of clicks
                '''
                if len(seq) == size:
                    return seq.unsqueeze(0)

                return torch.cat([get_all_binary_sequences(size, seq = torch.cat([seq, torch.zeros(1, dtype = torch.long)])),
                                    get_all_binary_sequences(size, seq = torch.cat([seq, torch.ones(1, dtype = torch.long)]))], dim = 0)

            #### Get all possible clicks
            apc = torch.cat([torch.zeros(2 ** (self.rec_size - 1), 1, dtype = torch.long), 
                                                    get_all_binary_sequences(self.rec_size - 1)], dim = 1)
        
        all_seqs = lambda length : torch.arange(2 ** (length - 1)) * 2 ** (self.rec_size - length)
        def forward_all_seqs(s_u, rec_feat, all_prev_clicks):
            '''
                This stores P(C_r = 1 | c_<r) for all r and c_<r
            '''
            seq_idx = [all_seqs(k) for k in range(1, self.rec_size + 1)]
            # (batch_size, rec_size, emebedd_size)
            rec_feat_seq = [rec_feat[:, k, :].repeat_interleave(len(seq_idx[k]), dim = 0).unsqueeze(1) for k in range(len(seq_idx))]
            batches = [TrajectoryBatch(None, None, None, [all_prev_clicks[seq_idx[k].repeat(len(rec_feat)), k].unsqueeze(1)], None, None, None)
                                    for k in range(self.rec_size)]

            hidden = None
            cond_click_probs = []
            for k, seq_k in enumerate(seq_idx):
                if k > 0:
                    hidden = hidden.repeat_interleave(2, dim = 1)
                _, cp, hidden = self.recurrent_module(s_u, rec_feat_seq[k], batches[k], input_clicks = "previous_target", h = hidden)
                cond_click_probs.append(cp)
            
            return cond_click_probs
        
        def get_prev_seq_prob(cond_click_probs):
            '''
                This stores P(C_<r = c_<r) for all r and c_<r
            '''
            prev_seq_probs = [cond_click_probs[0]]
            
            for r in range(1, self.rec_size):
                prev_seq_prob_r = prev_seq_probs[-1].repeat_interleave(2, dim = 0) * cond_click_probs[r]
                prev_seq_probs.append(prev_seq_prob_r)
            return prev_seq_probs
        
        batch_size = len(rf)
        #print(batch_size)
        cond_click_probs = forward_all_seqs(s_u, rf, apc)   # list[ (batch_size * 2 ** k), len = rec_size]
        # print(len(cond_click_probs))
        # print([len(ccp) for ccp in cond_click_probs])
        prev_seq_probs = get_prev_seq_prob(cond_click_probs) # list[ (batch_size * 2 ** k), len = rec_size]
        # print(len(prev_seq_probs))
        # print([len(ccp) for ccp in prev_seq_probs])
        prev_seq_probs = [psp.reshape(batch_size, 2 ** k) for k, psp in enumerate(prev_seq_probs)]  # list[ (batch_size, 2 ** k), len = rec_size]
        # print(len(prev_seq_probs))
        # print([ccp.shape for ccp in prev_seq_probs])
        marginal_click_probs = torch.stack([torch.sum(psp, dim = 1) for psp in prev_seq_probs], dim = 1)    # (batch_size, rec_size)
        return torch.sum(marginal_click_probs, dim = 1).squeeze()

    def max_reward_policy(self, query, docs, rels):
        '''
            Returns a policy maximizing CTR, according to the click model

            Parameters :
             - query : int
                Current query
             - docs : torch.LongTensor(n_docs_in_q)
                List of documents associated with this query
             - rels : torch.LongTensor(n_docs_in_q)
                Relevances of documents in docs
            
            Output : 
             - policy : torch.LongTensor(self.rec_size)
                Maximum reward policy
        '''
        ### Keep only top 10 most relevant documents
        top_rels, top_doc_idx = torch.sort(rels, descending = True)
        top_rels = top_rels[:self.rec_size]
        top_doc_idx = top_doc_idx[:self.rec_size]        
        
        all_rankings = torch.load("/home/rdeffaye/workspace/playground/perms.pt", map_location = self.device) 
                # All permutations of rankings (saved once to avoid recomputing it each time)
        # Sample
        sampling_rate = 900
        n = len(all_rankings)
        all_rankings = all_rankings[torch.randperm(n)[: n // sampling_rate]]
        batch_size = 42
        su_batch = TrajectoryBatch(torch.ones(1), None, None, None, [torch.tensor([[query]], device = self.device)], None, None)
        s_u = self.state_learner.forward(su_batch).squeeze() # state_dim
        s_u = s_u.unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1)

        ### 1 - Pre-process input
        rec_lists = torch.tensor([self.pairs_in_data[query, did.item()] for did in docs[top_doc_idx]])
        rec_features = self.item_embedd(rec_lists)


        def get_all_binary_sequences(size, seq = torch.LongTensor([])):
            '''
                This returns all possible sequences of clicks
            '''
            if len(seq) == size:
                return seq.unsqueeze(0)

            return torch.cat([get_all_binary_sequences(size, seq = torch.cat([seq, torch.zeros(1, dtype = torch.long)])),
                                get_all_binary_sequences(size, seq = torch.cat([seq, torch.ones(1, dtype = torch.long)]))], dim = 0)

        #### Get all possible click sequences
        all_prev_clicks = torch.cat([torch.zeros(2 ** (self.rec_size - 1), 1, dtype = torch.long), 
                                                get_all_binary_sequences(self.rec_size - 1)], dim = 1)

        traj_reward = torch.zeros(len(all_rankings))
        for b in range(len(all_rankings) // batch_size):
            rec_feat = rec_features[all_rankings[b * batch_size : (b+1) * batch_size , :]] # (batch_size, rec_size, emebedd_size)

            traj_reward[b * batch_size : (b+1) * batch_size] = self.compute_reward(s_u, None, None, rf = rec_feat, apc = all_prev_clicks)

            #print(traj_reward[b * batch_size : (b+1) * batch_size])

        best_traj = docs[top_doc_idx[all_rankings[torch.argmax(traj_reward)]]]

        return best_traj, top_rels[all_rankings[torch.argmax(traj_reward)]]
