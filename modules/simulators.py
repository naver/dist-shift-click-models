import torch
import torch.distributions as D
from typing import List, Dict

from tqdm import tqdm
from pathlib import Path

from modules.argument_parser import MyParser
from modules.data_handler import get_file_name_sim


class Simulator():
    '''
        Base class for simulators
    '''
    def __init__(self, num_items : int, rec_size : int, seed : int, dataset_name : str, filename : str, **kwargs):
        self.num_items = num_items
        self.rec_size = rec_size
        self.seed = seed
        self.dataset_name = dataset_name
        self.filename = filename
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_items', type=int, default=5000)
        parser.add_argument('--rec_size', type=int, default=10)
        parser.add_argument('--filename', type=str, default=None)
        return parser

    def set_policy(self, MyPolicy, kwargs):
        self.logging_policy = MyPolicy(**{**kwargs, **vars(self)})

    def generate_dataset(self, n_sess, path, chunksize = 0):
        '''
            Builds a dataset for click model training, with 'policy' as the logging policy.

            Parameters :
            - n_sess : int
                Size of the dataset (in number of sessions !)
            - path : string
                Name of the file to save
            - chunksize : int
                Size of each chunk of data. If chunksize = 0, data is saved in one chunk

        '''
        
        dataset = {}
        chunk_count = 1
        args_dict = {**self.__dict__, **self.logging_policy.__dict__}
        if self.dataset_name is None:
            dataset_name = get_file_name_sim(self.__class__.__name__, self.logging_policy.__class__.__name__, 
                                            n_sess = n_sess, **args_dict)
        else:
            dataset_name = self.dataset_name
        if self.filename is None:
            filename = "data"
        else:
            filename = self.filename

        for sess_id in tqdm(range(n_sess)):
            ## Get initial query and/or recommendation
            rl, sf, cl, inf, uf, info = self.reset()
            done = False

            rec_lists, clicks, user_feat, serp_feat, inter_feat = [], [], [], [], []
            while not done:
                ## Store user features
                if uf is not None:
                    user_feat.append(uf)
                else:
                    user_feat = None
                ## Return a ranking
                rl, sf = self.logging_policy.forward(rl, sf, cl, inf, uf, info)
                ## Let the simulated user interact with the ranking
                cl, inf, uf, done, info = self.step(rl, sf)
                ## Store the interaction
                if (-1 in rl):
                    rec_lists.append(info["oracle_rl"])
                else:
                    rec_lists.append(rl)
                clicks.append(cl)
                if sf is not None:
                    serp_feat.append(sf)
                else:
                    serp_feat = None
                if inf is not None:
                    inter_feat.append(inf)
                else :
                    inter_feat = None
                
            ## Form session dictionary
            sess = {"rec_lists"     : torch.stack(rec_lists),
                        "clicks"    : torch.stack(clicks),
                        "user_feat" : None if user_feat is None else torch.stack(user_feat),
                        "serp_feat" : None if serp_feat is None else torch.stack(serp_feat),
                        "inter_feat": None if inter_feat is None else torch.stack(inter_feat)}
            ## Add it to dataset
            dataset[sess_id] = sess
            
            if chunksize * sess_id != 0 and sess_id % chunksize == 0:
                Path(path + dataset_name).mkdir(parents=True, exist_ok=True)
                torch.save(dataset, path + dataset_name + "/" + filename + str(chunk_count) + ".pt")
                dataset = {}
                chunk_count +=1

        Path(path + dataset_name).mkdir(parents=True, exist_ok=True)
        if chunk_count == 1:
            torch.save(dataset, path + dataset_name + "/" + filename + ".pt")
        else:
            torch.save(dataset, path + dataset_name + "/" + filename + str(chunk_count) + ".pt")

    
    def reset(self):
        raise NotImplementedError("Please instantiate a child class.")
    
    def step(self, rec_list, serp_feat, oracle = False):
        raise NotImplementedError("Please instantiate a child class.")

class SemiSyntheticSearchSim(Simulator):
    '''
        Simulator using relevance labels for given query/document pairs
    '''
    def __init__(self, data_dir : str, distrib : str, max_query : int, frequencies : str, sampled : bool, 
                    cm_type : str, DBN_sigma : float, **kwargs):
        super().__init__(**kwargs)

        if frequencies is not None:
            self.frequencies = frequencies
            obs_freq_q = torch.load(data_dir + frequencies)

        self.cm_type = cm_type
        if cm_type == "DBN":
            click_model_params : dict = {'type':'DBN','gamma': 0.9,'alpha': 0.95,'sigma':DBN_sigma, 'vert_boost':[0.7, 1.0, 1.2, 0.8, 1.3]}
        elif cm_type == "PBM":
            click_model_params : dict = {'type':'PBM','alpha' : 0.7, 'gamma': [0.9**i for i in range(10)]}
        elif cm_type == "CoCM":
            click_model_params : dict = {'type':'CoCM','gamma': 0.9, 'alpha': 1,'sigma': 0.7, 
                    'epsilon' : [0.2 * 0.9**i for i in range(10)], 'prob': [0.1, 0.6, 0.3]}
        elif cm_type == "CoCM_mismatch":
            click_model_params : dict = {'type':'CoCM','gamma': 0.9, 'alpha': 1,'sigma': 0.7, 
                    'epsilon' : [0.2 * 0.9**i for i in range(10)], 'prob': [0.1, 0.2, 0.7]}
        else:
            raise NotImplementedError("This type of internal click model has not been implemented yet.")



        self.sampled = sampled
        if sampled:
            self.relevances = torch.load(data_dir + "relevances_sampled.pt")
        else:
            self.relevances = torch.load(data_dir + "relevances.pt")
        if frequencies is None:
            self.num_items = torch.load(data_dir + "dimensions.pt")
        else:
            self.relevances = {qid : self.relevances[qid] for qid in obs_freq_q.keys()}
            self.num_items = {qid : len(self.relevances[qid]) for qid in obs_freq_q.keys()}
        self.num_queries = len(self.relevances)
        if max_query > 0:
            self.num_items = self.num_items[:max_query]
            self.num_queries = max_query
        self.cmp = click_model_params

        self.distrib = distrib
        self.idx2qid = {i : qid for i, qid in enumerate(self.relevances.keys())}
        if distrib == "uniform":
            probs = torch.ones(self.num_queries)
        elif distrib.startswith("gaussian"):
            sigma = float(distrib[8:])
            shuffle_q = torch.randperm(self.num_queries)
            x0 = self.num_queries / 2
            probs = torch.exp(((torch.ones(self.num_queries) - x0) / (2 * sigma)).pow(2))
            probs = probs[shuffle_q]
        elif distrib == "zipf":
            shuffle_q = torch.randperm(self.num_queries)
            probs = 1 / torch.arange(1, self.num_queries + 1)
            probs = probs[shuffle_q]
        elif distrib.startswith("powerlaw"):
            alpha = float(distrib[8:]) ## default alpha = 1.12
            shuffle_q = torch.randperm(self.num_queries)
            probs = torch.arange(1, self.num_queries + 1).pow(- alpha)
            probs = probs[shuffle_q]
        elif distrib.startswith("fromfreq"):
            self.idx2qid = {i : qid for i, qid in enumerate(obs_freq_q.keys())}
            probs = torch.stack([torch.sum(val) for val in obs_freq_q.values()])
        else:
            raise NotImplementedError("This type of query distribution has not been implemented yet.")
        
        self.query_distrib = D.categorical.Categorical(probs)

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[Simulator.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--distrib', type=str, default="powerlaw1.12")
        parser.add_argument('--max_query', type = int, default = 0)
        parser.add_argument('--sampled', type = parser.str2bool, default = True)
        parser.add_argument('--cm_type', type = str, choices = ["PBM", "DBN", "CoCM", "CoCM_mismatch"], required = True)
        parser.add_argument('--DBN_sigma', type = int, default = 0.9)
        return parser
    
    def click_model(self, rels, verticals, params):
        clicks = torch.zeros(self.rec_size, dtype = torch.bool)
        if params['type']=='DBN':   
            alpha = params['alpha']
            sigma = params['sigma']
            gamma = params['gamma']
            if verticals is None:
                vert_boost = [1.0] * self.rec_size
            else:
                vert_boost = [params['vert_boost'][verticals[r]] for r in range(self.rec_size)]
            
            satisfied=False
            for r in range(self.rec_size):
                clicks[r] = (torch.rand(1) < max(0.02, alpha * vert_boost[r] * rels[r]))
                if clicks[r]:
                    satisfied = (torch.rand(1) < sigma * rels[r])
                if satisfied or (torch.rand(1) > gamma):  
                    break
        elif params['type'] == 'PBM':
            gamma = torch.tensor(params['gamma'])
            alpha = params['alpha']

            clicks = alpha * rels / 4 * gamma
            clicks = torch.bernoulli(clicks)
        elif params['type'] == "CoCM":
            alpha = params['alpha']
            sigma = params['sigma']
            gamma = params['gamma']          
            probs = params['prob']

            rd = torch.rand(1)
            if rd < probs[0]:
                clicks = torch.tensor(params['epsilon'])
                clicks = torch.bernoulli(clicks)
            elif rd < probs[0] + probs[1]:
                attr = torch.maximum(0.02 * torch.ones(self.rec_size), alpha * rels)
                exam = 1.0
                clicks[0] = (torch.rand(1) < attr[0] * (1 - attr[1]))
                for r in range(1, self.rec_size - 1):
                    if clicks[r-1]:
                        if (torch.rand(1) > 1 - sigma):
                            break
                        exam = gamma * exam
                    else:
                        exam = gamma * exam
                        clicks[r] = (torch.rand(1) < attr[r] * (1 - 0.5 * attr[r+1]) * exam)
                
                if clicks[-2]:
                    exam = gamma * exam
                    clicks[-1] = attr[-1] * exam

            else:
                attr = torch.maximum(0.02 * torch.ones(self.rec_size), alpha * rels)
                exam = 1.0
                clicks[-1] = (torch.rand(1) < attr[-1] * (1 - attr[-2]))
                for r in range(self.rec_size - 2, 0, -1):
                    if clicks[r+1]:
                        if (torch.rand(1) > 1 - sigma):
                            break
                        exam = gamma * exam
                    else:
                        exam = gamma * exam
                        clicks[r] = (torch.rand(1) < attr[r] * (1 - 0.5 * attr[r-1]) * exam)

                if clicks[1]:
                    exam = gamma * exam
                    clicks[0] = attr[0] * exam

        return clicks.long()

    def reset(self):

        query = self.query_distrib.sample()
        self.query = self.idx2qid[query.item()]

        return None, None, None, None, torch.tensor([self.query]), None
    
    def step(self, rec_list, serp_feat):
        info = {} 
        oracle = (-1 in rec_list)

        if oracle:
            true_rels = torch.sort(self.relevances[self.query], descending = True)[0][:self.rec_size]
        else:
            true_rels = torch.empty(self.rec_size, dtype = torch.float)

        relevances = torch.where(rec_list == -1, true_rels, self.relevances[self.query][rec_list])

        
        if serp_feat is None:
            verticals = None
        else:
            verticals = serp_feat.squeeze().tolist()

        clicks = self.click_model(relevances, verticals, self.cmp)

        return clicks, None, None, True, info

class LoggingPolicy():
    '''
        Base class for logging policies
    '''
    def __init__(self, num_items : Dict, rec_size : int, **kwargs):
        self.num_items = num_items
        self.rec_size = rec_size
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[parent_parser], add_help=False)
        return parser
    
    def forward(self, rec_list, serp_feat, clicks, inter_feat, user_feat, info):
        '''
            Common function for observing and choosing an action

            Parameters :
             - rec_list : torch.LongTensor(rec_size)
                Previous recommendation list
             - serp_feat : torch.{Long or Float}Tensor(rec_size, serp_feat_dim)
                Previous SERP features
             - clicks : torch.LongTensor(rec_size)
                Observed clicks
             - inter_feat : torch.{Long or Float}Tensor(rec_size, inter_feat_dim)
                Observed interaction features
             - user_feat : torch.{Long or Float}Tensor(rec_size, user_feat_dim)
                New user features
             - info : None or dictionary
                Info returned by the simulator

            Output :
             - rec_list : torch.LongTensor(rec_size)
                New recommendation list
             - serp_feat : torch.{Long or Float}Tensor(rec_size, serp_feat_dim)
                New SERP features
            
        '''

        return torch.zeros(rec_size, dtype = torch.long), None

class RandomPolicy(LoggingPolicy):
    '''
        Returns a random list with no SERP features
    '''
    def __init__(self, data_dir : str, frequencies : str, **kwargs):
        super().__init__(**kwargs)

        if frequencies is not None:
            obs_freq_q = torch.load(data_dir + frequencies)
            self.docs_in_q = {qid : torch.arange(len(freqs))[torch.nonzero(freqs).squeeze()] for qid, freqs in obs_freq_q.items()}
        else:
            self.docs_in_q = num_items
        
    def forward(self, rec_list, serp_feat, clicks, inter_feat, user_feat, info):
        docs = self.docs_in_q[user_feat.item()]

        return docs[torch.randperm(len(docs))[:self.rec_size]], None

class OraclePolicy(LoggingPolicy):
    '''
        Epsilon-Optimal Policy
    '''
    def __init__(self, epsilon : float, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[LoggingPolicy.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--epsilon', type=float, default = 0.2)
        return parser
    
    def forward(self, rec_list, serp_feat, clicks, inter_feat, user_feat, info):
        if user_feat.dtype == torch.long:
            num_items = self.num_items[user_feat]
        else:
            num_items = self.num_items[0]

        eps_greedy = torch.bernoulli(self.epsilon * torch.ones(self.rec_size))
        random = torch.randint(num_items, size = (self.rec_size,))
        greedy = - torch.ones(self.rec_size, dtyp = torch.long)
        return torch.where(eps_greedy, random, greedy), None


class LoggedPolicy(LoggingPolicy):
    '''
        Policy read from file. Ranks by decreasing order of relevance according to the file.
    '''
    def __init__(self, data_dir : str, policy_name : str, frequencies : str, n_vert : int, 
                    reverse : bool, sample10 : bool, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.policy_name = policy_name
        self.reverse = reverse
        self.sample10 = sample10
        if len(policy_name.split("/")) > 1:  # generated policies    
            # Not compatible with Plackett-Luce, random_rerank, etc
            self.sorted_docs = torch.load(data_dir + policy_name + ".pt")
        else:
            self.relevances = torch.load(data_dir + policy_name + ".pt")    # Should be a dict{q_id : torch.LongTensor(num_items_for_q)}
                                                                            # For each query, we need the argsorted relevances
        if len(policy_name.split("/")) == 1:
            if sample10:
                self.docs_in_q = {}
                for qid, rels in self.relevances.items():
                    rescaled_rels = torch.log2(rels * 15 + 1).long()
                    rels1 = rescaled_rels[rescaled_rels >= 1]
                    rels1_idx = torch.arange(len(rescaled_rels))[rescaled_rels >= 1]
                    if len(rels1) < self.rec_size:
                        rels1 = torch.cat([rels1, rescaled_rels[rescaled_rels == 0]])[:self.rec_size]
                        rels1_idx = torch.cat([rels1_idx, torch.arange(len(rescaled_rels))[rescaled_rels == 0]])[:self.rec_size]
                    sample = []
                    count_rels = torch.zeros(5)
                    for idx, rel in zip(rels1_idx, rels1):
                        if len(sample) == self.rec_size:
                            break
                        if count_rels[rel] < 3:
                            sample.append(idx)
                            count_rels[rel] += 1
                    if len(sample) < self.rec_size:
                        poorly_relevant = torch.cat([torch.arange(len(rescaled_rels))[rescaled_rels == 1], torch.arange(len(rescaled_rels))[rescaled_rels == 0]])
                        unique = torch.tensor([pr for pr in poorly_relevant if pr not in sample], dtype = torch.long)
                        sample = torch.cat([torch.tensor(sample, dtype = torch.long), unique ])[:self.rec_size]
                    else:
                        sample = torch.tensor(sample, dtype = torch.long)
                    self.docs_in_q[qid] = sample
                    if len(sample) < self.rec_size:
                        print("shit")

            else:
                self.docs_in_q = {qid : torch.arange(len(rels)) for qid, rels in self.relevances.items()}

            if frequencies is not None:
                self.obs_freq_q = torch.load(data_dir + frequencies)
                self.docs_in_q = {qid : self.docs_in_q[qid][torch.nonzero(freqs[self.docs_in_q[qid]]).squeeze()] 
                                            for qid, freqs in self.obs_freq_q.items()}
                self.relevances = {qid : self.relevances[qid][docs] for qid, docs in self.docs_in_q.items()}
                self.sorted_docs = {q_id : self.docs_in_q[q_id][torch.argsort(rels, descending = True)] 
                                            for q_id, rels in self.relevances.items()}
            else:
                if n_vert > 0:
                    self.vert_in_q = {q_id : torch.randint(n_vert, size = (len(rels), 2)) for q_id, rels in self.relevances.items()}
                self.sorted_docs = {q_id : self.docs_in_q[q_id][torch.argsort(rels[self.docs_in_q[q_id]], descending = True)] 
                                            for q_id, rels in self.relevances.items()}
                self.relevances = {qid : self.relevances[qid][docs] for qid, docs in self.docs_in_q.items()}

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[LoggingPolicy.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--policy_name', type=str, default = "bm25")
        parser.add_argument('--n_vert', type=int, default = 0)
        parser.add_argument('--reverse', type=parser.str2bool, default = False)
        parser.add_argument('--sample10', type=parser.str2bool, default = False)
        return parser

    def forward(self, rec_list, serp_feat, clicks, inter_feat, user_feat, info):
        rl = self.sorted_docs[user_feat.item()][:self.rec_size]
        if self.reverse:
            rl = torch.flip(rl, [0])
            #rl = torch.cat([rl[self.rec_size // 2:], rl[:self.rec_size // 2]])
        # if self.sample10:
        #     rl = rl[torch.cat([torch.randperm(self.rec_size // 2), 5 + torch.randperm(self.rec_size // 2)])]
        return rl, None

class PlackettLuceLoggedPolicy(LoggedPolicy):
    '''
        Transform a logged policy into a stochastic policy using a Plackett-Luce distribution.
    '''
    def __init__(self, temperature : float, **kwargs):
        super().__init__(**kwargs)
        self.T = temperature
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[LoggedPolicy.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--temperature', type=float, default = 0.1)
        return parser
    
    def forward(self, rec_list, serp_feat, clicks, inter_feat, user_feat, info):
        log_relevances = torch.log(self.relevances[user_feat.item()] + 1e-2)
        docs = self.docs_in_q[user_feat.item()]

        ## We sample using the Gumbel-trick
        noise = torch.rand(len(log_relevances))
        gumbel_noise = - torch.log(- torch.log(noise))
        perturbed_softmax = torch.nn.Softmax(dim = 0)(log_relevances + gumbel_noise * self.T)
        ranked_list = docs[torch.topk(perturbed_softmax, k = self.rec_size).indices]
        
        if self.reverse:
            ranked_list = torch.flip(ranked_list, [0])
        return ranked_list, None

class PlackettLuceNoisyOracle(PlackettLuceLoggedPolicy):
    '''
        Gaussian perturbation of oracle policy with Plackett Luce stochasticity
    '''
    def __init__(self, noise_var : float, policy_name : str, **kwargs):
        super().__init__(policy_name = "relevances", **kwargs)
        self.noise_var = noise_var

        self.relevances = {qid : torch.clip(rels + torch.randn(len(rels)) * noise_var, 0, 1) for qid, rels in self.relevances.items()}
        torch.save(self.relevances, self.data_dir + "noisy_oracle.pt")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[PlackettLuceLoggedPolicy.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--noise_var', type=float, default = 0.1)
        return parser
    
            
