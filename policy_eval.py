import torch
import random
import os
from pathlib import Path
from argparse import ArgumentParser
from pprint import pprint
from tqdm import tqdm

from generate_dataset import main_gen
from modules.click_models import ClickModel, PBM, UBM, DBN, ARM, NCM, CACM, CACM_minus, RandomClickModel, TopPop
from modules.argument_parser import MainParser, SimParser
from modules.item_embeddings import ItemEmbeddings
from modules.data_handler import get_file_name
from modules.simulators import SemiSyntheticSearchSim, LoggedPolicy



### Save Top-down policy and PRP policy

## Parser 
parser = ArgumentParser()
parser.add_argument("--dataset", type = str, help = "Dataset name.")
parser.add_argument("--cp_name", type = str, help = "Checkpoint name.")
args = parser.parse_args()

## Parameters
dataset = args.dataset

def get_params(filename):
    params = {"itemembedd" : "embedd_dim", "seed" : "seed", "userstate" : "state_dim", "onlyquery" : "only_query", 
                "attention" : "attention", "serpfeat" : "serp_feat", "normalized" : "normalized", "weighted" : "weighted", 
                "smooth" : "smooth", "simplified" : "simplified", "absolute" : "absolute", "stacked" : "stacked", 
                "relative" : "relative", "ker" : "kernel_size", "allserpcontexts" : "all_serp_contexts", 
                "userfeatprop" : "user_feat_prop", "serpfeatprop" : "serp_feat_prop", "GRU" : "inner_state_dim", 
                "LSTM" : "inner_state_dim", "node2vec" : "node2vec", "CTR" : "mode", "click" : "mode", "obs" : "mode", 
                "product" : "mode"}
    filename_split = filename[:-3].split("_")
    cm_name = filename_split[0]
    model_params = {}
    for spec in filename_split[1:]:
        val = None
        for i, p in enumerate(params.keys()):
            if spec.startswith(p):
                val = spec[len(p):]
                if val == "":
                    if p in ["CTR", "click", "obs", "product"]:
                        val = p
                    else:
                        val = True
                else:
                    if p in ["GRU", "LSTM"]:
                        model_params["recurrent_type"] = p
                    try :
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            break
                break
        if val is None:
            model_params["state_learner"] = spec
        else:
            model_params[params[p]] = val

    if "state_learner" not in model_params:
        model_params["state_learner"] = "none"

    if cm_name == "ClickModel":
        cm_class = ClickModel
    elif cm_name == "PBM":
        cm_class = PBM
    elif cm_name == "UBM":
        cm_class = UBM
    elif cm_name == "DBN":
        cm_class = DBN
    elif cm_name == "ARM":
        cm_class = ARM
    elif cm_name == "NCM":
        cm_class = NCM
    elif cm_name == "CACM":
        cm_class = CACM
    elif cm_name == "CACM_minus":
        cm_class = CACM_minus
    elif cm_name == "AICM":
        cm_class = AICM
    elif cm_name == "TopPop":
        cm_class = TopPop
    elif cm_name == "Random":
        cm_class = RandomClickModel
    elif cm_name == "Oracle":
        cm_class = Oracle
    else : 
        raise NotImplementedError("This click_model has not been implemented yet.")

    return model_params, cm_class

cp_name = args.cp_name
filename = cp_name.split(".")[0]  # Remove extension
model_params, cm_class = get_params(filename)

seed = model_params["seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

argparser = MainParser() # Program-wide parameters
argparser = cm_class.add_model_specific_args(argparser, model_params["state_learner"])  # Model-specific parameters
args = argparser.parse_args([])

dimensions = torch.load(dataset + "dimensions.pt")
arg_dict = {**vars(args), **model_params, **dimensions}
arg_dict["data_dir"] = dataset
arg_dict["device"] = "cpu"

item_embedd = ItemEmbeddings.from_scratch(arg_dict["num_items"], arg_dict["embedd_dim"])
cp_path = dataset + "checkpoints/"
cm = cm_class.load_from_checkpoint(cp_path + cp_name, map_location = arg_dict["device"], 
                                        item_embedd = item_embedd, **arg_dict)
cm.eval()


Path(dataset + "policies/CTR/").mkdir(parents=True, exist_ok=True)

td_policy, docs_per_query, relevances, expected_CTR = cm.get_policy(top_down = True)
# freqs = torch.load(dataset + "obs_freq_q.pt")
# freqs = { q : freq[docs_per_query[q]] for q, freq in freqs.items()}
# sum_freqs = torch.sum(torch.cat([torch.sum(freq).unsqueeze(0) for freq in freqs.values()]))
torch.save(td_policy, dataset + "policies/" + filename + "_TOPDOWN.pt")
torch.save(expected_CTR, dataset + "policies/CTR/" + filename + "_TOPDOWN.pt")


mr_policy, docs_per_query, relevances, expected_CTR = cm.get_policy(top_down = False)
torch.save(mr_policy, dataset + "policies/" + filename + "_MAXREWARD.pt")
torch.save(expected_CTR, dataset + "policies/CTR/" + filename + "_MAXREWARD.pt")


print("#### Generate dataset from policies")
cm_type = dataset.split("/")[-2].split("-")[0]
seed = filename.split("_")[-1][4:]

os.system("python generate_dataset.py --sim=sss --policy=logged --cm_type %s --sampled False --distrib fromfreq --data_dir %s --dataset_name policies/datasets/ --policy_name %s --filename %s --n_sess 100000 --frequencies obs_freq_q.pt --device cpu --seed %s" % 
                                                    (cm_type, dataset, "policies/" + filename + "_TOPDOWN", filename + "_TOPDOWN", seed))

os.system("python generate_dataset.py --sim=sss --policy=logged --cm_type %s --sampled False --distrib fromfreq --data_dir %s --dataset_name policies/datasets/ --policy_name %s --filename %s --n_sess 100000 --frequencies obs_freq_q.pt --device cpu --seed %s" % 
                                                    (cm_type, dataset, "policies/" + filename + "_MAXREWARD", filename + "_MAXREWARD", seed))


print("#### Measuring CTRs on generated datasets")
Path(dataset + "policies/datasets/CTR/").mkdir(parents=True, exist_ok=True)
data = torch.load(dataset + "policies/datasets/" + filename + "_TOPDOWN.pt")
CTR_topdown = 0
n = len(data)
for traj in tqdm(data.values(), total = n):
    CTR_topdown += torch.sum(traj["clicks"]) / (10 * n)# * 100  # this gives the CTR in % in the dataset

data = torch.load(dataset + "policies/datasets/" + filename + "_MAXREWARD.pt")
CTR_maxreward = 0
n = len(data)
for traj in tqdm(data.values(), total = n):
    CTR_maxreward += torch.sum(traj["clicks"]) / (10 * n)# * 100  # this gives the CTR in % in the dataset


torch.save({"TOPDOWN" : CTR_topdown, "MAXREWARD" : CTR_maxreward}, dataset + "policies/datasets/CTR/" + filename + ".pt")