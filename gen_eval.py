import torch
import os
from tqdm import tqdm
import sys
from argparse import ArgumentParser
from pathlib import Path

from train_click_models import main
from modules.argument_parser import MainParser
from modules.click_models import ClickModel, PBM, UBM, DBN, ARM, NCM, CACM, CACM_minus, RandomClickModel, TopPop

parser = ArgumentParser()
parser.add_argument("--cp_name", type = str, help = "Name of checkpoint file")
parser.add_argument("--dataset", type = str, help = "Name of training dataset.")
args = parser.parse_args()

cp_name = args.cp_name
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
    elif cm_name == "TopPop":
        cm_class = TopPop
    elif cm_name == "Random":
        cm_class = RandomClickModel
    else : 
        raise NotImplementedError("This click_model has not been implemented yet.")

    return model_params, cm_class


model_params, cm_class = get_params(cp_name)

argparser = MainParser() # Program-wide parameters
argparser = cm_class.add_model_specific_args(argparser, model_params["state_learner"])  # Model-specific parameters
args = argparser.parse_args([])

dimensions = torch.load(dataset + "dimensions.pt")
arg_dict = {**vars(args), **model_params, **dimensions}
arg_dict["test"] = True
arg_dict["ndcg_eval"] = False
arg_dict["device"] = "cpu"
arg_dict["exp_name"] = "gen_eval"
arg_dict["run_name"] = dataset.split("/")[-2]
if cm_class == TopPop:
    arg_dict["input_dir"] = dataset
elif cm_class != RandomClickModel:
    arg_dict["load_model"] = "../../checkpoints/" + cp_name

# Train on all test datasets 
test_datasets = next(os.walk(dataset))[1]
for test_dir in test_datasets:
    if test_dir in ["checkpoints", "policies", "results"]:
        continue
    print(" -> Testing on " + test_dir)
    arg_dict["data_dir"] = dataset + test_dir + "/"
    Path(arg_dict["data_dir"] + "checkpoints/").mkdir(parents=True, exist_ok=True)
    main(arg_dict, cm_class)
