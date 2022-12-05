import torch 
import random
import os
from shutil import copy
from pathlib import Path

from utils.parse_mslr import parse_mslr
from utils.preprocess_mslr_datasets import preprocess_dataset
from modules.argument_parser import MyParser


parser = MyParser()
main_parser.add_argument('--path', type=str, required = True, help='Path to mslr/10K folder.')
main_parser.add_argument('--seed', type=int, default = 2021, help='Random seed.')
main_parser.add_argument('--gen_sample10', type=parser.str2bool, default = True, help='Whether we want to generate the datasets needed for experiment 6.1 of the paper.')
args = parser.parse_args()


path = args.path

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

gen_sample10 = args.gen_sample10    ### This boolean allows to recreate the experiment in Section 5.1 of the paper

print("### First : Parse the dataset")
parse_mslr(path)

if gen_sample10:
    print("### Then : generate datasets for experiment in Section 6.1")
    for icm in ["DBN", "CoCM"]:
        print(" -> For %s : " % icm)

        print("     -> Create training dataset")
        dataset_name = "sample10-" + icm
        dataset_path = path + dataset_name + "/"
        os.system("python generate_dataset.py --sim=sss --sampled True --policy=plackettluce --cm_type %s --data_dir %s --dataset_name %s --policy_name relevances --n_sess 100000 --sample10 True --temperature 0.5 --DBN_sigma 0.5" % 
                                                    (icm, path, dataset_name))
        preprocess_dataset(dataset_path)
        os.remove(dataset_path + "data.pt")

        print("     -> Create test dataset")
        test_dataset_name = "reverse"
        Path(dataset_path + test_dataset_name).mkdir(parents=True, exist_ok=True)
        ### Copy necessary files
        copy(dataset_path + "dimensions.pt", dataset_path + test_dataset_name + "/dimensions.pt")
        copy(dataset_path + "pair_embedding_dict.pt", dataset_path + test_dataset_name + "/pair_embedding_dict.pt")

        copy(dataset_path + "click_freq.pt", dataset_path + test_dataset_name + "/click_freq.pt")
        copy(dataset_path + "click_freq_last.pt", dataset_path + test_dataset_name + "/click_freq_last.pt")
        copy(dataset_path + "click_freq_rank.pt", dataset_path + test_dataset_name + "/click_freq_rank.pt")
        copy(dataset_path + "click_freq_before_last.pt", dataset_path + test_dataset_name + "/click_freq_before_last.pt")

        copy(dataset_path + "obs_freq.pt", dataset_path + test_dataset_name + "/obs_freq.pt")
        copy(dataset_path + "obs_freq_rank.pt", dataset_path + test_dataset_name + "/obs_freq_rank.pt")
        copy(dataset_path + "obs_freq_before_last.pt", dataset_path + test_dataset_name + "/obs_freq_before_last.pt")
        ### Generate test dataset
        os.system("python generate_dataset.py --sim=sss --policy=logged --cm_type %s --sampled False --distrib fromfreq --data_dir %s --dataset_name reverse --policy_name relevances --filename test --n_sess 10000 --frequencies obs_freq_q.pt --sample10 True --reverse True" % 
                                                (icm, dataset_path))

print("### Then : Create a dataset for each ICM, training policy and test_policy")
ICMs = ["DBN", "CoCM", "CoCM_mismatch"]
training_policies = ["plackettluce-lambdamart", "plackettluce-bm25", "plackettluceoracle"]
temperatures = [0.1, 0.03, 0.1]
test_policies = ["random", "logged-lambdamart", "logged-bm25", "logged-relevances"]
for icm in ICMs:
    print(" -> ICM : ", icm)
    for pol, temp in zip(training_policies, temperatures):
        print("     -> policy : ", pol)
        policy_class = pol.split("-")[0]
        if len(pol.split("-")) > 1:
            policy_name = "--policy_name " + pol.split("-")[1]
        else:
            policy_name = ""
        dataset_name = icm + "-" + pol
        dataset_path = path + dataset_name + "/"
        os.system("python generate_dataset.py --sim=sss --sampled True --policy=%s --cm_type %s --data_dir %s --dataset_name %s %s --n_sess 100000 --temperature %f" % 
                                                    (policy_class, icm, path, dataset_name, policy_name, temp))
        preprocess_dataset(dataset_path)
        os.remove(dataset_path + "data.pt")
        for test_pol in test_policies:
            print("             -> test_policy : ", test_pol)
            test_policy_class = test_pol.split("-")[0]
            if len(test_pol.split("-")) > 1:
                test_policy_name = "--policy_name " + test_pol.split("-")[1]
            else:
                test_policy_name = ""
            test_dataset_name = test_pol
            Path(dataset_path + test_dataset_name).mkdir(parents=True, exist_ok=True)
            ### Copy necessary files
            copy(dataset_path + "dimensions.pt", dataset_path + test_dataset_name + "/dimensions.pt")
            copy(dataset_path + "pair_embedding_dict.pt", dataset_path + test_dataset_name + "/pair_embedding_dict.pt")

            copy(dataset_path + "click_freq.pt", dataset_path + test_dataset_name + "/click_freq.pt")
            copy(dataset_path + "click_freq_last.pt", dataset_path + test_dataset_name + "/click_freq_last.pt")
            copy(dataset_path + "click_freq_rank.pt", dataset_path + test_dataset_name + "/click_freq_rank.pt")
            copy(dataset_path + "click_freq_before_last.pt", dataset_path + test_dataset_name + "/click_freq_before_last.pt")

            copy(dataset_path + "obs_freq.pt", dataset_path + test_dataset_name + "/obs_freq.pt")
            copy(dataset_path + "obs_freq_rank.pt", dataset_path + test_dataset_name + "/obs_freq_rank.pt")
            copy(dataset_path + "obs_freq_before_last.pt", dataset_path + test_dataset_name + "/obs_freq_before_last.pt")
            ### Generate test dataset
            os.system("python generate_dataset.py --sim=sss --policy=%s --cm_type %s --sampled False --distrib fromfreq --data_dir %s --dataset_name %s %s --filename test --n_sess 10000 --frequencies obs_freq_q.pt" % 
                                                    (test_policy_class, icm, dataset_path, test_dataset_name, test_policy_name))
