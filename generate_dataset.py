import torch
import random

import sys
from argparse import ArgumentParser

from modules.argument_parser import SimParser
from modules.simulators import SemiSyntheticSearchSim, RandomPolicy, OraclePolicy, LoggedPolicy, PlackettLuceLoggedPolicy, PlackettLuceNoisyOracle


def main_gen(arg_dict, sim_class, policy_class):
    sim = sim_class(**arg_dict)
    sim.set_policy(policy_class, arg_dict)
    sim.generate_dataset(arg_dict["n_sess"], arg_dict["data_dir"], chunksize = arg_dict["chunk_size"])


if __name__ == '__main__':
    main_parser = ArgumentParser()
    main_parser.add_argument('--sim', type=str, required = True, 
                                choices=['mms', 'MultiModalSearchSim', 'mmr', 'MultiModalRecSim', "sss", "SemiSyntheticSearchSim"], 
                                help='Name of the simulator.')
    main_parser.add_argument('--policy', type=str, required = True, 
                                choices=['random', 'oracle', 'logged', 'plackettluce', 'plackettluceoracle'], help='Name of the logging policy.')

    def get_elem(l, ch):
        for i,el in enumerate(l):
            if el.startswith(ch):
                return el
    sim_name = get_elem(sys.argv, "--sim")
    policy_name = get_elem(sys.argv, "--policy")
    main_args = main_parser.parse_args([sim_name, policy_name])
    sys.argv.remove(sim_name)
    sys.argv.remove(policy_name)

    if main_args.sim == "mms" or main_args.sim == "MultiModalSearchSim":
        sim_class = MultiModalSearchSim
    elif main_args.sim == "mmr" or main_args.sim == "MultiModalRecSim":
        sim_class = MultiModalRecSim
    elif main_args.sim == "sss" or main_args.sim == "SemiSyntheticSearchSim":
        sim_class = SemiSyntheticSearchSim
    else : 
        raise NotImplementedError("This simulator has not been implemented yet.")
    
    if main_args.policy == "random":
        policy_class = RandomPolicy
    elif main_args.policy == "oracle":
        policy_class = OraclePolicy
    elif main_args.policy == "logged":
        policy_class = LoggedPolicy
    elif main_args.policy == "plackettluce":
        policy_class = PlackettLuceLoggedPolicy
    elif main_args.policy == "plackettluceoracle":
        policy_class = PlackettLuceNoisyOracle
    else:
        raise NotImplementedError("This logging policy has not been implemented yet.")


    argparser = SimParser() # Program-wide parameters
    argparser = sim_class.add_model_specific_args(argparser)  # Simulator-specific parameters
    argparser = policy_class.add_model_specific_args(argparser)  # Policy-specific parameters
    args = argparser.parse_args(sys.argv[1:])

    seed = int(args.seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    arg_dict = {**vars(args), **vars(main_args)}

    main_gen(arg_dict, sim_class, policy_class)
    