import torch
import sys
import random
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger

from modules.click_models import ClickModel, PBM, UBM, DBN, ARM, NCM, CACM, CACM_minus, RandomClickModel, TopPop, CoCM
from modules.item_embeddings import ItemEmbeddings
from modules.argument_parser import MainParser
from modules.data_handler import ClickModelDataModule, get_file_name

#@profile
def main(arg_dict, cm_class):
    print("Dataset : ", arg_dict["data_dir"])
    print('\n')
    print("    **********************************\n       Initializing item embeddings\n    **********************************")

    ########################
    # Document Embeddings  #
    ########################

    print("No pre-training : initializing item embeddings randomly ...")
    item_embedd = ItemEmbeddings.from_scratch(arg_dict["num_items"], arg_dict["embedd_dim"])


    ##################
    #    Training    #
    ##################

    print("\n    **********************************\n        Training the click model\n    **********************************")
    print("Building model ...")
    if arg_dict["load_model"] is not None:
        if arg_dict["state_learner"] == "immediate":
            sl_cn = "ImmediateStateLearner"
        arg_dict["load_model"] = get_file_name(cn = cm_class.__name__, sl_cn = sl_cn, **arg_dict) + ".ckpt"
        print("Checkpoint found, loading model ...")
        cp_path = arg_dict["data_dir"] + "checkpoints/"
        cm = cm_class.load_from_checkpoint(cp_path + arg_dict["load_model"], map_location = arg_dict["device"], 
                                                item_embedd = item_embedd, **arg_dict)
    else:
        cm = cm_class(item_embedd, **arg_dict)

    print("Creating datasets ...")
    datamod = ClickModelDataModule(None, None, None, **arg_dict)

    ### Loggers and Callbacks
    if arg_dict["logger"]:
        aim_logger = AimLogger(experiment=arg_dict["exp_name"] + "_" + arg_dict["run_name"])
        aim_logger.log_hyperparams(arg_dict)
    cp_path = arg_dict["data_dir"] + "checkpoints/"
    Path(cp_path).mkdir(parents=True, exist_ok=True)
    es = EarlyStopping(monitor='my_val_loss', patience = 3, check_on_train_epoch_end = False)
    filename = get_file_name(cn = cm_class.__name__, sl_cn = cm.state_learner.__class__.__name__, **arg_dict)
    mc = ModelCheckpoint(dirpath = cp_path, filename = filename, save_on_train_epoch_end = False,
                            monitor = 'my_val_loss', mode = 'min', save_weights_only = True)

    print("Launch training ...")
    trainer = pl.Trainer(gpus = int((arg_dict["device"] != "cpu")), callbacks = [es, mc], logger = aim_logger 
                            if (not arg_dict["debug"]) and (not arg_dict["test"]) and arg_dict["logger"] else True, 
                            log_every_n_steps=10, max_epochs= arg_dict["max_epochs"], val_check_interval = arg_dict["val_check_interval"], 
                            progress_bar_refresh_rate=int(arg_dict["progress_bar"]))
    if arg_dict["debug"] == False and arg_dict["test"] == False:
        trainer.fit(cm, datamod)


    ####################
    #    Evaluation    #
    ####################

    print("\n    *********************************\n        Testing the click model\n    *********************************")

    pass_model = arg_dict["load_model"] is not None or arg_dict["debug"] or arg_dict["test"]

    res = trainer.test(model = cm if pass_model else None, 
                        ckpt_path = cp_path + filename + ".ckpt" if not pass_model else None, datamodule = datamod)

if __name__ == '__main__':
    ###############
    # Definitions #
    ###############

    main_parser = ArgumentParser()
    main_parser.add_argument('--cm', type=str, required = True, 
                                choices=['Random', 'TopPop', 'CoCM', 'ClickModel', 'PBM', 'UBM',
                                'DBN', 'ARM', 'NCM', 'CACM', 'CACM_minus'], help='Name of the click model.')
    main_parser.add_argument('--sl', type=str, required = True, 
                                choices=['none', 'immediate', 'pair_embedding', 'GRU', 'CACM'], 
                                help='Name of the state learner.')  
    def get_elem(l, ch):
        for i,el in enumerate(l):
            if el.startswith(ch):
                return el
    cm_name = get_elem(sys.argv, "--cm")
    sl_name = get_elem(sys.argv, "--sl")
    main_args = main_parser.parse_args([cm_name, sl_name])
    sys.argv.remove(cm_name)
    sys.argv.remove(sl_name)

    # Models to use : 

    print("\n\n")
    print(" -> Training %s Click Model with %s state learner" % (main_args.cm, main_args.sl))

    item_embedd_class = ItemEmbeddings


    if main_args.cm == "ClickModel":
        cm_class = ClickModel
    elif main_args.cm == "PBM":
        cm_class = PBM
    elif main_args.cm == "UBM":
        cm_class = UBM
    elif main_args.cm == "DBN":
        cm_class = DBN
    elif main_args.cm == "ARM":
        cm_class = ARM
    elif main_args.cm == "NCM":
        cm_class = NCM
    elif main_args.cm == "CACM":
        cm_class = CACM
    elif main_args.cm == "CACM_minus":
        cm_class = CACM_minus
    elif main_args.cm == "TopPop":
        cm_class = TopPop
    elif main_args.cm == "Random":
        cm_class = RandomClickModel
    elif main_args.cm == "CoCM":
        cm_class = CoCM
    else : 
        raise NotImplementedError("This click_model has not been implemented yet.")

    # Arguments definition
    argparser = MainParser() # Program-wide parameters
    argparser = cm_class.add_model_specific_args(argparser, main_args.sl)  # Model-specific parameters

    args = argparser.parse_args(sys.argv[1:])

    # Seeds for reproducibility
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    arg_dict = vars(args)
    dimensions = torch.load(args.data_dir + "dimensions.pt")
    arg_dict = {**arg_dict, **dimensions}
    arg_dict["state_learner"] = main_args.sl
    arg_dict["click_model"] = main_args.cm


    main(arg_dict, cm_class)