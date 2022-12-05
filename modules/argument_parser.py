from argparse import ArgumentParser

class MyParser(ArgumentParser):
    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')



class MainParser(MyParser):
    def __init__(self):
        ArgumentParser.__init__(self)
        
        #   ---- General parameters ----   #
        self.add_argument(
            "--exp_name", type=str, default="test_exp",
            help="Experiment name."
        )
        self.add_argument(
            "--run_name", type=str, default="test_run",
            help="Run name."
        )
        self.add_argument(
            "--debug", type=self.str2bool, default=False,
            help="Toggle Debugging, i.e. only a few iterations."
        )
        self.add_argument(
            "--test", type=self.str2bool, default=False,
            help="Toggle Testing only, i.e. no training."
        )
        self.add_argument(
            "--seed", type=int, default=2021,
            help="Seed for reproducibility."
        )
        self.add_argument(
            "--num_workers", type=int, default=0,
            help="Number of workers for data serving."
        )
        self.add_argument(
            "--data_dir", type=str, default="mslr/10K/",
            help="Path to data folder."
        )
        self.add_argument(
            "--progress_bar", type=self.str2bool, default=True,
            help="Toggle progress bar."
        )
        self.add_argument(
            "--ndcg_eval", type=self.str2bool, default=True,
            help="Toggles NDCG Evaluation on ground truth data."
        )
        self.add_argument(
            "--ndcg2_eval", type=self.str2bool, default=False,
            help="Toggles NDCG* Evaluation on ground truth data."
        )
        self.add_argument(
            "--logger", type=self.str2bool, default=False,
            help="Toggles Logging."
        )


        #   ---- Training ----   #
        self.add_argument(
            "--load_model", type=str, default=None,
            help="Filename of weights to load."
        )
        self.add_argument(
            "--device", type=str, default="cuda",
            help="Device to train on."
        )
        self.add_argument(
            "--max_epochs", type=int, default=20,
            help="Maximum number of epochs."
        )
        self.add_argument(
            "--val_check_interval", type=int, default=2000,
            help="Number of gradient steps between each validation epoch."
        )
        self.add_argument(
            "--batch_size", type=int, default=32,
            help="Size of a batch for training."
        )
        self.add_argument(
            "--gradient_clip_val", type=float, default=40,
            help="Value for clipping gradient norm."
        )

        #   ---- Item embeddings ----   #
        self.add_argument(
            "--pre_training", type=self.str2bool, default=False,
            help="Whether to use pre-trained embeddings."
        )
        self.add_argument(
            "--embedd_path", type=str, default="embeddings.pt",
            help="Path (relative to the data folder) to embedding dictionary to be loaded. " +
                        "Trains new embedding if they don't already exist."
        )
        self.add_argument(
            "--embedd_dim", type=int, default=64,
            help="Dimension of the item embeddings."
        )
        self.add_argument(
            "--num_epochs_embedd", type=int, default=100,
            help="Number of epochs for embedding training."
        )
        self.add_argument(
            "--lr_embedd", type=float, default=0.001,
            help="Learning rate for embeddings training."
        )
        self.add_argument(
            "--batch_size_embedd", type=int, default=256,
            help="Batch size for embedding training."
        )
        self.add_argument(
            "--train_val_split_embedd", type=float, default=.9,
            help="Training/validation split ratio for embeddings training." + 
                    " 0.9 means 90 training and 10 validation."
        )
        self.add_argument(
            "--weight_decay_embedd", type=float, default=0.0,
            help="Weight decay for L2 regularization during embeddings training."
        )
        self.add_argument(
            "--num_neg_sample", type=int, default=1,
            help="Number of negative sample items in matrix factorization."
        )

        #   ---- State learner ----   #
        self.add_argument(
            "--state_dim", type=int, default=32,
            help="Dimension of the user state."
        )

        #   ---- Evaluation ----   #
        self.add_argument(
            "--gen_ppl", type=self.str2bool, default=False,
            help="Toggle weak labels generation from the test set in order " +
                    "to compute Reverse/Forward perplexity."
        )
        self.add_argument(
            "--n_gen", type=int, default=7,
            help="How many times we sample clicks from each ranking in the test set."
        )
        self.add_argument(
            "--ppl_metric", type=str, default=None, choices=["reverse", "forward", None],
            help="Toggles reverse/forward perplexity."
        )
        self.add_argument(
            "--gen_data_paths", type=str, nargs='+', default=None,
            help="Paths to generated dataset that need to be evaluated with " +
                    "Reverse/Forward perplexity."
        )



class SimParser(MyParser):
    def __init__(self):
        ArgumentParser.__init__(self)
        
        #   ---- General parameters ----   #
        self.add_argument(
            "--seed", type=int, default=2021,
            help="Seed for reproducibility."
        )
        self.add_argument(
            "--data_dir", type=str, default="mslr/10K/",
            help="Path to data folder."
        )
        self.add_argument(
            "--device", type=str, default="cpu",
            help="Device."
        )

        #   ---- Dataset generation ----   #
        
        self.add_argument(
            "--n_sess", type=int, default=10000,
            help="Number of sessions in the dataset."
        )
        self.add_argument(
            "--chunk_size", type=int, default=0,
            help="Size of each data chunk (infinite if chunk_size = 0)."
        )
        self.add_argument(
            "--frequencies", type = str, default = None,
            help="Frequencies to use for conditional dataset generation."
        )
        self.add_argument(
            "--dataset_name", type = str, default = None,
            help="If we want the dataset to be saved with a special name."
        )
    