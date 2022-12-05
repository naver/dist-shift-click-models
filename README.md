# Evaluating the Robustness of Click Models to Policy Distributional Shift

This repository contains the supporting code and implementation details for our paper [_Evaluating the Robustness of Click Models to Policy Distributional Shift_](https://dl.acm.org/doi/abs/10.1145/3569086).


We provide below architecture diagrams for the 6 click models used in our experiments :

**PBM** : 

<img src="./figs/PBM.png" width="400" >

**UBM** :

<img src="./figs/UBM.png" width="420" >

**DBN** : 

<img src="./figs/DBN.png" width="450" >

**NCM** : 

<img src="./figs/NCM.png" width="450" >

**CACM-** : 

<img src="./figs/NEHBM.png" width="600" >

**ARM** : 

<img src="./figs/ARM.png" width="600" >

## Repository Usage

### Getting started

The code was tested with python 3.7.4 on a CentOS 7 machine.
```
pip install -r requirements.txt
```

### To preprocess yandex data
1. Download Yandex's relevance prediction dataset and put it in the desired path.*
2. Execute the following scripts :
```
 python utils/parse_yandex.py --path [path]
 python utils/preprocessing.py --dataset [path/serp_based]
 python utils/filter_ground_truth.py --dataset [path/serp_based]
```
Now the ready-to-train Yandex data is ready !

 \* <sub>Unfortunately the Yandex dataset is no longer available online. If you wish to work with it, please contact us and we may be able to invite you to work with it at our premises.</sub> 

### To generate the simulated data :
1. Download the [MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/) dataset.
2. Put the ```Fold1``` folder in your desired path.
3. Add lambdmart policy's [two](./utils/lambdamart.pt) [files](./utils/lambdamart_sampled.pt) in the same directory 
4. ``` python generate_data.py --path [path/to/directory/]```


This will generate datasets for 3 different internal click models (DBN, CoCM, CoCM mismatch), each with 3 training policies (PL-oracle, PL-bm25, PL-lambdamart), each with 4 test policies (oracle, random, bm25, lambdamart), as well as the data required for the experiment in Section 6.1.

### To launch click model training on a specific dataset:
```
python train_click_models.py --cm=[CLICK MODEL] --sl=[STATE LEARNER] --data_dir [path/to/dataset/directory/] [--hp_name hp_value] 
```
This will train the desired click model on the dataset given as argument, save perplexity and NDCG results in ```data_dir/results/```, and save the best checkpoint in ```data_dir/checkpoints/```.

:warning: You must use the format ```--cm=XXX``` instead of ```--cm XXX```.

A complete list of default hyperparameters can be found in [```argument_parser.py```](./modules/argument_parser.py) for program-wide parameters and in [```click_models.py```](./modules/click_models.py) or [```state_learners.py```](./modules/state_learners.py) for model-specific parameters. We provide configuration files for reproducing the experiments in the paper in the [```config```](./config/) folder : [```specs_yandex.yml```](./config/specs_yandex.yml) for reproducing Table 2, [```specs_random_rerank.yml```](./config/specs_random_rerank.yml) for reproducing the in-distribution results of Tables 3 and 4, and [```specs_sim.yml```](./config/specs_sim.yml) for reproducing the red dashed line in Figure 1, after having generated the data.

### To launch in debugging mode (few iterations of test, and NDCG evaluation) :
Add  ``` --debug True``` 

### To run an experiment on robustness of click prediction (after training) :
```
python gen_eval.py --cp_name [checkpoint_filename] --dataset [path/to/dataset/directory/]
```
This will load the checkpoint ```dataset/checkpoints/cp_name``` and test it on all the target datasets present in the folder (by default oracle, random, bm25 and lambdamart). Ood-perplexity results are saved in ```dataset/target_dataset/results/```. We provide configuration files for reproducing the experiments in the paper in the [```config```](./config/) folder : [```specs_random_rerank_ood.yml```](./config/specs_random_rerank_ood.yml) for reproducing the out-of-distribution results of Tables 3 and 4, and [```specs_sim_ood.yml```](./config/specs_sim_ood.yml) for reproducing the blue line in Figure 1.

### To plot a similar spider chart as Figure 1 in the paper (after the robustness of click prediction experiment)
```
python spider_chart_gen_eval.py --path [path/to/datasets/directory/]
```
This will read results and plot them in an interpretable fashion. The figure is saved in ```path/gen_eval.png```.


### To run an experiment on robustness of subsequent policies (after training) :
```
python policy_eval.py --cp_name [checkpoint_filename] --dataset [path/to/dataset/directory/]
```
This will load the checkpoint ```dataset/checkpoints/cp_name```, extract the Top-Down and Max-Reward policies corresponding to this checkpoint, generate datasets with these policies and save the CTR of these datasets in ```dataset/policies/```. We provide configuration files for reproducing the experiments in the paper in the [```config```](./config/) folder : [```specs_pol.yml```](./config/specs_pol.yml) for reproducing the results displayed in Figure 2.

### To plot a similar bar chart as Figure 2 in the paper (after the robustness of policies experiment) :
```
python bar_chart_policy_eval.py --path [path/to/datasets/directory/]
```
This will read results and plot them in an interpretable fashion. The figure is saved in ```path/policy_eval.png```.
