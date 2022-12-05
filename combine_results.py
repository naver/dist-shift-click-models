import torch
import os
from pathlib import Path
from argparse import ArgumentParser
from pprint import pprint

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
sns.set_theme(style = "ticks", palette = "pastel", color_codes = True)
import pandas as pd
import scipy.stats as st
import numpy as np


parser = ArgumentParser()
parser.add_argument("data_path", type = str, help = "Path to dataset.")
parser.add_argument("--t_test", type = str, nargs = '+', help = "Models and metrics on which to perform a statistical test.", default = None)
parser.add_argument("--files", type = str, nargs = '+', help = "Files to combine.", default = ["all"])
args = parser.parse_args()

if args.data_path[:4] == "/scr":
    path = args.data_path
else:
    path = "/scratch/1/user/rdeffaye/" + args.data_path

if args.files == ['all']:
    files = os.listdir(path + "/results/")
else:
    files = args.files

params = ["itemembedd", "seed", "userstate", "onlyquery", "attention", "serpfeat", "normalized", "weighted", "smooth",
            "simplified", "absolute", "stacked", "relative", "ker", "allserpcontexts", "userfeatprop", "serpfeatprop", 
            "GRU", "LSTM", "node2vec"]

metrics = []

results = pd.DataFrame(columns = ["Model", "seed", "Metric", "Rank", "Value", "mode"] + params)
overall_results = pd.DataFrame(columns = ["Model", "Seed", "Metric", "Rank", "Value"])

for filename in files:
    # First, fill all models and parameters
    model_split = filename[:-3].split("_")
    overall_model_split = filename[:-3].split("seed")
    overall_model_params = {"Model" : overall_model_split[0][:-1], "Seed" : overall_model_split[1]}
    model_params = {}
    if model_split[0] == "TopPop":
        model_params["mode"] = model_split[1]
    for spec in model_split[1:]:
        val = None
        for i, p in enumerate(params):
            if spec.startswith(p):
                val = spec[len(p):]
                if val == "":
                    val = "True"
                else:
                    try :
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            break
                break
        if val is None:
            model_params["Model"] = model_split[0] + "_" + spec

        else:
            model_params[p] = val

    # Then, load results into df
    print(filename)
    res = torch.load(path + "/results/" + filename)
    for metric, val in res.items():
        metric_split = metric.split("@")
        if len(metric_split) == 1:
            rank = "mean"
        else:
            rank = metric_split[1]
        row = {**model_params, **{"Metric" : metric_split[0], "Rank" : rank, "Value" : val}}
        overall_row = {**overall_model_params, **{"Metric" : metric_split[0], "Rank" : rank, "Value" : val}}
        results = results.append(row, ignore_index = True)
        overall_results = overall_results.append(overall_row, ignore_index = True)


results.Value = results.Value.astype(float)
overall_results.Value = overall_results.Value.astype(float)

# ----------------------------- #
#            Saving             #
# ----------------------------- #

torch.save(results, path + "/aggregate_results.pt")

if args.t_test is None:
    # ----------------------------- #
    #           Printing            #
    # ----------------------------- #

    groups = overall_results.groupby(['Metric','Rank'], as_index=False).size()
    models = overall_results['Model'].unique()

    for i, model in enumerate(models):
        print('--------------------------------------')
        print(model + " : ")
        for j, row in groups.iterrows():
            metric = row['Metric']
            rank = row['Rank']
            data = overall_results[(overall_results['Model'] == model) & (overall_results['Metric'] == metric) & (overall_results['Rank'] == rank)]
            mean = data['Value'].mean()
            interval = st.t.interval(alpha=0.95, df=len(data)-1, loc=mean, scale=st.sem(data['Value'])) 
            if np.isnan(interval[0]):
                thresh = 0.0
            else:
                thresh = interval[1] - mean
            print("\t %s@%s : %.4f (+- %.4f)" % (metric, str(rank), mean, thresh))
        print("Computed on %d seeds." % len(data))

    # ----------------------------- #
    #           Plotting            #
    # ----------------------------- #

    ### Overall results

    # fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    # fig.suptitle('Results on ' + args.data_path)


    # sns.despine(fig)

    # # Perplex
    # perplex = overall_results[(overall_results["Metric"] == "Perplexity")]
    # lp0 = sns.lineplot(ax=axes[0], x = "Rank", y = "Value", hue = "Model", style = None, 
    #                 data = perplex[(perplex["Rank"] != "mean")])
    # lp0.set(ylim=(1, 1.6))
    # lp0.get_legend().remove()
    # axes[0].set_title("Click prediction")
    # axes[0].set_ylabel("Perplexity@Rank")


    # # NDCG
    # if len(overall_results[overall_results["Metric"] == "NDCG"]) > 0:
    #     lp1 = sns.lineplot(ax=axes[1], x = "Rank", y = "Value", hue = "Model", style = None, 
    #                     data = overall_results[overall_results["Metric"] == "NDCG"])
    #     lp1.get_legend().remove()
    #     axes[1].set_title("Relevance estimation")
    #     axes[1].set_ylabel("NDCG@Rank")

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.05), fontsize = 8, loc='upper center')
    # plot_path = "./results/"
    # Path(plot_path).mkdir(parents=True, exist_ok=True)
    # fig.savefig(plot_path + "results.png", bbox_inches='tight')



    # standout_param = None

    # if standout_param is not None:
    #     results_standout = results[not results.loc[standout_param].isna()]

    #     title = "dCTR - " + args.data_path

    #     sns_plot = sns.lineplot(x = standout_param, y = "Value", hue = "Model", style = None, data = df)
    #     sns_plot.set_title(title)
    #     sns_plot.set(xscale='log')

    #     fig = sns_plot.get_figure()
    #     plot_path = "./results/"
    #     Path(plot_path).mkdir(parents=True, exist_ok=True)
    #     fig.savefig(plot_path + "results_standout.png")

else:
    tests = args.t_test
    ## Format {Ref Model}|{Model1}&{Model2}&{...}|{Metric1}&{Metric2}&{...}|{Rank1}&{Rank2}&{...}

    for t in tests:
        t_split = t.split("|")
        ref_model = t_split[0]
        target_models = t_split[1].split("&")
        metrics = t_split[2].split("&")
        ranks = t_split[3].split("&")

        for model in target_models:
            for metric in metrics:
                for rank in ranks:
                    data_ref = overall_results[(overall_results['Model'] == ref_model) & (overall_results['Metric'] == metric) & (overall_results['Rank'] == rank)]
                    data = overall_results[(overall_results['Model'] == model) & (overall_results['Metric'] == metric) & (overall_results['Rank'] == rank)]

                    t_test_res = st.ttest_ind(data_ref["Value"], data["Value"], equal_var=False)
                    print("'-------------------------")
                    print("%s \n \t vs \n%s \n \ton %s@%s :" % (ref_model, model, metric, rank))
                    print(t_test_res)
                    if t_test_res.pvalue < 0.05:
                        print("Statistically significant")
                    else:
                        print("Not statistically significant")
        



