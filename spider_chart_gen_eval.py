
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Ellipse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import math
import os
import torch
from typing import Dict
from argparse import ArgumentParser

plt.rcParams['text.usetex'] = True


parser = ArgumentParser()
parser.add_argument("--path", type = str, required = True, help = "Path to datasets")
parser.add_argument("--get_intervals", type = bool, default = False, help = "Whether to plot confidence bounds")
parser.add_argument("--extension", type=str, default = "png", choices = ["png", "svg"], help = "File extension.")
args = parser.parse_args()

path = args.path
configurations = {"TCM" : ["PBM", "UBM", "DBN", "NCM", "ARM", "CACM_minus"], 
                    "ICM" : {"mismatch" : "CoCM_mismatch", "CoCM" : "CoCM", "DBN" : "DBN"},
                    "TRP" : {"PL-oracle" : "oracle", "PL-$\lambda$MART" : "-lambdamart", "PL-bm25" : "-bm25"},
                    "TEP" : {"oracle" : "logged-relevances", "$\lambda$MART" : "logged-lambdamart", "bm25" : "logged-bm25" , "random" : "random"}}

def make_spider(values, intervals, configs):

    # Define number of classes
    TCMs = configs["TCM"]
    n_ax = len(TCMs)
    ICMs = list(configs["ICM"].keys())
    n_icm = len(ICMs)
    TRPs = list(configs["TRP"].keys())
    n_trp = len(TRPs)
    TEPs = list(configs["TEP"].keys())
    #########n_tep = len(TEPs)
    n_tep = len(TEPs) - 1
    n_angles = n_icm * n_trp * n_tep     
    
    angles = [n / float(n_angles) * 2 * np.pi for n in range(n_angles)]
    angles += angles[:1]

    # General figure parameters
    fig = plt.figure(figsize = (10, 16)) #(16,10) for 2x3 and (10, 16) for 3x2
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.2, hspace=None)
    plt.grid(linestyle = "dotted")
    plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    
    # Normalize the values
    norm_ood = np.zeros((n_angles + 1, n_ax))
    norm_ind = np.zeros((n_angles + 1, n_ax))
    if intervals is not None:
        norm_ood_inter = np.zeros((n_angles + 1, n_ax, 2))
        norm_ind_inter = np.zeros((n_angles + 1, n_ax, 2))

    for key, val in values.items() : 
        ks = key.split("_")
        i = TCMs.index(ks[0])
        
        insert_idx = n_trp * n_tep * ICMs.index(ks[1]) + n_tep * TRPs.index(ks[2])
        

        norm_ind[insert_idx:insert_idx + n_tep, i] = val[0]
        ###############norm_ood[insert_idx:insert_idx + n_tep, i] = val[1:]
        index_ind = TRPs.index(ks[2]) + 1
        norm_ood[insert_idx:insert_idx + n_tep, i] = np.concatenate((val[1:index_ind], val[index_ind + 1:]))
        
        if intervals is not None:
            norm_ind_inter[insert_idx:insert_idx + n_tep, i] = intervals[key][0]
            #################norm_ood_inter[insert_idx:insert_idx + n_tep, i] = intervals[key][1:]
            norm_ood_inter[insert_idx:insert_idx + n_tep, i] = np.concatenate((intervals[key][1:index_ind], intervals[key][index_ind + 1:]), axis = 0)
            

    norm_ood[-1, :] = norm_ood[0, :]
    norm_ind[-1, :] = norm_ind[0, :]
    if intervals is not None:
        norm_ood_inter[-1, :] = norm_ood_inter[0, :]
        norm_ind_inter[-1, :] = norm_ind_inter[0, :]
    
    min_ppl_ood = np.zeros((n_angles + 1, 1))
    max_ppl_ood = np.zeros((n_angles + 1, 1))
    min_ppl_ind = np.zeros((n_angles + 1, 1))
    max_ppl_ind = np.zeros((n_angles + 1, 1))
    for j in range(n_angles + 1):
        if np.any(norm_ood[j]):
            min_ppl_ood[j] = np.min(norm_ood[j][norm_ood[j] > 1])
            max_ppl_ood[j] = np.max(norm_ood[j][norm_ood[j] > 1])
        else:
            max_ppl_ood[j] = 1

        if np.any(norm_ind[j]):
            min_ppl_ind[j] = np.min(norm_ind[j][norm_ind[j] > 1])
            max_ppl_ind[j] = np.max(norm_ind[j][norm_ind[j] > 1])
        else:
            max_ppl_ind[j] = 1

    norm_ood = 0.2 + np.log(1 + np.clip((norm_ood - min_ppl_ood) / (max_ppl_ood - min_ppl_ood), a_min = 0, a_max = None))
    norm_ind = 0.2 + np.log(1 + np.clip((norm_ind - min_ppl_ind) / (max_ppl_ind - min_ppl_ind), a_min = 0, a_max = None))
    if intervals is not None:
        min_ppl_ood_inter = np.expand_dims(min_ppl_ood, axis = 2)
        min_ppl_ind_inter = np.expand_dims(min_ppl_ind, axis = 2)
        max_ppl_ood_inter = np.expand_dims(max_ppl_ood, axis = 2)
        max_ppl_ind_inter = np.expand_dims(max_ppl_ind, axis = 2)
        norm_ood_inter = 0.2 + np.log(1 + np.clip((norm_ood_inter - min_ppl_ood_inter) / (max_ppl_ood_inter - min_ppl_ood_inter), a_min = 0, a_max = None))
        norm_ind_inter = 0.2 + np.log(1 + np.clip((norm_ind_inter - min_ppl_ind_inter) / (max_ppl_ind_inter - min_ppl_ind_inter), a_min = 0, a_max = None))
    
    # For each click model
    for i in range(n_ax):
        # Axis general parameters
        ax = plt.subplot(3, math.ceil(n_ax/3),1 + i, polar=True)
        ax.grid(linewidth = 1)        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Plot separation lines ### Right now this only works with 3 ICMs
        plt.plot([0, 0], [0, 1], color = 'black', linewidth = 3)
        plt.fill_betweenx([0, 2.24], [0, 0], [2 * np.pi / n_icm, 2 * np.pi / n_icm], color = 'gray', alpha = 0.2)
        plt.plot([2 * np.pi / n_icm, 2 * np.pi / n_icm], [0, 1], color = 'black', linewidth = 3)
        plt.fill_betweenx([0, 2.24], [2 * np.pi / n_icm, 2 * np.pi / n_icm], [4 * np.pi / n_icm, 4 * np.pi / n_icm], color = 'yellow', alpha = 0.2)
        plt.plot([4 * np.pi / n_icm, 4 * np.pi / n_icm], [0, 1], color = 'black', linewidth = 3)
        plt.fill_betweenx([0, 2.24], [4 * np.pi / n_icm, 4 * np.pi / n_icm], [0, 0], color = 'green', alpha = 0.2)

        for k in range(n_icm * n_trp):
            plt.plot([k * 2 *np.pi / (n_icm * n_trp), k * 2 *np.pi / (n_icm * n_trp)], [0, 1], color = 'black', linewidth = 1)

        padding = 5
        n_points = int(100 * 2 * np.pi / (n_trp * n_icm))
        for l in range(n_trp * n_icm):
            plt.plot([0.01 * k + 2 * np.pi / (n_trp * n_icm) * l for k in range (padding, n_points - padding)], [1.35] * (n_points - 2 * padding), 
                            color = "black", linewidth = 2, clip_on = False)


        ###########plt.xticks(np.array(angles[:-1]), TEPs * n_icm * n_trp, color='black', size = 5)
        filtered_teps = ["$\lambda$MART", "bm25", "random", "oracle", "bm25", "random","oracle", "$\lambda$MART", "random"] * 3
        plt.xticks(np.array(angles[:-1]), filtered_teps, color='black', size = 5)

        labels = []
        for label, angle in zip(ax.get_xticklabels(), np.rad2deg(angles[:-1])):
            x,y = label.get_position()
            lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
                        ha=label.get_ha(), va=label.get_va(), fontsize = 7, position = (label._x + np.pi / float(n_angles), label._y))
            if angle < 180:
                lab.set_rotation(90 - angle - np.rad2deg(np.pi / float(n_angles)))
            else:
                lab.set_rotation(270 - angle - np.rad2deg(np.pi / float(n_angles)))
            labels.append(lab)
        ax.set_xticklabels([])
        
        
        ax.set_rlabel_position(0)
        plt.yticks([0, 0.2,0.4,0.6, 0.8,1.0], ["0","0.2","0.4","0.6","0.8", "1.0"], color="black", size = 1)
        plt.ylim(0,1)

        sup_labels = []
        for m, (trp, angle) in enumerate(zip(TRPs * n_icm, np.rad2deg(angles[0:-1:n_tep]))) :
            lab = ax.text(x = (2 * m + 1) * np.pi / (n_icm * n_trp), y = 1.45, s = trp, ha = 'center', va = 'center')
            new_angle = angle + np.rad2deg(3 * np.pi / float(n_angles))
            if new_angle < 90 or new_angle > 270:
                lab.set_rotation(- new_angle)
            else:
                lab.set_rotation(180 - new_angle)
            sup_labels.append(lab)
    
        ax.plot(np.array(angles) + np.pi / float(n_angles), norm_ood[:, i], linewidth=2, linestyle='solid', color = "blue", label = "ood Perplexity")
        ax.plot(np.array(angles) + np.pi / float(n_angles), norm_ind[:, i], linewidth=2, linestyle='dashed', color = "red", label = "ind Perplexity")
        if intervals is not None:
            ax.fill_between(np.array(angles) + np.pi / float(n_angles), norm_ood_inter[:, i, 0], norm_ood_inter[:, i, 1], color = "blue", alpha = 0.2)
            ax.fill_between(np.array(angles) + np.pi / float(n_angles), norm_ind_inter[:, i, 0], norm_ind_inter[:, i, 1], color = "red", alpha = 0.2)

        if i == n_ax - 1:
            handles, labels = ax.get_legend_handles_labels()

        if TCMs[i] == "CACM_minus":
            title = "CACM\\textsuperscript{\\textcircled{-}}"
        else:
            title = TCMs[i]
        plt.title(title, fontsize = 20, y = 1.25)
    
    # Bottom legend
    fig.legend(handles, labels, loc='lower center', ncol = 2, frameon = False)
    ###### For 2 rows :
    # fig.text(0.42, 0.05, "DBN       CoCM       CoCM mismatch", fontsize = 14)
    # fig.patches.extend([plt.Circle((0.405,0.058),0.01,
    #                               fill=True, color='green', alpha=0.2,
    #                               transform=fig.transFigure, figure=fig),
    #                     plt.Circle((0.46,0.058),0.01,
    #                               fill=True, color='yellow', alpha=0.2,
    #                               transform=fig.transFigure, figure=fig),
    #                     plt.Circle((0.523,0.058),0.01,
    #                               fill=True, color='gray', alpha=0.2,
    #                               transform=fig.transFigure, figure=fig)
    #                               ])

    ###### For 3 rows :
    fig.text(0.35, 0.05, " $\,$DBN \quad \quad CoCM \quad \quad $\,$ CoCM mismatch", fontsize = 14)
    fig.patches.extend([Ellipse((0.335,0.0535), 0.02, 0.01,
                                  fill=True, color='green', alpha=0.2,
                                  transform=fig.transFigure, figure=fig),
                        Ellipse((0.418,0.0535),0.02, 0.01,
                                  fill=True, color='yellow', alpha=0.2,
                                  transform=fig.transFigure, figure=fig),
                        Ellipse((0.52,0.0535),0.02, 0.01,
                                  fill=True, color='gray', alpha=0.2,
                                  transform=fig.transFigure, figure=fig)
                                  ])

def load_results(path : str, configurations : Dict, get_intervals):
    values = {}
    if get_intervals:
        intervals = {}
    else:
        intervals = None
    for icm_key, icm in configurations["ICM"].items():
        for trp in configurations["TRP"].items():
            for tcm in configurations["TCM"]:
                key = tcm + "_" + icm_key + "_" + trp[0]
                if key not in values:
                    values[key] = np.zeros(len(configurations["TEP"]) + 1)
                    if get_intervals:
                        intervals[key] = np.zeros((len(configurations["TEP"]) + 1, 2))

                rel_path = icm + "-plackettluce" +  trp[1] + "/results/"
                all_files = os.listdir(path + rel_path)
                results = []
                for filename in all_files:
                    if filename[-3:] != ".pt":
                        continue
                    fs = filename.split("_")
                    if fs[1] == "PairEmbeddingStateLearner":    ### Only Immediate state learner for now
                        continue
                    if fs[0] == tcm:
                        results.append(torch.load(path + rel_path + filename)["Perplexity"])
                values[key][0] = np.mean(results)
                if get_intervals:
                    intervals[key][0] = st.t.interval(alpha=0.95, df=len(results)-1, loc=np.mean(results), scale=st.sem(results))
                            
                for i, tep in enumerate(configurations["TEP"].items()):
                    rel_path = icm + "-plackettluce" +  trp[1] + "/" + tep[1] + "/results/"
                    all_files = os.listdir(path + rel_path)
                    results = []
                    for filename in all_files:
                        if filename[-3:] != ".pt":
                            continue
                        fs = filename.split("_")
                        if fs[1] == "PairEmbeddingStateLearner":    ### Only Immediate state learner for now
                            continue
                        if fs[0] == tcm:
                            results.append(torch.load(path + rel_path + filename)["Perplexity"])
                        
                    values[key][i + 1] = np.mean(results)
                    if get_intervals:
                        intervals[key][i+1] = st.t.interval(alpha=0.95, df=len(results)-1, loc=np.mean(results), scale=st.sem(results))

    return values, intervals

values, intervals = load_results(path, configurations, get_intervals = args.get_intervals)
make_spider(values, intervals, configurations)
if args.extension == "svg":
    plt.savefig(path + "gen_eval.svg", bbox_inches = 'tight')
elif args.extension == "png":
    plt.savefig(path + "gen_eval.png", dpi=300, bbox_inches='tight')