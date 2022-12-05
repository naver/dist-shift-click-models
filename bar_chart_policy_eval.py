import numpy as np
import scipy.stats as st
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from argparse import ArgumentParser

plt.rcParams['text.usetex'] = True


parser = ArgumentParser()
parser.add_argument("--path", type = str, required = True, help = "Path to datasets")
parser.add_argument("--get_intervals", type = bool, default = False, help = "Whether to plot confidence bounds")
parser.add_argument("--extension", type=str, default = "png", choices = ["png", "svg"], help = "File extension.")
args = parser.parse_args()

configurations = {"TCM" : ["PBM", "UBM", "DBN", "NCM", "ARM", "CACM_minus"], 
                    "ICM" : ["CoCM_mismatch", "CoCM"],
                    "TRP" : ["oracle", "-lambdamart", "-bm25"],
                    "policy" : ["MAXREWARD", "TOPDOWN"]}

CTRs = {}
if args.get_intervals:
    CTR_bounds = {}
for icm in configurations["ICM"]:
    for trp in configurations["TRP"]:
        rel_path = icm + "-plackettluce" + trp + "/policies/datasets/CTR/"
        for policy in configurations["policy"]:
            key = icm + "_" + trp + "_" + policy
            CTRs[key] = np.zeros(len(configurations["TCM"]))
            if args.get_intervals:
                CTR_bounds[key] = np.zeros((len(configurations["TCM"]), 2))
        for i, tcm in enumerate(configurations["TCM"]):
            all_files = os.listdir(args.path + rel_path)
            results = {policy : [] for policy in configurations["policy"]}
            for filename in all_files:
                fs = filename.split("_")
                if fs[0] == tcm:
                    ctr = torch.load(args.path + rel_path + filename)
                    for policy in configurations["policy"]:
                        results[policy].append(ctr[policy])     
            for policy in configurations["policy"]:
                key = icm + "_" + trp + "_" + policy
                CTRs[key][i] = np.mean(results[policy])
                if args.get_intervals:
                    CTR_bounds[key][i] = st.t.interval(alpha=0.95, df=len(results[policy])-1, loc=np.mean(results[policy]), scale=st.sem(results[policy]))

fig = plt.figure(figsize = (11, 5))
#subfig1, subfig2 = fig.subfigures(1, 2)
spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1, 1], height_ratios=[7, 1])

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax1) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0])
#ax1, ax2 = subfig1.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [7, 1]})
fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# plot the same data on both axes
ax1.bar([0], [0], color = "black", label = "Max-Reward")
ax1.bar([0], [0], edgecolor = "black", color = "none", linewidth = 1.5, label = "Top-Down")
ax1.bar(np.zeros(1), np.zeros(1), color='w', alpha=0, label=' ')

#Top plot
ax1.bar(np.arange(6)-0.3, CTRs["CoCM_oracle_MAXREWARD"], width = 0.3, label = "PL-oracle", color = "lightskyblue")
ax1.bar(np.arange(6)-0.3, CTRs["CoCM_oracle_TOPDOWN"], width = 0.27, edgecolor = "royalblue", color = 'none', linewidth = 1.5)
ax1.bar(np.arange(6), CTRs["CoCM_-lambdamart_MAXREWARD"], width = 0.3, label = "PL-$\lambda$MART", color = "sandybrown")
ax1.bar(np.arange(6), CTRs["CoCM_-lambdamart_TOPDOWN"], width = 0.27, edgecolor = "sienna", color = 'none', linewidth = 1.5)
ax1.bar(np.arange(6)+0.3, CTRs["CoCM_-bm25_MAXREWARD"], width = 0.3, label = "PL-bm25", color = "palegreen")
ax1.bar(np.arange(6)+0.3, CTRs["CoCM_-bm25_TOPDOWN"], width = 0.27, edgecolor = "forestgreen", color = 'none', linewidth = 1.5)

# Confidence bounds
if args.get_intervals:
    for k in range(6):
        for pol in ["_MAXREWARD", "_TOPDOWN"]:
            for log_pol, offset in zip(["oracle", "-lambdamart", "-bm25"], [-0.3, 0.0, 0.3]):
                ax1.plot([k + offset - 0.05, k + offset + 0.05], 
                            [CTR_bounds["CoCM_" + log_pol + pol][k,0], CTR_bounds["CoCM_" + log_pol + pol][k,0]], 
                            color = "dimgrey", linewidth = 1)
                ax1.plot([k + offset - 0.05, k + offset + 0.05], 
                            [CTR_bounds["CoCM_" + log_pol + pol][k,1], CTR_bounds["CoCM_" + log_pol + pol][k,1]], 
                            color = "dimgrey", linewidth = 1)
                ax1.plot([k + offset, k + offset], 
                            [CTR_bounds["CoCM_" + log_pol + pol][k,0], CTR_bounds["CoCM_" + log_pol + pol][k,1]], 
                            color = "dimgrey", linewidth = 1)

#Bottom plot
ax2.bar(np.arange(6)-0.3, CTRs["CoCM_oracle_MAXREWARD"], width = 0.3, color = "lightskyblue")
ax2.bar(np.arange(6)-0.3, CTRs["CoCM_oracle_TOPDOWN"], width = 0.27, edgecolor = "royalblue", color = 'none', linewidth = 1.5)
ax2.bar(np.arange(6), CTRs["CoCM_-lambdamart_MAXREWARD"], width = 0.3, color = "sandybrown")
ax2.bar(np.arange(6), CTRs["CoCM_-lambdamart_TOPDOWN"], width = 0.27, edgecolor = "sienna", color = 'none', linewidth = 1.5)
ax2.bar(np.arange(6)+0.3, CTRs["CoCM_-bm25_MAXREWARD"], width = 0.3, color = "palegreen")
ax2.bar(np.arange(6)+0.3, CTRs["CoCM_-bm25_TOPDOWN"], width = 0.27, edgecolor = "forestgreen", color = 'none', linewidth = 1.5)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(0.066, 0.092)  # top axis
ax2.set_ylim(0, 0.005)  # bottom axis
ax2.set_xlim(ax1.get_xlim())


# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.right.set_visible(False)
ax1.spines.right.set_visible(False)
ax1.set_xticks([])
#ax1.xaxis.tick_top()
#ax2.tick_params(labelleft=False)  # don't put tick labels at the bottom left
ax2.set_yticks([0.0])
ax2.set_yticklabels([0.0])
ax1.set_yticks([0.07 + 0.004 * k for k in range(0, 6)])
ax1.set_yticklabels([round(0.07 + 0.004 * k, 3) for k in range(0, 6)])
ax2.set_xticks(np.arange(6))
ax2.set_xticklabels(["PBM", "UBM", "DBN", "NCM", "CACM\\textsuperscript{\\textcircled{-}}", "ARM"])

# Now, let's turn towards the cut-out slanted lines.
# We create line objects in axes coordinates, in which (0,0), (0,1),
# (1,0), and (1,1) are the four corners of the axes.
# The slanted lines themselves are markers at those locations, such that the
# lines keep their angle and position, independent of the axes size or scale
# Finally, we need to disable clipping.

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)

handles, labels = ax1.get_legend_handles_labels()
#handles[1].set_color("black")
leg = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.05, 0.5), ncol = 1, frameon = False)
#leg = ax1.legend(loc = "best", ncol = 2)
#leg.legendHandles[0].set_color("black"])
#leg.legendHandles[1].set_color("black")



# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax1) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
ax1 = fig.add_subplot(spec[0, 1])
ax2 = fig.add_subplot(spec[1, 1])
#ax1, ax2 = subfig2.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [7, 1]})
#subfig2.subplots_adjust(hspace=0.05)  # adjust space between axes

#Top plot
ax1.bar(np.arange(6)-0.3, CTRs["CoCM_mismatch_oracle_MAXREWARD"], width = 0.3, label = "PL-oracle", color = "lightskyblue")
ax1.bar(np.arange(6)-0.3, CTRs["CoCM_mismatch_oracle_TOPDOWN"], width = 0.27, label = "Top-down", edgecolor = "royalblue", color = 'none', linewidth = 1.5)
ax1.bar(np.arange(6), CTRs["CoCM_mismatch_-lambdamart_MAXREWARD"], width = 0.3, label = "PL-$\lambda$MART", color = "sandybrown")
ax1.bar(np.arange(6), CTRs["CoCM_mismatch_-lambdamart_TOPDOWN"], width = 0.27, edgecolor = "sienna", color = 'none', linewidth = 1.5)
ax1.bar(np.arange(6)+0.3, CTRs["CoCM_mismatch_-bm25_MAXREWARD"], width = 0.3, label = "PL-bm25", color = "palegreen")
ax1.bar(np.arange(6)+0.3, CTRs["CoCM_mismatch_-bm25_TOPDOWN"], width = 0.27, edgecolor = "forestgreen", color = 'none', linewidth = 1.5)

# Confidence bounds
if args.get_intervals:
    for k in range(6):
        for pol in ["_MAXREWARD", "_TOPDOWN"]:
            for log_pol, offset in zip(["oracle", "-lambdamart", "-bm25"], [-0.3, 0.0, 0.3]):
                ax1.plot([k + offset - 0.05, k + offset + 0.05], 
                            [CTR_bounds["CoCM_mismatch_" + log_pol + pol][k,0], CTR_bounds["CoCM_mismatch_" + log_pol + pol][k,0]], 
                            color = "grey", linewidth = 1)
                ax1.plot([k + offset - 0.05, k + offset + 0.05], 
                            [CTR_bounds["CoCM_mismatch_" + log_pol + pol][k,1], CTR_bounds["CoCM_mismatch_" + log_pol + pol][k,1]], 
                            color = "grey", linewidth = 1)
                ax1.plot([k + offset, k + offset], 
                            [CTR_bounds["CoCM_mismatch_" + log_pol + pol][k,0], CTR_bounds["CoCM_mismatch_" + log_pol + pol][k,1]], 
                            color = "grey", linewidth = 1)

#Bottom plot
ax2.bar(np.arange(6)-0.3, CTRs["CoCM_mismatch_oracle_MAXREWARD"], width = 0.3, label = "PL-oracle", color = "lightskyblue")
ax2.bar(np.arange(6)-0.3, CTRs["CoCM_mismatch_oracle_TOPDOWN"], width = 0.27, label = "Top-down", edgecolor = "royalblue", color = 'none', linewidth = 1.5)
ax2.bar(np.arange(6), CTRs["CoCM_mismatch_-lambdamart_MAXREWARD"], width = 0.3, label = "PL-$\lambda$MART", color = "sandybrown")
ax2.bar(np.arange(6), CTRs["CoCM_mismatch_-lambdamart_TOPDOWN"], width = 0.27, edgecolor = "sienna", color = 'none', linewidth = 1.5)
ax2.bar(np.arange(6)+0.3, CTRs["CoCM_mismatch_-bm25_MAXREWARD"], width = 0.3, label = "PL-bm25", color = "palegreen")
ax2.bar(np.arange(6)+0.3, CTRs["CoCM_mismatch_-bm25_TOPDOWN"], width = 0.27, edgecolor = "forestgreen", color = 'none', linewidth = 1.5)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(0.066, 0.092)  # top axis
ax2.set_ylim(0, 0.005)  # bottom axis
ax2.set_xlim(ax1.get_xlim())


# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.right.set_visible(False)
ax1.spines.right.set_visible(False)
ax1.set_xticks([])
#ax1.xaxis.tick_top()
#ax2.tick_params(labelleft=False)  # don't put tick labels at the bottom left
ax2.set_yticks([0.0])
ax2.set_yticklabels([0.0])
ax1.set_yticks([0.07 + 0.004 * k for k in range(0, 6)])
ax1.set_yticklabels([round(0.07 + 0.004 * k, 3) for k in range(0, 6)])
ax2.set_xticks(np.arange(6))
ax2.set_xticklabels(["PBM", "UBM", "DBN", "NCM", "CACM\\textsuperscript{\\textcircled{-}}", "ARM"])

# Now, let's turn towards the cut-out slanted lines.
# We create line objects in axes coordinates, in which (0,0), (0,1),
# (1,0), and (1,1) are the four corners of the axes.
# The slanted lines themselves are markers at those locations, such that the
# lines keep their angle and position, independent of the axes size or scale
# Finally, we need to disable clipping.

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)

plt.savefig(args.path + "policy_eval." + args.extension, bbox_inches = 'tight')