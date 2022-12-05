import torch
from tqdm import tqdm
import os
from pprint import pprint
from modules.argument_parser import MyParser

'''
    This script extracts useful dimensions and click statistics from the dataset.
'''

parser = MyParser()
main_parser.add_argument('--dataset', type=str, required = True, help='Path to yandex dataset.')
args = parser.parse_args()

data_path = args.dataset
all_files = [f for f in os.listdir(data_path) if "train" in f or "val" in f or "test" in f]
all_files_train = [f for f in os.listdir(data_path) if "train" in f]

## To be modified for other datasets
serp_feat_dim = 0
user_feat_dim = 1
rec_size = 10

## Init
pair_dataset = False
pairs_in_data = {}
click_freq, click_freq_rank = {}, {}
obs_freq, obs_freq_rank = {}, {}
obs_freq_before_last, click_freq_before_last, click_freq_last = {}, {}, {}
compt = 0
num_user_embedd = [torch.tensor(0) for _ in range(user_feat_dim)]
num_serp_embedd = [torch.tensor(0) for _ in range(serp_feat_dim)]
num_items = 0
traj_lens = []

## Main loop
for i, chunk in tqdm(enumerate(all_files), total = len(all_files)):
    sessions = torch.load(data_path + chunk)
    with tqdm(total = len(sessions), leave = False) as pbar:
        for sess_id, sess in sessions.items():
            traj_lens.append(len(sess["user_feat"]))
            rec_size = len(sess["clicks"][0])
            items = sess["rec_lists"].flatten()
            clicks = sess["clicks"].flatten()
            last_click = [torch.nonzero(cl)[-1].item() if torch.sum(cl) > 0  else len(cl) for cl in sess["clicks"]]
            user_feat = sess["user_feat"]
            if serp_feat_dim > 0:
                serp_feat = sess["serp_feat"].flatten(end_dim = -1)
            
            for idx, (item, click) in enumerate(zip(items, clicks)):
                doc_id = item.item()
                pid = idx // rec_size
                rank = idx % rec_size
                query_id = user_feat[pid, 0].item()
                c = click.item()

                ### Check if pair is already seen
                if (query_id, doc_id) not in pairs_in_data:
                    pairs_in_data[query_id, doc_id] = compt
                    compt += 1
                
                ### Update observation and click frequency
                if chunk in all_files_train:
                    if (query_id,doc_id) in click_freq:
                        obs_freq[query_id,doc_id] += 1
                        click_freq[query_id,doc_id] += c

                        if (query_id,doc_id) in obs_freq_before_last:
                            if rank <= last_click[pid]:
                                obs_freq_before_last[query_id, doc_id] += 1
                                click_freq_before_last[query_id, doc_id] += c
                                if rank == last_click[pid]:
                                    click_freq_last[query_id, doc_id] += 1 
                        else:
                            if rank <= last_click[pid]:
                                obs_freq_before_last[query_id, doc_id] = 1
                                click_freq_before_last[query_id, doc_id] = c
                                if rank == last_click[pid]:
                                    click_freq_last[query_id, doc_id] = 1 
                                else:
                                    click_freq_last[query_id, doc_id] = 0

                        if (query_id, doc_id, rank) in click_freq_rank:
                            obs_freq_rank[query_id,doc_id, rank] += 1
                            click_freq_rank[query_id, doc_id, rank] += c
                        else:
                            obs_freq_rank[query_id,doc_id, rank] = 1
                            click_freq_rank[query_id, doc_id, rank] = c
                    else:
                        obs_freq[query_id,doc_id] = 1
                        click_freq[query_id,doc_id] = c

                        obs_freq_rank[query_id,doc_id, rank] = 1
                        click_freq_rank[query_id, doc_id, rank] = c

                        if rank <= last_click[pid]:
                            obs_freq_before_last[query_id, doc_id] = 1
                            click_freq_before_last[query_id, doc_id] = c
                            if rank == last_click[pid]:
                                click_freq_last[query_id, doc_id] = 1 
                            else:
                                click_freq_last[query_id, doc_id] = 0
            
            ### Get dimensions of dataset
            num_items = max(num_items, torch.max(items))
            max_user_feat = torch.max(user_feat, dim = 0)[0]
            if serp_feat_dim > 0:
                max_serp_feat = torch.max(serp_feat, dim = 0)[0]
            for i in range(user_feat_dim):
                num_user_embedd[i] = max(num_user_embedd[i], max_user_feat[i])
            for i in range(serp_feat_dim):
                num_serp_embedd[i] = max(num_serp_embedd[i], max_serp_feat[i])

            pbar.update(1)

if pair_dataset : 
    num_items = torch.tensor(len(pairs_in_data))

## Saving
dimensions = {"num_items" : num_items.item() + 1, 
                "rec_size" : rec_size,
                "user_feat_dim" : user_feat_dim,
                "num_user_embedd" : [t.item() + 1 for t in num_user_embedd],
                "user_feat_type" : "int",
                "serp_feat_dim" : serp_feat_dim,
                "num_serp_embedd" : [t.item() + 1 for t in num_serp_embedd],
                "serp_feat_type" : "int",
                "pair_dataset" : pair_dataset}

print("Dimensions :")
pprint(dimensions)
torch.save(dimensions, data_path + "dimensions.pt")
torch.save(torch.tensor(traj_lens), data_path + "traj_lens.pt")

print("Number of pairs in data :", len(pairs_in_data))
torch.save(pairs_in_data, data_path + "pair_embedding_dict.pt")
torch.save(click_freq, data_path + 'click_freq.pt')
torch.save(obs_freq, data_path + 'obs_freq.pt')
torch.save(click_freq_rank, data_path + 'click_freq_rank.pt')
torch.save(obs_freq_rank, data_path + 'obs_freq_rank.pt')
torch.save(click_freq_before_last, data_path + 'click_freq_before_last.pt')
torch.save(obs_freq_before_last, data_path + 'obs_freq_before_last.pt')
torch.save(click_freq_last, data_path + 'click_freq_last.pt')