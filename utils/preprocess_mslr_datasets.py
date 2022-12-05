import torch
import os
from tqdm import tqdm
from pprint import pprint


def split(dataset : str):
    chunks = [f for f in os.listdir(dataset) if f.startswith("data")]

    full_chunks = False     # The datasets we generate are small, no need to split them into chunks
    if full_chunks:
        train_chunks = chunks[:int(18 * len(chunks)//20)]
        test_chunks = chunks[int(18 * len(chunks)//20):int(19 * len(chunks)//20)]
        val_chunks = chunks[int(19 * len(chunks)//20):]
    else:
        train_chunks = chunks

    train = {}
    val = {}
    test = {}

    for i, chunk in tqdm(enumerate(chunks), total = len(chunks)):
        data = torch.load(dataset + chunk)
        if chunk in train_chunks:
            if full_chunks:
                train = {**train, **data}
            else:
                items = list(data.items())
                train = {**train, **dict(items[: int(8 * len(data) // 10)])}
                val = {**val, **dict(items[int(8 * len(data) // 10) : int(9 * len(data) // 10)])}
                test = {**test, **dict(items[int(9 * len(data) // 10):])}
        elif chunk in val_chunks:
            val = {**val, **data}
        elif chunk in test_chunks:
            test = {**test, **data}
        else:
            print("???")
        
    debug = dict(list(test.items())[:1280])

    torch.save(train, dataset + "train.pt")
    torch.save(val, dataset + "val.pt")
    torch.save(test, dataset + "test.pt")
    torch.save(debug, dataset + "debug.pt")
        
def process(dataset : str):
    all_files = [f for f in os.listdir(dataset) if "train" in f or "val" in f or "test" in f]
    all_files_train = [f for f in os.listdir(dataset) if "train" in f]

    ## To be modified
    serp_feat_dim = 0
    user_feat_dim = 1
    rec_size = 10

    ## Init
    pair_dataset = True # In MSLR, doc IDs are defined per query
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
        sessions = torch.load(dataset + chunk)
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
    torch.save(dimensions, dataset + "dimensions.pt")
    torch.save(torch.tensor(traj_lens), dataset + "traj_lens.pt")

    print("Number of pairs in data :", len(pairs_in_data))
    torch.save(pairs_in_data, dataset + "pair_embedding_dict.pt")
    torch.save(click_freq, dataset + 'click_freq.pt')
    torch.save(obs_freq, dataset + 'obs_freq.pt')
    torch.save(click_freq_rank, dataset + 'click_freq_rank.pt')
    torch.save(obs_freq_rank, dataset + 'obs_freq_rank.pt')
    torch.save(click_freq_before_last, dataset + 'click_freq_before_last.pt')
    torch.save(obs_freq_before_last, dataset + 'obs_freq_before_last.pt')
    torch.save(click_freq_last, dataset + 'click_freq_last.pt')

def filter_ground_truth(dataset : str, cutoff_frequency : int = 0):
    ##### Filtered ground truth data
    relabeling = False
    gt = torch.load(dataset + "../ground_truth.pt")

    print("Number of pairs before any filtering : ", len(gt))

    # Remove pairs that appear less than X times
    obs_freq = torch.load(dataset + "obs_freq.pt")
    cutoff_gt = {}

    max_q = 0
    for (q,doc), rel in gt.items():
        if relabeling:
            if q in qid_dict:
                q = qid_dict[q]
            if doc in did_dict:
                doc = did_dict[doc]
        max_q = max(q, max_q)
        if (q,doc) in obs_freq and obs_freq[q,doc] > cutoff_frequency:
            cutoff_gt[q,doc] = rel

    print("Number of frequent pairs : ", len(cutoff_gt))
    # Then remove unnecessary lists
    n_queries = max_q

    relevances = [[] for i in range(n_queries + 1)]
    docs_per_query = [[] for i in range(n_queries + 1)]
    for (query, doc), rel in cutoff_gt.items():
        relevances[query].append(rel)
        docs_per_query[query].append(doc)

    # Then filter out all queries who don't have any document associated with them or whose documents all have the same relevance
    check_equal = lambda l, v = None : (l == [] or (check_equal(l[1:], l[0]) if v is None else (l[0] == v and check_equal(l[1:], v))))

    queries = [idx for idx,value in enumerate(relevances) if len(value) > 0 and not check_equal(value)]
    docs_per_query = [value for idx, value in enumerate(docs_per_query) if len(value) > 0 and not check_equal(relevances[idx])]
    relevances = [value for value in relevances if len(value) > 0 and not check_equal(value)]
    nq = len(relevances)
    print("Number of queries with non-constant relevance : ", nq)


    # Then randomly rearrange ground truth data
    item_permutation = [torch.randperm(len(relevances[q])).tolist() for q in range(nq)]
    relevances = [[relevances[q][i] for i in item_permutation[q]] for q in range(nq)]
    docs_per_query = [[docs_per_query[q][i] for i in item_permutation[q]] for q in range(nq)]

    new_gt = {}
    for q, docs, rels in zip(queries, docs_per_query, relevances):
        for doc, rel in zip(docs, rels):
            new_gt[q, doc] = rel

    new_gt_rerank = {}
    compt = 0
    for q, docs, rels in zip(queries, docs_per_query, relevances):
        gt_query = {"rec_lists" : torch.tensor(docs).unsqueeze(0),
                    "clicks" : None,
                    "user_feat" : torch.tensor([[q]]),
                    "serp_feat" : None,
                    "inter_feat" : None,
                    "relevances" : torch.tensor(rels)}
                    #"relevances" : torch.tensor([torch.mean(torch.tensor(rel, dtype = torch.float)) for rel in rels])}
        new_gt_rerank[compt] = gt_query
        compt += 1

    print("Number of pairs after processing : ", len(new_gt))

    if cutoff_frequency > 0:
        cutoff = "_cutoff" + str(cutoff_frequency)
    else:
        cutoff = ""
    torch.save(new_gt, dataset + "ground_truth" + cutoff + ".pt")
    torch.save(new_gt_rerank, dataset + "ground_truth_rerank" + cutoff + ".pt")

def filter_relevances(dataset : str):
    gen_obs_freq_q = True

    relevances = torch.load(dataset + "../relevances_sampled.pt")
    bm25 = torch.load(dataset + "../bm25_sampled.pt")
    lmir = torch.load(dataset + "../lmir_dir_sampled.pt")
    lambdamart = torch.load(dataset + "../lambdamart_sampled.pt")
    #noisy_oracle = torch.load(dataset + "../noisy_oracle_sampled.pt")
    gt = torch.load(dataset + "../ground_truth.pt")

    if gen_obs_freq_q:
        obs_freq = torch.load(dataset + "obs_freq.pt")
        obs_freq_q = {qid : torch.empty(len(relevances[qid]), dtype = torch.long) for qid in relevances.keys()}
        for (q,d) in gt.keys():
            if (q,d) in obs_freq:
                obs_freq_q[q][d] = obs_freq[q,d]
            else:
                obs_freq_q[q][d] = 0
        torch.save(obs_freq_q, dataset + "obs_freq_q.pt")
    else:
        obs_freq_q = torch.load(dataset + "obs_freq_q.pt")

    # relevances = {qid : rels[(obs_freq_q[qid] > 0)] for qid, rels in relevances.items()}
    # bm25 = {qid : b[(obs_freq_q[qid] > 0)] for qid, b in bm25.items()}
    #bm25 = {qid : b for (qid, b), rels in zip(bm25.items(), relevances.values()) if len(rels) > 0 and torch.max(rels) != torch.min(rels)}
    #lmir = {qid : l[(obs_freq_q[qid] > 0)] for qid, l in lmir.items()}
    #lmir = {qid : l for (qid, l), rels in zip(lmir.items(), relevances.values()) if len(rels) > 0 and torch.max(rels) != torch.min(rels)}
    # noisy_oracle = {qid : l[(obs_freq_q[qid] > 0)] for qid, l in noisy_oracle.items()}
    # noisy_oracle = {qid : l for (qid, l), rels in zip(noisy_oracle.items(), relevances.values()) if len(rels) > 0 and torch.max(rels) != torch.min(rels)}
    # relevances = {qid : rels for qid, rels in relevances.items() if len(rels) > 0 and torch.max(rels) != torch.min(rels)}
    # obs_freq_q = {qid : freqs[(obs_freq_q[qid] > 0)] for qid, freqs in obs_freq_q.items()}
    # obs_freq_q = {qid : obs_freq_q[qid] for qid in relevances.keys()}

    torch.save(relevances, dataset + "relevances.pt")
    #torch.save(obs_freq_q, dataset + "obs_freq_q.pt")
    torch.save(bm25, dataset + "bm25.pt")
    torch.save(lmir, dataset + "lmir_dir.pt")
    torch.save(lambdamart, dataset + "lambdamart.pt")
    #torch.save(noisy_oracle, dataset + "noisy_oracle.pt")


def preprocess_dataset(dataset : str):
    print("         -> Splitting the data into train/test/val")
    split(dataset)
    print("         -> Process the dataset and get relevant statistics")
    process(dataset)
    print("         -> Filter ground truth")
    filter_ground_truth(dataset)
    print("         -> Filter relevances.")
    filter_relevances(dataset)

