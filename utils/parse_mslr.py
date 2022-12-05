import torch


def parse(path : str):
    relevances = {}
    bm25 = {}
    lmir_dir = {}
    q_id2my_id = {}
    count = 0
    for filename in ["train.txt", "test.txt", "vali.txt"]:
        f = open(path + "Fold1/" + filename, "r")

        for line in f.readlines():
            split_line = line.split(" ")

            rel = int(split_line[0])
            q_id = int(split_line[1].split(":")[1])
            bm25_rel = float(split_line[111].split(":")[1])
            lmir_dir_rel = float(split_line[121].split(":")[1])

            if q_id in q_id2my_id:
                relevances[q_id2my_id[q_id]].append(rel)
                bm25[q_id2my_id[q_id]].append(bm25_rel)
                lmir_dir[q_id2my_id[q_id]].append(lmir_dir_rel)
            else:
                q_id2my_id[q_id] = count
                count += 1
                relevances[q_id2my_id[q_id]] = [rel]
                bm25[q_id2my_id[q_id]] = [bm25_rel]
                lmir_dir[q_id2my_id[q_id]] = [lmir_dir_rel]
        
        print('     ' + filename + " done.")
    
    ### Let's keep only queries with 10+ docs
    relevances = {q_id : torch.tensor(rels) for q_id,rels in relevances.items() if len(rels) >= 10}
    bm25 = {q_id : torch.tensor(bm25_rel) for q_id,bm25_rel in bm25.items() if len(bm25_rel) >= 10}
    lmir_dir = {q_id : torch.tensor(lmir_dir_rel) for q_id,lmir_dir_rel in lmir_dir.items() if len(lmir_dir_rel) >= 10}

    ### Then let's keep only queries with nonequal relevances and normalize bm25 and lmir
    nonequal = {q : 0 for ((q, bm25_rel), (_, lmir_dir_rel), (_, rel)) in zip(bm25.items(), lmir_dir.items(), relevances.items())
                    if (torch.max(bm25_rel) - torch.min(bm25_rel)) * (torch.max(lmir_dir_rel) - torch.min(lmir_dir_rel)) * (torch.max(rel) - torch.min(rel)) != 0}
                    
    bm25 = {q : (bm25_rel - torch.min(bm25_rel)) / (torch.max(bm25_rel) - torch.min(bm25_rel)) 
                        for q, bm25_rel in bm25.items() if q in nonequal}
    bm25 = {i : bm25_rel for i, bm25_rel in enumerate(bm25.values())}
    lmir_dir = {q : (lmir_dir_rel - torch.min(lmir_dir_rel)) / (torch.max(lmir_dir_rel) - torch.min(lmir_dir_rel)) 
                        for q, lmir_dir_rel in lmir_dir.items() if q in nonequal}
    lmir_dir = {i : lmir_dir_rel for i, lmir_dir_rel in enumerate(lmir_dir.values())}

    print("     Number of queries in the dataset : ", len(bm25))

   
    relevances = {q_id : (2**rels - 1) / 15 for q_id,rels in relevances.items() if q_id in nonequal}
    dimensions = {q_id : len(rels) for q_id,rels in relevances.items()}
    dimensions = {i : length for i, length in enumerate(dimensions.values())}
    q_id2my_id = {key : val for key,val in q_id2my_id.items() if val in relevances}
    q_id2my_id = {key : i for i, key in enumerate(q_id2my_id.keys())}
    relevances = {i : rel for i, rel in enumerate(relevances.values())}


    torch.save(q_id2my_id, path + "qid_dict.pt")
    torch.save(dimensions, path + "dimensions.pt")
    torch.save(relevances, path + "relevances.pt")
    torch.save(bm25, path + "bm25.pt")
    torch.save(lmir_dir, path + "lmir_dir.pt")

def sample_queries(path : str, n = 1000):
    relevances = torch.load(path + "relevances.pt")
    bm25 = torch.load(path + "bm25.pt")
    lmir = torch.load(path + "lmir_dir.pt")
    #noisy_oracle = torch.load(path + "noisy_oracle.pt")
    perm = 1 + torch.randperm(len(relevances) - 1)[:n-1]
    sampled = [0] + perm.tolist()
    gt = {(qid, did) : relevances[qid][did].item() for qid in sampled for did in range(len(relevances[qid]))}
    relevances_sampled = {qid : relevances[qid] for qid in sampled}
    bm25_sampled = {qid : bm25[qid] for qid in sampled}
    lmir_sampled = {qid : lmir[qid] for qid in sampled}
    #noisy_oracle_sampled = {qid : noisy_oracle[qid] for qid in sampled}

    torch.save(gt, path + "ground_truth.pt")
    torch.save(relevances_sampled, path + "relevances_sampled.pt")
    torch.save(bm25_sampled, path + "bm25_sampled.pt")
    torch.save(lmir_sampled, path + "lmir_dir_sampled.pt")
    #torch.save(noisy_oracle_sampled, path + "noisy_oracle_sampled.pt")


def parse_mslr(path : str):
    print(" # 1 / Parsing the text files")
    parse(path)
    print(" # 2 / Sampling 1000 queries for evaluation")
    sample_queries(path, 1000)