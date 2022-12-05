import torch
from tqdm import tqdm
from pathlib import Path
import os

from modules.argument_parser import MyParser


'''
    This scripts parses yandex relevance prediction data.
    We can create a session-based or serp_based dataset depending on the value of the boolean session.
    We discard sessions/serps without clicks and serps whose query is not labeled in the ground truth data.
'''

parser = MyParser()
main_parser.add_argument('--path', type=str, required = True, help='Path to yandex dataset.')
main_parser.add_argument('--parse_gt', type=parser.str2bool, default = True, help='Whether we want to parse the ground truth.')
main_parser.add_argument('--parse_logs', type=parser.str2bool, default = True, help='Whether we want to parse the click logs.')
main_parser.add_argument('--session', type=parser.str2bool, default = False, help='Toggle session-based parsing.')
main_parser.add_argument('--chunk_size', type=int, default = 1000000, help='Number of sqmples in one chunk.')
args = parser.parse_args()

parse_gt = args.parse_gt
parse_logs = args.parse_logs
session = args.session
chunk_size = args.chunk_size

data_dir = args.path

if session:
    data_type = "session_based"
else:
    data_type = "serp_based"


if parse_gt:
    qid_dict = {}
    did_dict = {}

    count_q = 0
    count_d = 0

    f = open(data_dir + "relevance_prediction/Trainq.txt", "r")
    prev_q = -1

    gt_rerank = {}
    gt = {}

    rels = []

    for line in tqdm(f.readlines(), total = 41275):
        ls = line[:-1].split('\t')

        qid = int(ls[0])
        if qid in qid_dict:
            qid = qid_dict[qid]
            if qid != prev_q:
                gt_rerank[prev_q] = rels
                rels = []
        else:
            if prev_q != -1:
                gt_rerank[prev_q] = rels
                rels = []
            qid_dict[qid] = count_q
            qid = count_q
            count_q += 1

            
        

        doc_id = int(ls[2])
        if doc_id in did_dict:
            doc_id = did_dict[doc_id]
        else:
            did_dict[doc_id] = count_d
            doc_id = count_d
            count_d += 1
        rel = int(ls[3])

        gt[qid, doc_id] = rel
        rels.append(rel)
        prev_q = qid
    
    gt_rerank[qid] = rels
    
    print("Number of pairs in ground truth : ", len(gt))
    print("Number of queries in ground truth : ", len(gt_rerank))
    torch.save(qid_dict, data_dir + "qid_dict.pt")
    torch.save(did_dict, data_dir + "did_dict.pt")
    torch.save(gt, data_dir + "ground_truth.pt")
    torch.save(gt_rerank, data_dir + "ground_truth_rerank.pt")



if parse_logs:
    f = open(data_dir + "relevance_prediction/YandexClicks.txt", "r")
    count = 0
    chunk_count = 1
    sess = {}
    rec_lists = torch.empty(1, 10, dtype = torch.long)
    clicks = torch.empty(1, 10, dtype = torch.long)
    queries = torch.empty(1, 1, dtype = torch.long)
    sess_len = 0
    session_id = 0

    qid_dict = torch.load(data_dir + "qid_dict.pt")
    did_dict = torch.load(data_dir + "did_dict.pt")
    count_d = len(did_dict)
    qid = -1

    with tqdm(total = 146278823) as pbar:
        for line in f:
            ls = line[:-1].split('\t')
            if ls[2] == 'Q':
                if (session and int(ls[0]) != session_id) or (not session):
                    if torch.sum(clicks) > 0 and qid in qid_dict:       ### We discard sessions / serps without clicks
                        sess[count] = {"rec_lists" : rec_lists,
                                            "clicks" : clicks,
                                            "user_feat" : queries,
                                            "serp_feat" : None,
                                            "inter_feat" : None}
                        count += 1  
                    pbar.update(1)
                    
                    rec_lists = torch.empty(1, 10, dtype = torch.long)
                    clicks = torch.zeros(1 ,10, dtype = torch.long)
                    queries = torch.empty(1, 1, dtype = torch.long)

                    sess_len = 1
                    sess_queries = {}
                    session_id = int(ls[0])


                    if len(sess) == chunk_size:
                        Path(data_dir + data_type + "/").mkdir(parents=True, exist_ok=True)
                        torch.save(sess, data_dir + data_type + "/train" + str(chunk_count) + ".pt")
                        sess = {}
                        chunk_count += 1
                else:
                    sess_len += 1
                    rec_lists = torch.cat([rec_lists, torch.empty(1, 10, dtype = torch.long)], dim = 0)
                    clicks = torch.cat([clicks, torch.zeros(1, 10, dtype = torch.long)], dim = 0)
                    queries = torch.cat([queries, torch.empty(1, 1, dtype = torch.long)], dim = 0)

                
                qid = int(ls[3])
                if qid in qid_dict:         #### Works only for serp_based
                    qid_new = qid_dict[qid]
                else:
                    continue
                docs = [int(doc) for doc in ls[5:]]
                if len(docs) != 10:
                    print("Incomplete list ...")
                    continue
                doc_idx = {doc : i for i, doc in enumerate(docs)}
                new_docs = []
                for doc_id in docs:
                    if doc_id in did_dict: 
                        new_docs.append(did_dict[doc_id])
                    else:
                        did_dict[doc_id] = count_d
                        new_docs.append(count_d)
                        count_d += 1
                
                queries[sess_len - 1, 0] = qid_new
                rec_lists[sess_len - 1] = torch.tensor(new_docs)     

                

            elif ls[2] == 'C':
                #qid = int(ls[3])
                ## Click is automatically associated with the latest SERP. That is not ideal as it could correspond to
                ## the same document but for a different query in the same session but there is no indication of
                ## the corresponding SERP in the data. If the clicked document is not aprt of the latest SERP, we simply
                ## don't register the click
                if int(ls[3]) in doc_idx:       
                    clicks[sess_len - 1, doc_idx[int(ls[3])]] = 1
            else:
                print("???")

    print("Number of unique queries : ", len(qid_dict))
    print("Number of unique docs : ", len(did_dict))
    sess[count] = {"rec_lists" : rec_lists,
                    "clicks" : clicks,
                    "user_feat" : queries,
                    "serp_feat" : None,
                    "inter_feat" : None}

    sess = dict(list(sess.items())[:chunk_size // 3])   ## More convenient for recurring validation checks
    torch.save(sess, data_dir + data_type + "/val.pt")
    os.rename(data_dir + data_type + "/train" + str(chunk_count - 1) + ".pt", data_dir + data_type + "/test.pt")



        
