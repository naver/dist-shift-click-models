import torch
from modules.argument_parser import MyParser

'''
    This script filters ground truth to only keep pairs appearing in the train dataset and queries with non-all-equal relevance labels.
'''

parser = MyParser()
main_parser.add_argument('--dataset', type=str, required = True, help='Path to yandex dataset.')
args = parser.parse_args()

data_path = args.dataset

cutoff_frequency = 0

##### Filtered ground truth data
gt = torch.load(data_path + "../ground_truth.pt")

print("Number of pairs before any filtering : ", len(gt))

# Remove pairs that appear less than X times
obs_freq = torch.load(data_path + "obs_freq.pt")
cutoff_gt = {}

max_q = 0
for (q,doc), rel in gt.items():
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
torch.save(new_gt, data_path + "ground_truth" + cutoff + ".pt")
torch.save(new_gt_rerank, data_path + "ground_truth_rerank" + cutoff + ".pt")




