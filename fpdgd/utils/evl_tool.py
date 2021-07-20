import numpy as np
from scipy import stats

def online_mrr_at_k(clicks, k):
    reciprocal_rank = 0.0
    n_docs = len(clicks)
    for i in range(min(k, n_docs)):
        if clicks[i] > 0:
            reciprocal_rank = 1.0 / (1.0 + i)
            break
    return reciprocal_rank

def query_ndcg_at_k(dataset, result_list, query, k):

    if len(dataset.get_relevance_docids_by_query(query)) == 0:
        return 0.0
    else:
        pos_docid_set = set(dataset.get_relevance_docids_by_query(query))

    dcg = 0.0
    for i in range(0, min(k, len(result_list))):
        docid = result_list[i]
        relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
        dcg += ((2 ** relevance - 1) / np.log2(i + 2))
    rel_set = []

    for docid in pos_docid_set:
        rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
    rel_set = sorted(rel_set, reverse=True)
    n = len(pos_docid_set) if len(pos_docid_set) < k else k
    idcg = 0
    for i in range(n):
        idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

    ndcg = (dcg / idcg)
    return ndcg

def average_ndcg_at_k(dataset, query_result_list, k, count_bad_query=False):
    ndcg = 0.0
    num_query = 0
    for query in dataset.get_all_querys():

        if len(dataset.get_relevance_docids_by_query(query)) == 0:
            if count_bad_query:
                num_query += 1
            continue
        else:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        dcg = 0.0
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        ndcg += (dcg / idcg)
        num_query += 1
    return ndcg / float(num_query)

def get_all_query_ndcg(dataset, query_result_list, k):
    query_ndcg = {}
    for query in dataset.get_all_querys():
        try:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        except:
            # print("Query:", query, "has no relevant document!")
            query_ndcg[query] = 0
            continue
        dcg = 0.0
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        ndcg = (dcg / idcg)
        query_ndcg[query] = ndcg
    return query_ndcg

def ttest(l1, l2):
    _, p = stats.ttest_ind(l1, l2, equal_var=False)
    return p
