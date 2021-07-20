from typing import Dict, Any, NamedTuple, List
import numpy as np
from tqdm import tqdm

from client.client import RankingClient
from ranker.PDGDLinearRanker import PDGDLinearRanker
from data.LetorDataset import LetorDataset

TrainResult = NamedTuple("TrainResult", [
    ("ranker", PDGDLinearRanker),
    ("ndcg_server", list),
    ("mrr_server", list),
    ("ndcg_client", list),
    ("mrr_client", list)
])



def train_uniform(params: Dict[str, Any], traindata: LetorDataset, testdata: LetorDataset, message, num_update=None) -> TrainResult:
    """

    :param params:
    :param traindata: dataset used for training server ranker
    :param testdata: dataset used for testing true performance of server ranker - using true relevance label
    :param message:
    :return:
    """
    seed = params["seed"]
    np.random.seed(seed)

    n_clients = params["n_clients"]
    interactions_per_feedback = params["interactions_per_feedback"]
    click_model = params["click_model"]
    ranker = params["ranker_generator"]
    multi_update = params["multi_update"]
    sensitivity = params["sensitivity"]
    epsilon = params["epsilon"]
    enable_noise = params["enable_noise"]

    clients = [RankingClient(traindata, ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients, client_id) for client_id in range(n_clients)]

    n_iterations = params["interactions_budget"] // n_clients // interactions_per_feedback # total iteration times (training times) for federated training

    ndcg_server = [] # off-line metric (on testset)
    mrr_server = [] # off-line metric (on testset)
    ndcg_clients = [] # averaged online metric
    mrr_clients = [] # averaged online metric

    # initialize gradient
    gradients = np.zeros(traindata._feature_size)
    for i in tqdm(range(n_iterations), desc=message):
        i += 1
        feedback = []
        online_ndcg = []
        online_mrr = []
        for client in clients:
            client_message, client_metric = client.client_ranker_update(interactions_per_feedback, multi_update)
            feedback.append(client_message)
            # online evaluation
            online_ndcg.append(client_metric.mean_ndcg)
            online_mrr.append(client_metric.mean_mrr)

        # online-line metrics
        ndcg_clients.append(np.mean(online_ndcg))
        mrr_clients.append(np.mean(online_mrr))

        # off-line metrics
        if num_update is not None:
            if i % int((n_iterations/num_update))== 0:
                all_result = ranker.get_all_query_result_list(testdata)
                ndcg = average_ndcg_at_k(testdata, all_result, 10)
                mrr = average_mrr_at_k(testdata, all_result, 10)
                ndcg_server.append(ndcg)
                mrr_server.append(mrr)

        else:

            all_result = ranker.get_all_query_result_list(testdata)
            ndcg = average_ndcg_at_k(testdata, all_result, 10)
            mrr = average_mrr_at_k(testdata, all_result, 10)
            ndcg_server.append(ndcg)
            mrr_server.append(mrr)

        # train the server ranker (clients send feedback to the server)
        ranker.federated_averaging_weights(feedback)

        # the server send the newly trained model to every client
        for client in clients:
            client.update_model(ranker)

    return TrainResult(ranker=ranker, ndcg_server = ndcg_server, mrr_server=mrr_server, ndcg_client=ndcg_clients, mrr_client=mrr_clients)



# generate metrics: ndcg@k & mrr@10
def average_ndcg_at_k(dataset, query_result_list, k):
    ndcg = 0.0
    num_query = 0
    for query in dataset.get_all_querys():
        if len(dataset.get_relevance_docids_by_query(query)) == 0:  # for this query, ranking list is None
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

        if idcg != 0:
            ndcg += (dcg / idcg)

        num_query += 1
    return ndcg / float(num_query)



def average_mrr_at_k(dataset: LetorDataset, query_result_list, k):
    rr = 0
    num_query = 0
    for query in dataset.get_all_querys():
        if len(dataset.get_relevance_docids_by_query(query)) == 0: # for this query, ranking list is None
            continue
        got_rr = False
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            if relevance in {1,2,3,4} and got_rr == False:
                rr += 1/(i+1)
                got_rr = True

        num_query += 1
    return rr / float(num_query)
