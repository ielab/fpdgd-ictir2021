from typing import NamedTuple
import numpy as np
import copy
from utils import evl_tool
from clickModel.click_simulate import CcmClickModel
from data.LetorDataset import LetorDataset
from utils.dp import gamma_noise

# The message that each client send to the server:
# 1.updated parameters from client
# 2.volume of data that client use for each update
ClientMessage = NamedTuple("ClientMessage",[("gradient", np.ndarray), ("parameters", np.ndarray), ("n_interactions", int)])

# Metric values (ndcg@k, mrr@k) of each client averaged on whole batch (computed by relevance label)
ClientMetric = NamedTuple("ClientMetric", [("mean_ndcg", float), ("mean_mrr", float), ("ndcg_list", list), ("mrr_list", list)])

class RankingClient:
    """
    emulate clients
    """
    def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model: CcmClickModel, sensitivity, epsilon, enable_noise, n_clients):
        """
        :param dataset: representing a (query -> {document relevances, document features}) mapping
                for the queries the client can submit
        :param init_model: A ranking model
        :param seed: random seed used to generate queries for client
        :param click_model: A click model that emulate the user's behaviour; together with a click metric it is used
                to reflect the ranking quality
        :param sensitivity: set global sensitivity of ranking model
        :param epsilon: privacy budget
        :param enable_noise: use differential privacy noise or not
        :param n_clients: number of clients
        """
        self.dataset = dataset
        self.model = copy.deepcopy(init_model)
        self.random_state = np.random.RandomState(seed)
        self.click_model = click_model
        self.query_set = dataset.get_all_querys()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = enable_noise
        self.n_clients = n_clients

    def update_model(self, model) -> None:
        """
        Update the client-side model
        :param model: The new model
        :return: None
        """
        self.model = copy.deepcopy(model)

    # evaluation metric: ndcg@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k = 10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg/idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k = 10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            if r > 0 and got_rr == False: # TODO: decide the threshold value for relevance label
                rr = 1/(i+1)
                got_rr = True

        return rr

    def client_ranker_update(self, n_interactions: int, multi_update=True):
        """
        Run submits queries to a ranking model and gets its performance (eval metrics) and updates gradient / models
        :param n_interactions:
        :param ranker:
        :return:
        """
        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        index = self.random_state.randint(self.query_set.shape[0], size=n_interactions) # randomly choose queries for simulation on each client (number of queries based on the set n_interactions)
        gradients = np.zeros(self.dataset._feature_size) # initialize gradient
        for i in range(n_interactions): # run in batches
            id = index[i]
            qid = self.query_set[id]

            ranking_result, scores = self.model.get_query_result_list(self.dataset, qid)
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance
            # # compute online performance
            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance)) # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset,ranking_result,qid,10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_label = self.click_model(ranking_relevance, self.random_state)

            g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else: # accumulate gradients in batch (sum)
                gradients += g

        if not multi_update:
            self.model.update_to_gradients(gradients)

        updated_weights = self.model.get_current_weights()

        ## add noise
        if self.model.enable_noise:
            noise = gamma_noise(np.shape(updated_weights), self.sensitivity, self.epsilon, self.n_clients)

            updated_weights += noise

        mean_client_ndcg = np.mean(per_interaction_client_ndcg)
        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg, mean_mrr=mean_client_mrr, ndcg_list=per_interaction_client_ndcg, mrr_list=per_interaction_client_mrr)
