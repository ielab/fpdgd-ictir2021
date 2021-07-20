
from ranker.AbstractRanker import AbstractRanker
import numpy as np
from scipy.linalg import norm
import copy


class LinearRanker(AbstractRanker):

    def __init__(self, num_features, learning_rate, learning_rate_decay=1, learning_rate_clip=0.01, random_initial=True):
        super().__init__(num_features)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_clip = learning_rate_clip

        if random_initial:
            unit_vector = np.random.randn(self.num_features)
            unit_vector /= norm(unit_vector)
            self.weights = unit_vector * 0.01
        else:
            self.weights = np.zeros(self.num_features)

    def update(self, gradient):
        self.weights += self.learning_rate * gradient
        if self.learning_rate > self.learning_rate_clip:
            self.learning_rate *= self.learning_rate_decay
        else:
            self.learning_rate = self.learning_rate_clip

    def assign_weights(self, weights):
        self.weights = weights

    def get_current_weights(self):
        return copy.copy(self.weights)

    def get_query_result_list(self, dataset, query):
        docid_list = dataset.get_candidate_docids_by_query(query)
        feature_matrix = dataset.get_all_features_by_query(query)

        score_list = self.get_scores(feature_matrix)

        docid_score_list = zip(docid_list, score_list)
        docid_score_list = sorted(docid_score_list, key=lambda x: x[1], reverse=True)

        query_result_list = []
        for i in range(0, len(docid_list)):
            (docid, socre) = docid_score_list[i]
            query_result_list.append(docid)
        return query_result_list

    def get_all_query_result_list(self, dataset):
        query_result_list = {}

        for query in dataset.get_all_querys():
            docid_list = np.array(dataset.get_candidate_docids_by_query(query))
            docid_list = docid_list.reshape((len(docid_list), 1))
            feature_matrix = dataset.get_all_features_by_query(query)
            score_list = self.get_scores(feature_matrix)

            docid_score_list = np.column_stack((docid_list, score_list))
            docid_score_list = np.flip(docid_score_list[docid_score_list[:, 1].argsort()], 0)

            query_result_list[query] = docid_score_list[:, 0]

        return query_result_list

    def get_scores(self, features):

        weights = np.array([self.weights])

        score = np.dot(features, weights.T)[:, 0]

        return score

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
