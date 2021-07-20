class AbstractRanker:
    def __init__(self, num_features):
        self.num_features = num_features

    def update(self, gradient):
        raise NotImplementedError("Derived class needs to implement "
                                  "update.")

    def assign_weights(self, weights):
        raise NotImplementedError("Derived class needs to implement "
                                  "assign_weights.")

    def get_current_weights(self):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_current_weights.")

    def get_query_result_list(self, dataset, query):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_query_result_list.")

    def get_all_query_result_list(self, dataset):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_all_query_result_list.")

    def get_scores(self, features):
        raise NotImplementedError("Derived class needs to implement "
                                  "features.")