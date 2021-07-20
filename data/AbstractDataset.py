class AbstractDataset:

    def __init__(self,
                 path,
                 feature_size,
                 query_level_norm=False):
        self._path = path
        self._feature_size = feature_size
        self._query_docid_get_features = {}
        self._query_get_docids = {}
        self._query_get_all_features = {}
        self._query_docid_get_rel = {}
        self._query_pos_docids = {}
        self._query_relevant_labels = {}
        self._query_level_norm = query_level_norm

    def _load_data(self):
        raise NotImplementedError("Derived class needs to implement "
                                  "_load_data.")

    def get_features_by_query_and_docid(self, query, docid):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_features_by_query_and_docid.")

    def get_candidate_docids_by_query(self, query):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_candidate_docids_by_query.")

    def get_all_features_by_query(self, query):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_all_features_by_query.")

    def get_relevance_label_by_query_and_docid(self, query, docid):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_relevance_by_query_and_docid.")


    def get_relevance_docids_by_query(self, query):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_relevance_docids_by_query.")

    def get_all_querys(self):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_all_querys.")
