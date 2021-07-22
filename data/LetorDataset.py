from data.AbstractDataset import AbstractDataset
import numpy as np
import math
import os
import hashlib
import pickle


class LetorDataset(AbstractDataset):

    def __init__(self,
                 path,
                 feature_size,
                 query_level_norm=False,
                 binary_label=0,
                 cache_root=None
                 ):
        super().__init__(path, feature_size, query_level_norm)
        self._binary_label = binary_label
        self._comments = {}
        self._docid_map = {}
        # self._docstr_map = {}

        if cache_root is not None:
            new = self.from_path(path, cache_root)
            if new is not None:
                self.__dict__.update(new.__dict__)
            else:
                self._load_data()
                cache_name = f"{hashlib.md5(path.encode()).hexdigest()}.pkl"
                file_path = f'./{cache_root}/{cache_name}'

                with open(file_path, 'wb') as f:
                    pickle.dump(self, f)
                print(f"Cached the array to {file_path}")
        else:
            self._load_data()

    def from_path(self, path: str, cache_root: str) -> 'LetorDataset':
        """
        Constructs a dataset by reading form a disk
        :param root_path: A path to the root that contains (Fold1, Fold2, ...)
        :param cache_root: None if no caching is needed, otherwise a path to the cache dir;
                        if the cache dir already contains the data it will be used. Hence, cleaning
                        of the cache has to be done manually if needed.
        :return: Constructed Dataset instance
        """

        cache_name = f"{hashlib.md5(path.encode()).hexdigest()}.pkl"
        file_path = f'./{cache_root}/{cache_name}'
        if os.path.exists(file_path):
            print(f"Loading from cache file {file_path}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            print("no cache found for", path)
            return None

    def _load_data(self):
        print("Loading {}......".format(self._path))
        with open(self._path, "r") as fin:
            current_query = None
            for line in fin:
                cols = line.strip().split()
                query = cols[1].split(':')[1]
                if query == current_query:
                    docid = len(self._query_get_docids[query])
                    old_query = True

                else:
                    if current_query != None and self._query_level_norm:
                        self._normalise(current_query)
                    old_query = False
                    docid = 0
                    current_query = query
                    self._docid_map[query] = {}
                    # self._docstr_map[query] = {}
                    self._query_pos_docids[query] = []

                comments_part = line.split("#")
                if len(comments_part) == 2:
                    if query not in self._comments:
                        self._comments[query] = []
                    self._comments[query].append(comments_part[1].strip())

                relevence = float(cols[0])  # Sometimes the relevance label can be a float.
                if relevence.is_integer():
                    relevence = int(relevence)  # But if it is indeed an int, cast it into one.
                if self._binary_label != 0:
                    if relevence >= self._binary_label:
                        relevence = 1
                    else:
                        relevence = 0

                features = [0] * self._feature_size

                for i in range(2, len(cols)):
                    feature_id = cols[i].split(':')[0]

                    if not feature_id.isdigit():
                        if feature_id[0] == "#":
                            self._docid_map[query][docid] = cols[i][1:]
                            # self._docstr_map[query][cols[i][1:]] = docid
                        break

                    feature_id = int(feature_id) - 1
                    feature_value = float(cols[i].split(':')[1])
                    if math.isnan(feature_value):
                        feature_value = 0

                    features[feature_id] = feature_value

                if relevence > 0:
                    self._query_pos_docids[query].append(docid)

                if old_query:
                    self._query_docid_get_features[query][docid] = np.array(features)
                    self._query_get_docids[query].append(docid)
                    self._query_get_all_features[query] = np.vstack((self._query_get_all_features[query], features))
                    self._query_docid_get_rel[query][docid] = relevence
                    self._query_relevant_labels[query].append(relevence)
                else:
                    self._query_docid_get_features[query] = {docid: np.array(features)}
                    self._query_get_docids[query] = [docid]
                    self._query_get_all_features[query] = np.array([features])
                    self._query_docid_get_rel[query] = {docid: relevence}
                    self._query_relevant_labels[query] = [relevence]

        if self._query_level_norm:
            self._normalise(current_query)

    def _normalise(self, query):
        norm = np.zeros((len(self._query_get_docids[query]), self._feature_size))
        # if there is more than 1 candidate docs, do the norm #TODO - fix the bug
        if norm.shape[0] != 1:
            query_features = self._query_get_all_features[query]
            min = np.amin(query_features, axis=0)
            max = np.amax(query_features, axis=0)
            safe_ind = max - min != 0
            norm[:, safe_ind] = (query_features[:, safe_ind] - min[safe_ind]) / (
                    max[safe_ind] - min[safe_ind])
            self._query_get_all_features[query] = norm

    def update_relevance_label(self, qrel_dic: dict):
        for qid in self._query_docid_get_rel.keys():

            self._query_pos_docids[qid] = []
            ind = 0
            for docid in self._query_docid_get_rel[qid].keys():
                if self._docid_map[qid][docid] in qrel_dic[qid].keys():
                    rel = qrel_dic[qid][self._docid_map[qid][docid]]
                else:
                    rel = 0
                self._query_docid_get_rel[qid][docid] = rel
                self._query_relevant_labels[qid][ind] = rel

                if rel > 0:
                    self._query_pos_docids[qid].append(docid)
                ind += 1

        # self._query_get_all_features_temp = {}
        # for qid in self._query_docid_get_rel.keys():
        #     self._query_pos_docids[qid] = []
        #     self._query_get_docids[qid] = []
        #     self._query_get_all_features_temp[qid] = []
        #     for docstr in qrel_dic[qid].keys():
        #         docid = self._docstr_map[qid][docstr]
        #         rel = qrel_dic[qid][docstr]
        #
        #         self._query_docid_get_rel[qid][docid] = rel
        #         self._query_relevant_labels[qid][docid] = rel
        #         self._query_get_docids[qid].append(docid)
        #         features = self._query_get_all_features[qid][docid]
        #         self._query_get_all_features_temp[qid].append(features)
        #         if rel > 0:
        #             self._query_pos_docids[qid].append(docid)
        #     self._query_get_all_features[qid] = np.array(self._query_get_all_features_temp[qid])

    def update_relevance_by_qrel(self, path: str):

        # q-d pair dictionary
        qrel_dic = {}

        with open(path, 'r') as f:
            for line in f:
                qid, _, docid, rel = line.strip().split()
                if qid in qrel_dic.keys():
                    qrel_dic[qid][docid] = int(rel)
                else:
                    qrel_dic[qid] = {docid: int(rel)}
        self.update_relevance_label(qrel_dic)

    def get_features_by_query_and_docid(self, query, docid):
        return self._query_docid_get_features[query][docid]

    def get_candidate_docids_by_query(self, query):
        return self._query_get_docids[query]

    def get_all_features_by_query(self, query):
        return self._query_get_all_features[query]

    def get_relevance_label_by_query_and_docid(self, query, docid):
        return self._query_docid_get_rel[query][docid]

    def get_all_relevance_label_by_query(self, query):
        return self._query_relevant_labels[query]

    def get_relevance_docids_by_query(self, query):
        return self._query_pos_docids[query]

    def get_all_querys(self):
        return np.array(list(self._query_get_all_features.keys()))

    def get_all_comments_by_query(self, query):
        return self._comments[query]

    def write(self, output_file):
        s = ""
        for query in self.get_all_querys():
            comments = self.get_all_comments_by_query(query)
            for i, features in enumerate(self.get_all_features_by_query(query)):
                comment = comments[i]
                label = self.get_relevance_label_by_query_and_docid(query, i)
                features_str = ""
                for i, feature in enumerate(features):
                    # if feature == 0:
                    #     continue
                    features_str += "{}:{} ".format(i + 1, feature)
                s += "{} qid:{} {}#{}\n".format(label, query, features_str, comment)
        with open(output_file, "w") as f:
            f.write(s)

    def write_by_queries(self, output_file, queries):
        s = ""
        for query in queries:
            comments = self.get_all_comments_by_query(query)
            for i, features in enumerate(self.get_all_features_by_query(query)):
                comment = comments[i]
                label = self.get_relevance_label_by_query_and_docid(query, i)
                features_str = ""
                for i, feature in enumerate(features):
                    # if feature == 0:
                    #     continue
                    features_str += "{}:{} ".format(i + 1, feature)
                s += "{} qid:{} {}#{}\n".format(label, query, features_str, comment)
        with open(output_file, "w") as f:
            f.write(s)

    def write_cross_validation_datasets(self, path: str, fold_num: int):
        """
        :param fold_num: number of fold to do cross validation.
        :param path: folder address to store the cross sets.
        :return:
        """

        for fold in range(fold_num):
            fold_path = "{}/fold{}".format(path, fold + 1)
            # Create target Directory if don't exist
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
                print("Directory ", fold_path, " Created ")
            else:
                print("Directory ", fold_path, " already exists")

        all_queries = self.get_all_querys()

        np.random.shuffle(all_queries)

        query_chunks = np.array_split(all_queries, fold_num)

        for i in range(fold_num):
            test_path = "{}/Fold{}/test.txt".format(path, i + 1)
            train_path = "{}/Fold{}/train.txt".format(path, i + 1)
            test_queries = query_chunks[i]
            train_chunks = query_chunks[:i]
            train_chunks.extend(query_chunks[i + 1:])
            train_queries = np.concatenate(train_chunks)

            self.write_by_queries(test_path, test_queries)
            self.write_by_queries(train_path, train_queries)

    @staticmethod
    def runs_to_letor(input_folder: str, output_folder: str):
        """
        Convert run files into LTR dataset.
        :param input_folder: folder path that contains all run files.
        :param output_folder:
        :return:
        """
        files = os.listdir(input_folder)
        num_feature = len(files)

        # q-d pair dictionary
        query_dic = {}
        for feature_id in range(num_feature):
            # feature id in standard letor datasets start from 1.
            with open(os.path.join(input_folder, files[feature_id]), 'r') as f:
                for line in f:
                    qid, _, docid, rank, score, rname = line.strip().split()
                    if qid in query_dic.keys():
                        if docid in query_dic[qid].keys():
                            query_dic[qid][docid].append((feature_id + 1, score))
                        else:
                            query_dic[qid][docid] = [(feature_id + 1, score)]
                    else:
                        query_dic[qid] = {docid: [(feature_id + 1, score)]}
        s = ""
        for qid in query_dic.keys():
            for docid in query_dic[qid].keys():
                # the first column is relevance label, dont know for now.
                s += "0 "
                s += "qid:{} ".format(qid)

                for feature_id, socre in query_dic[qid][docid]:
                    s += "{}:{} ".format(feature_id, socre)
                s += "#docid = {}\n".format(docid)
        with open(output_folder + "letor.txt", "w") as f:
            f.write(s)

        s = ""
        for fid in range(len(files)):
            s += "{}:{}\n".format(fid + 1, files[fid])
        with open(output_folder + "feature_description.txt", "w") as f:
            f.write(s)

    # bad implementation only for PMGD:
    def get_query_docid_get_feature(self):
        return self._query_docid_get_features

    def get_query_get_all_features(self):
        return self._query_get_all_features

    def get_query_get_docids(self):
        return self._query_get_docids