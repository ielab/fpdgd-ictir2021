import numpy as np
from clickModel.CM import CM
import copy

class SDBN(CM):

    def __init__(self, pc=None, ps=None, alpha=1, beta=1):
        self.name = 'SDBN'
        self.parameter_dict = {}
        self.stat_dict = {}
        self.alpha = alpha
        self.beta = beta
        self.pc = pc
        self.ps = ps

    def set_probs(self, pc, ps):
        self.pc = pc
        self.ps = ps

    def simulate(self, query, result_list, dataset):
        clicked_doc = []
        click_label = np.zeros(len(result_list))
        satisfied = False
        for i in range(0, len(result_list)):
            click_prob = np.random.rand()
            stop_prob = np.random.rand()
            docid = result_list[i]

            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)

            if click_prob < self.pc[relevance]:
                click_label[i] = 1
                clicked_doc.append(result_list[i])
                if stop_prob < self.ps[relevance]:
                    satisfied = True
                    break

        return clicked_doc, click_label, satisfied

    def online_training(self, qid, result_list, clicks):
        lastClickRank = np.where(clicks == 1)[0][-1] + 1
        if qid not in self.stat_dict.keys():
            self.stat_dict[qid] = {}
        doc_stat = self.stat_dict[qid]
        for rank in range(lastClickRank):
            docID = result_list[rank]
            if docID not in doc_stat.keys():
                doc_stat[docID] = (0, 0, 0)
            exam = doc_stat[docID][0] + 1
            c = doc_stat[docID][1]
            lc = doc_stat[docID][2]
            if clicks[rank] == 1:
                c += 1
                if rank == lastClickRank - 1:
                    lc += 1
            doc_stat[docID] = (exam, c, lc)

    def click_noise_reduce(self, qid, result_list, clicks, threshold, num_exam):
        reduce = False
        reduced_index = []
        for rank in range(len(result_list)):
            docID = result_list[rank]
            if qid in self.stat_dict.keys():
                if docID in self.stat_dict[qid].keys():
                    # print(self.stat_dict[qid][docID], clicks[rank])
                    if clicks[rank] == 1 and self.stat_dict[qid][docID][0] >= num_exam:
                        # print(self.stat_dict[qid][docID][1] / self.stat_dict[qid][docID][0])
                        if self.stat_dict[qid][docID][1]/self.stat_dict[qid][docID][0] <= threshold:
                            clicks[rank] = 0
                            reduced_index.append(rank)
                            # print("reduce for ", qid, docID, self.stat_dict[qid][docID][0])
                            reduce = True
        return reduce, reduced_index



    def train(self, click_log):
        self._get_train_stat(click_log)
        print("{} training.......".format(self.name))
        for qid in self.stat_dict.keys():
            self.parameter_dict[qid] = {}
            for docID in self.stat_dict[qid].keys():
                a = (self.stat_dict[qid][docID][1] + self.alpha) / (self.stat_dict[qid][docID][0] + self.alpha + self.beta)
                s = (self.stat_dict[qid][docID][2] + self.alpha) / (self.stat_dict[qid][docID][1] + self.alpha + self.beta)
                self.parameter_dict[qid][docID] = (a, s)

    def _get_train_stat(self, click_log):
        print("{} processing log.......".format(self.name))
        dataset_size = click_log.shape[0]
        for line in range(dataset_size):

            qid = click_log[line][0]
            docIds = click_log[line][1:11]
            clicks = click_log[line][11:21]

            if qid not in self.stat_dict.keys():
                self.stat_dict[qid] = {}

            doc_stat = self.stat_dict[qid]

            if np.where(clicks == '1')[0].size == 0:
                continue

            lastClickRank = np.where(clicks == '1')[0][-1] + 1

            for rank in range(lastClickRank):
                docID = docIds[rank]
                if docID not in doc_stat.keys():
                    doc_stat[docID] = (0, 0, 0)
                exam = doc_stat[docID][0] + 1
                c = doc_stat[docID][1]
                lc = doc_stat[docID][2]
                if clicks[rank] == '1':
                    c += 1
                    if rank == lastClickRank - 1:
                        lc += 1
                doc_stat[docID] = (exam, c, lc)

            # if line % 10000 == 0:
            #     print("process %d/%d of dataset" % (line, dataset_size))

    def get_click_probs(self, session):
        qid = session[0]
        docIds = session[1:11]
        a_probs = np.zeros(10)
        exam_probs = np.zeros(10)
        exam_probs[0] = 1
        unseen_docs_index = []
        for i in range(1, 10):
            if docIds[i - 1] not in self.parameter_dict[qid].keys():
                ar = self.alpha / (self.alpha + self.beta)
                sr = self.alpha / (self.alpha + self.beta)
            else:
                ar = self.parameter_dict[qid][docIds[i - 1]][0]
                sr = self.parameter_dict[qid][docIds[i - 1]][1]
            exam_probs[i] = exam_probs[i - 1] * (ar * (1 - sr) + (1 - ar))

        for i in range(10):
            if docIds[i] not in self.parameter_dict[qid].keys():
                unseen_docs_index.append(i)
                a = self.alpha / (self.alpha + self.beta)
            else:
                a = self.parameter_dict[qid][docIds[i]][0]
            a_probs[i] = a

        probs = np.multiply(exam_probs, a_probs)

        return probs

    def get_real_click_probs(self, session, dataset):
        qid = session[0]
        docIds = session[1:11]
        a_probs = np.zeros(10)
        exam_probs = np.zeros(10)
        exam_probs[0] = 1

        for i in range(1, 10):
            relevance = dataset.get_relevance_label_by_query_and_docid(qid, int(docIds[i - 1]))
            ar = self.pc[relevance]
            sr = self.ps[relevance]
            exam_probs[i] = exam_probs[i - 1] * (ar * (1 - sr) + (1 - ar))

        for i in range(10):
            relevance = dataset.get_relevance_label_by_query_and_docid(qid, int(docIds[i]))
            a = self.pc[relevance]
            a_probs[i] = a

        return np.multiply(exam_probs, a_probs)

