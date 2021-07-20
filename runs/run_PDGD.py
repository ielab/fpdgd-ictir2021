# experiments run for PDGD (single client)

import os
import sys
sys.path.append('../')
from data.LetorDataset import LetorDataset
from ranker.PDGDLinearRanker import PDGDLinearRanker
from clickModel.SDBN import SDBN
from clickModel.PBM import PBM
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle
from tqdm import tqdm


def run(train_set, test_set, ranker, num_interation, click_model, seed):
    # initialise
    offline_ndcg_list = []
    online_ndcg_list = []
    query_set = train_set.get_all_querys()
    np.random.seed(seed)
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0

    for i in index:
        num_iter += 1

        # one interaction (randomly choose a query from dataset)
        qid = query_set[i]
        result_list, scores = ranker.get_query_result_list(train_set, qid)
        clicked_doc, click_label, _ = click_model.simulate(qid, result_list, train_set)

        # online evaluation
        online_ndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10) # ndcg@k evaluation on training_set (use ture relevance label)
        online_ndcg_list.append(online_ndcg)

        # offline evaluation (to the current ranker)
        all_result = ranker.get_all_query_result_list(test_set)
        offline_ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10) # off-line ndcg evaluation on test_set of each batch
        offline_ndcg_list.append(offline_ndcg)

        # model update
        ranker.update_to_clicks(click_label, result_list, scores, train_set.get_all_features_by_query(qid))

        final_weights = ranker.get_current_weights()
    return offline_ndcg_list, online_ndcg_list, final_weights



def job(model_type, f, train_set, test_set, output_fold, pc, ps):
    cm = SDBN(pc, ps) # pc: click probability, ps: stop probability

    for seed in tqdm(range(1, 2)):
        ranker = PDGDLinearRanker(FEATURE_SIZE, Learning_rate)
        print("\n", "PDGD fold{} {} run{} start!".format(f, model_type, seed), "\n")
        offline_ndcg, online_ndcg, final_weights = run(train_set, test_set, ranker, NUM_INTERACTION, cm, seed)
        os.makedirs(os.path.dirname("{}/fold{}/".format(output_fold, f)),
                    exist_ok=True)  # create directory if not exist
        with open(
                "{}/fold{}/{}_run{}_offline_ndcg.txt".format(output_fold, f, model_type, seed),
                "wb") as fp:
            pickle.dump(offline_ndcg, fp)
        with open(
                "{}/fold{}/{}_run{}_online_ndcg.txt".format(output_fold, f, model_type, seed),
                "wb") as fp:
            pickle.dump(online_ndcg, fp)
        with open(
                "{}/fold{}/{}_run{}_weights.txt".format(output_fold, f, model_type, seed),
                "wb") as fp:
            pickle.dump(final_weights, fp)
        print("\n", "PDGD fold{} {} run{} finished!".format(f, model_type, seed), "\n")


if __name__ == "__main__":
    NUM_INTERACTION = 10000
    click_models = ["informational", "navigational", "perfect"]
    Learning_rate = 0.1

    datasets = ["MQ2007"] # ["MQ2007", "MSLR10K"]

    dataset_root_dir = "./datasets"
    output_root_dir = "./results"
    cache_path = "./datasets/cache"

    for dataset in datasets:
        output_fold = f"{output_root_dir}/PDGD/{dataset}/{dataset}_PDGD_total_interaction{NUM_INTERACTION}"
        processors = []
        for click_model in tqdm(click_models):
            # adding parameters based on different datasets and click_model
            # (feature_size, normalization, fold_range, clicking probability, stopping probability)
            if dataset == "MQ2007":
                FEATURE_SIZE = 46
                norm = False
                fold_range = range(1, 6)
                if click_model == "perfect":
                    pc = [0.0, 0.5, 1.0]
                    ps = [0.0, 0.0, 0.0]
                elif click_model == "navigational":
                    pc = [0.05, 0.5, 0.95]
                    ps = [0.2, 0.5, 0.9]
                elif click_model == "informational":
                    pc = [0.4, 0.7, 0.9]
                    ps = [0.1, 0.3, 0.5]
            elif dataset == "MSLR10K":
                FEATURE_SIZE = 136
                norm = True
                fold_range = range(1, 6)
                if click_model == "perfect":
                    pc = [0.0, 0.2, 0.4, 0.8, 1.0]
                    ps = [0.0, 0.0, 0.0, 0.0, 0.0]
                elif click_model == "navigational":
                    pc = [0.05, 0.3, 0.5, 0.7, 0.95]
                    ps = [0.2, 0.3, 0.5, 0.7, 0.9]
                elif click_model == "informational":
                    pc = [0.4, 0.6, 0.7, 0.8, 0.9]
                    ps = [0.1, 0.2, 0.3, 0.4, 0.5]

            for f in fold_range:
                train_path = f"{dataset_root_dir}/{dataset}/Fold{f}/train.txt"
                test_path = f"{dataset_root_dir}/{dataset}/Fold{f}/test.txt"
                train_set = LetorDataset(train_path, FEATURE_SIZE, query_level_norm=norm,
                                         cache_root=cache_path)
                test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=norm,
                                        cache_root=cache_path)

                print(dataset, click_model, f)
                p = mp.Process(target=job,
                               args=(click_model, f, train_set, test_set, output_fold, pc, ps))
                p.start()
                processors.append(p)
        for p in processors:
            p.join()

