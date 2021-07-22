# experiments run for FPDGD

import itertools
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from data.LetorDataset import LetorDataset
from clickModel.click_simulate import CcmClickModel
from client.federated_optimize import train_uniform
from ranker.PDGDLinearRanker import PDGDLinearRanker
import os


def do_task(task):
    """
    Single task
    :param task:
    :return:
    """
    fold_id, click_model, e_s, ranker_id = task

    epsilon = e_s[0]
    sensitivity = e_s[1]
    params = common_params.copy()
    linear_ranker = PDGDLinearRanker(n_features, Learning_rate)
    if epsilon == 0 and sensitivity == 0:
        enable_noise = False
        linear_ranker.enable_noise_and_set_sensitivity(enable_noise, sensitivity)
    else:
        enable_noise = True
        linear_ranker.enable_noise_and_set_sensitivity(enable_noise, sensitivity)
    ranker_generator = linear_ranker
    params.update(
        dict(click_model=click_model,
             sensitivity=sensitivity,
             epsilon=epsilon,
             ranker_generator=ranker_generator,
             linear_ranker=linear_ranker,
             enable_noise=enable_noise
             ))

    trainset = LetorDataset("{}/Fold{}/train.txt".format(params['dataset_path'], fold_id + 1),
                            params['n_features'], query_level_norm=data_norm,
                            cache_root="../datasets/cache")
    testset = LetorDataset("{}/Fold{}/test.txt".format(params['dataset_path'], fold_id + 1),
                           params['n_features'], query_level_norm=data_norm,
                           cache_root="../datasets/cache")

    task_info = "click_model:{} folder:{}".format(click_model.name, fold_id + 1)
    train_result = train_uniform(params=params, traindata=trainset, testdata=testset, message=task_info, num_update=num_update)
    return train_result


def run(path, tasks):
    tasks = list(tasks)
    print("num tasks:", len(tasks))
    # multi-processing
    n_cpu = min(30, len(tasks))
    print("num cpus:", n_cpu)
    with Pool(n_cpu) as p:
        results = p.map(do_task, tasks)

    for task, result in zip(tasks, results):
        fold_id, click_model, e_s, ranker_id = task
        epsilon = e_s[0]
        sensitivity = e_s[1]
        click_model = click_model.name
        save_path = path + "fold_{}/{}/sensitivity_{}/epsilon_{}/result.npy".format(fold_id, click_model, sensitivity, epsilon)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, result)


if __name__ == "__main__":
    # dataset parameters
    datasets = ["MQ2007"]  # ["MQ2007", "MSLR10K"]
    experiment = "FPDGD" # ["FPDGD", "pdgd"]
    # experiment parameters
    n_clients = 10 # number of clients
    interactions_per_feedback = 4 # batch size
    interactions_budget = 40000 # total interactions budget
    Learning_rate = 0.1
    update = True #client_multi_update
    num_update = 1000 # number of global ranker update
    e_s_list = [(4.5, 5)] # e_s_list = [(1.2, 3), (2.3, 3), (4.5, 5), (10, 5)] (epsilon, sensitivity) pair used in experiments
    enable_noise = True  # set True if you want to add DP noise, otherwise set False

    dataset_root_dir = "../datasets"
    output_root_dir = "../results"

    for seed in range(1, 2):
        for dataset in datasets:
            if dataset == "MQ2007":
                n_folds = 5
                n_features = 46
                data_norm = False

            elif dataset == "MSLR10K":
                n_folds = 5
                n_features = 136
                data_norm = True

            dataset_path = f"{dataset_root_dir}/{dataset}"
            output_path = f"{output_root_dir}/{experiment}/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{interactions_per_feedback}_total{interactions_budget}"

            # click models
            if dataset == "MQ2007":
                PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 0.5, 2: 1.0},
                                              stop_relevance={0: 0.0, 1: 0.0, 2: 0.0},
                                              name="Perfect", depth=10)
                NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.5, 2: 0.95},
                                                   stop_relevance={0: 0.2, 1: 0.5, 2: 0.9},
                                                   name="Navigational", depth=10)
                INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.4, 1: 0.7, 2: 0.9},
                                                    stop_relevance={0: 0.1, 1: 0.3, 2: 0.5},
                                                    name="Informational", depth=10)
            elif dataset == "MSLR10K":
                PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                                              stop_relevance={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                              name="Perfect",
                                              depth=10)
                NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.95},
                                                   stop_relevance={0: 0.2, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9},
                                                   name="Navigational",
                                                   depth=10)
                INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.4, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9},
                                                    stop_relevance={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5},
                                                    name="Informational",
                                                    depth=10)
            # multi or single update on clients
            if update:
                output_path = output_path + "_multi/run_{}/".format(seed)
            elif not update:
                output_path = output_path + "_single/run_{}/".format(seed)

            common_params = dict(
                n_clients=n_clients,
                interactions_budget=interactions_budget,
                seed=seed,
                interactions_per_feedback=interactions_per_feedback,
                multi_update=update,
                n_features=n_features,
                dataset_path=dataset_path,
            )

            tasks = itertools.product(range(n_folds),
                                      [INFORMATIONAL_MODEL,
                                       NAVIGATIONAL_MODEL,
                                       PERFECT_MODEL
                                       ],
                                      e_s_list,
                                      range(1))
            run(output_path, tasks)