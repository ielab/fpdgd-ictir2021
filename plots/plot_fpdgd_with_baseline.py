from matplotlib.pylab import plt
import numpy as np
import seaborn as sns
COLORS = ['b', 'g', 'r', 'm', 'y']


def cumulative_online_score(ys):
    cndcg = 0
    for i, score in enumerate(ys):
        cndcg += 0.9995 ** i * score
    return cndcg


def smoothen_trajectory(ys, group_size=10):
    return np.convolve(np.ones(group_size)/group_size, ys, "valid")


def load_data(file_name):
    result = np.load(file_name, allow_pickle=True)
    ranker, ndcg_server, mrr_server, ndcg_client, mrr_client = result
    return ndcg_server, mrr_server, ndcg_client, mrr_client


def draw_line(ys, label, ax, color, style, do_smooth=False):
    ys = np.stack(ys)
    ys_mean = np.mean(ys, axis=0)
    ys_std = np.std(ys, axis=0)
    ys_low = np.subtract(ys_mean, ys_std)
    ys_high = np.add(ys_mean, ys_std)

    xs = np.array(range(np.shape(ys_mean)[0]))

    if do_smooth:
        ax.plot(smoothen_trajectory(xs), smoothen_trajectory(ys_mean), color=color, linestyle=style, label=label)
        ax.fill_between(smoothen_trajectory(xs), smoothen_trajectory(ys_low), smoothen_trajectory(ys_high), color=color, linestyle=style, alpha=0.1)
    else:
        ax.plot(xs, ys_mean, color=color, linestyle=style, label=label)
        ax.fill_between(xs, ys_low, ys_high, color=color, linestyle=style, alpha=0.1)
    return cumulative_online_score(ys_mean)


# Against baseline
dataset = "MQ2007" #"MSLR10K"
do_smooth = False
metric = 'fndcg'

click_models = ['Perfect', 'Navigational', 'Informational']
e_s_list = [(1.2, 3), (2.3, 3), (4.5, 5), (10, 5)]
p_list = [0.25, 0.5, 0.9, 1.0]
runs = range(1, 2)
n_fold = 5

n_clients = 1000
client_batch_size = 4
total_interactions = 4000000
n_iterations = total_interactions // n_clients // client_batch_size
save_path = f'{dataset}_{metric}.png'


if metric == 'fndcg': # offline ndcg
    metric_ind = 0
elif metric =='fmrr': # offline mrr
    metric_ind = 1
elif metric == 'ondcg': # online ndcg
    metric_ind = 2
elif metric == 'omrr': # online mrr
    metric_ind = 3

# sns.set(style="darkgrid")
plt.close('all')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
f, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(40, 8))
f.tight_layout()

m = client_batch_size * n_clients
mid = 0
for click_model in click_models:
    a = ax[mid]
    for i in range(len(e_s_list)):
        epsilon, sensitivity = e_s_list[i]
        p = p_list[i]
        baseline_ys_list = []
        fpdgd_ys_list = []
        for run in runs:
            for f in range(n_fold):

                baseline_file_name = f"results/FPDGD/{dataset}/{dataset}_foltr_linear_clients{n_clients}_batch{client_batch_size}_total{total_interactions}" \
                            f"/run_{run}/fold_{f}/{click_model}/p_{p}/result.npy"
                fpdgd_file_name = f"results/FPDGD/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{client_batch_size}_total{total_interactions}_multi" \
                            f"/run_{run}/fold_{f}/{click_model}/sensitivity_{sensitivity}/epsilon_{epsilon}/result.npy"

                baseline_ys_list.append(load_data(baseline_file_name)[metric_ind])
                fpdgd_ys_list.append(load_data(fpdgd_file_name)[metric_ind])

        baseline_score = draw_line(baseline_ys_list, f"FOLtR-ES:p={p}", a, COLORS[i], '--', do_smooth)
        fpdgd_score = draw_line(fpdgd_ys_list, f"FPDGD:\u03B5={epsilon}", a, COLORS[i], '-', do_smooth)
        print("epsilon:", epsilon,
              "click_model:", click_model,
              "baseline:", baseline_score,
              'FPDGD:', fpdgd_score,)

    fpdgd_ys_list = []
    fpdgd_budget_ys_list = []
    for run in runs:
        for f in range(n_fold):
            pdgd_file_name = f"results/pdgd/{dataset}/{dataset}_FPDGD_clients{1}_batch{1}_total{1000}_multi" \
                             f"/run_{run}/fold_{f}/{click_model}/sensitivity_{0}/epsilon_{0}/result.npy"
            pdgd_budget_file_name = f"results/pdgd/{dataset}/{dataset}_FPDGD_clients{1}_batch{1}_total{4000000}_multi" \
                             f"/run_{run}/fold_{f}/{click_model}/sensitivity_{0}/epsilon_{0}/result.npy"
            fpdgd_ys_list.append(load_data(pdgd_file_name)[metric_ind])
            fpdgd_budget_ys_list.append(load_data(pdgd_budget_file_name)[metric_ind])

    a2 = a.twinx()
    draw_line(fpdgd_ys_list, f"PDGD, fixed num_updates", a2, 'black', '--', do_smooth=do_smooth)
    draw_line(fpdgd_budget_ys_list, f"PDGD, fixed budget", a2, 'black', '-', do_smooth=do_smooth)
    if mid == 0:
        a2.legend(loc='upper left', fontsize=17)
    a2.set_ylim([0.16, 0.50])
    a2.get_yaxis().set_visible(False)

    a.set_ylim([0.45, 0.85])
    # a.set_ylim([0.16, 0.50])
    # a.set_xlim([0, 200])
    a.set_xlim([0, 1000])
    a.set_title(f"{click_model}", fontsize=26)
    ax[0].set_ylabel("Mean batch offline nDCG@10", fontsize=26)
    # ax[0].set_ylabel("Mean batch offline MaxRR", fontsize=26)
    ax[mid].set_xlabel("Number of updates", fontsize=26)
    ax[0].legend(loc='lower left', ncol=4, fontsize=17)
    mid += 1

# ax[1].set_xlabel("{}".format(dataset))
# plt.savefig(save_path, bbox_inches='tight')