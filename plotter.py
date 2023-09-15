import os
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import numpy as np

pareto_2d_path = 'integration_pareto_2d'
pareto_3d_path = 'pareto_3d'
pareto_3d_best = 'integration_pareto_3d'

biases = {'chp': 0, 'chp_gh': 4, 'chp_smt': 10, 'fst': 17, 'fst_gh': 21, 'fst_smt': 27}


def draw_scatter(ax,
                 x,
                 y,
                 z,
                 c,
                 label):
    ax.scatter(x,
               y,
               z,
               s=10,
               c=c,
               label=label)


def pareto_frontier_multi(dat):
    dat_len = len(dat)
    dominated = np.zeros(dat_len, dtype=bool)
    avail_dat = dat[:, :-1]
    for i in range(dat_len):
        for j in range(dat_len):
            if all(avail_dat[j] <= avail_dat[i]) and any(avail_dat[j] < avail_dat[i]):
                dominated[i] = True
                break
    return dat[~dominated]


def draw_pareto_3d(modu):
    modu_map = {'incl': 'ext_dat/incl_cost/',
                'excl': 'ext_dat/excl_cost/',
                'bf': 'bruteforce_dat/'}
    dat_path = modu_map[modu]
    runtime_idx = 5
    price_idx = 6
    max_fr_idx = 8
    csvs = os.listdir(dat_path)
    for csv in csvs:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()
        proj_name = csv.replace('.csv', '')
        df = pd.read_csv(dat_path + csv)
        # (runtime, price, max failure rate)
        tup_dat = []
        for _, itm in df.iterrows():
            if np.isnan(itm[max_fr_idx]):
                continue
            tup = (itm[runtime_idx], itm[price_idx], itm[max_fr_idx])
            tup_dat.append(tup)
        xyz = np.array(tup_dat)
        frontiers = pareto_frontier_multi(xyz)
        x_front = frontiers[:, 0]
        y_front = frontiers[:, 1]
        z_front = frontiers[:, 2]
        for node in frontiers:
            tup_dat.remove(tuple(node))
        xyz_non = np.array(tup_dat)
        x_non = xyz_non[:, 0]
        y_non = xyz_non[:, 1]
        z_non = xyz_non[:, 2]
        draw_scatter(ax,
                     x_front,
                     y_front,
                     z_front,
                     'r',
                     f'Pareto frontiers: {len(frontiers)}')
        draw_scatter(ax,
                     x_non,
                     y_non,
                     z_non,
                     'royalblue',
                     f'Normal: {len(xyz_non)}')
        ax.set_title(textwrap.fill(proj_name),
                     fontsize=14)
        ax.set_xlabel('Runtime')
        ax.set_ylabel('Price')
        ax.set_zlabel('Max failure rate')
        ax.legend()
        plt.savefig(f'{pareto_3d_path}/{proj_name}.png',
                    dpi=1000)
        plt.close()


def draw_integration_2d(modu,
                        dat_name,
                        fr):
    subdir = f'{pareto_2d_path}/{modu}_failure_rate_{fr}'
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    proj_idx = 0
    colors = ['mediumpurple', 'thistle', 'lightsteelblue', 'darkseagreen', 'slategrey', 'darkkhaki']
    nodes = {'chp': 'Cheapest', 'chp_gh': 'Cheapest github_caliber', 'chp_smt': 'Cheapest smart baseline',
             'fst': 'Fastest', 'fst_gh': 'Fastest github caliber', 'fst_smt': 'Fastest smart baseline'}
    runtime_idx = 3
    price_idx = 4
    max_fr_idx = 5
    dat = pd.read_csv(dat_name)
    for _, row in dat.iterrows():
        proj_name = row[proj_idx]
        plt.title(textwrap.fill(proj_name),
                  fontsize=14)
        if np.isnan(row[max_fr_idx]):
            plt.savefig(f'{subdir}/{proj_name}.png',
                        dpi=1000)
            plt.close()
            continue
        cnt = 0
        xy_fr_map = {}
        x_runtime = []
        y_price = []
        max_frs = []
        labels = []
        for key, bias in biases.items():
            xi = row[runtime_idx + bias]
            yi = row[price_idx + bias]
            x_runtime.append(xi)
            y_price.append(yi)
            max_frs.append(row[max_fr_idx + bias])
            labels.append(nodes[key])
            xy_fr_map[(xi, yi)] = cnt
            cnt += 1
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.scatter(x_runtime,
                    y_price,
                    s=10,
                    c=colors)
        plt.xlabel('Runtime')
        plt.ylabel('Price')
        plt.legend(handles=[plt.scatter([], [], c=c) for c in colors],
                   labels=labels)
        plt.savefig(f'{subdir}/{proj_name}.png',
                    dpi=1000)
        plt.close()


def draw_integration_3d(modu):
    modu_map = {'incl': 'integration_dat_incl_cost/',
                'excl': 'integration_dat_excl_cost/',
                'bf': 'integration_dat_bruteforce/'}
    dat_path = modu_map[modu]
    rec_degree_df = pd.DataFrame(None,
                                 columns=[
                                     'project',
                                     'ga_number',
                                     'ga_pareto_front',
                                     'github_caliber_number',
                                     'github_caliber_pareto_front',
                                     'smart_baseline_number',
                                     'smart_baseline_pareto_front',
                                     'total_pareto_front_number',
                                     'ga_rate',
                                     'github_caliber_rate',
                                     'smart_baseline_rate'
                                 ])
    proj_idx = 0
    runtime_idx = 3
    price_idx = 4
    max_fr_idx = 5
    csvs = os.listdir(dat_path)[1:]
    dfs = [pd.read_csv(dat_path + csv) for csv in csvs]
    df_len = len(dfs[0])
    for i in range(df_len):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()
        proj_name = dfs[0].iloc[i, proj_idx]
        ga_xyz = []
        github_xyz = []
        smart_xyz = []
        # 0: ga; 1: github; 2: smart
        for df in dfs:
            if np.isnan(df.iloc[i, max_fr_idx]):
                continue
            ga_xyz.append((df.iloc[i, runtime_idx],
                           df.iloc[i, price_idx],
                           df.iloc[i, max_fr_idx],
                           0))
            ga_xyz.append((df.iloc[i, runtime_idx + biases['fst']],
                           df.iloc[i, price_idx + biases['fst']],
                           df.iloc[i, max_fr_idx + biases['fst']],
                           0))
            if not np.isnan(df.iloc[i, max_fr_idx + biases['chp_gh']]):
                github_xyz.append((df.iloc[i, runtime_idx + biases['chp_gh']],
                                   df.iloc[i, price_idx + biases['chp_gh']],
                                   df.iloc[i, max_fr_idx + biases['chp_gh']],
                                   1))
            if not np.isnan(df.iloc[i, max_fr_idx + biases['fst_gh']]):
                github_xyz.append((df.iloc[i, runtime_idx + biases['fst_gh']],
                                   df.iloc[i, price_idx + biases['fst_gh']],
                                   df.iloc[i, max_fr_idx + biases['fst_gh']],
                                   1))
            if not np.isnan(df.iloc[i, max_fr_idx + biases['chp_smt']]):
                smart_xyz.append((df.iloc[i, runtime_idx + biases['chp_smt']],
                                  df.iloc[i, price_idx + biases['chp_smt']],
                                  df.iloc[i, max_fr_idx + biases['chp_smt']],
                                  2))
            if not np.isnan(df.iloc[i, max_fr_idx + biases['fst_smt']]):
                smart_xyz.append((df.iloc[i, runtime_idx + biases['fst_smt']],
                                  df.iloc[i, price_idx + biases['fst_smt']],
                                  df.iloc[i, max_fr_idx + biases['fst_smt']],
                                  2))
        xyz = ga_xyz + github_xyz + smart_xyz
        frontiers = pareto_frontier_multi(np.array(xyz))
        ga_frontiers = []
        github_frontiers = []
        smart_frontiers = []
        for itm in frontiers:
            tup = tuple(itm)
            if tup[3] == 0:
                ga_xyz.remove(tup)
                ga_frontiers.append(tup)
            elif tup[3] == 1:
                github_xyz.remove(tup)
                github_frontiers.append(tup)
            else:
                smart_xyz.remove(tup)
                smart_frontiers.append(tup)
        ga_frontiers = np.array(ga_frontiers) if len(ga_frontiers) > 0 else np.empty((0, 4))
        github_frontiers = np.array(github_frontiers) if len(github_frontiers) > 0 else np.empty((0, 4))
        smart_frontiers = np.array(smart_frontiers) if len(smart_frontiers) > 0 else np.empty((0, 4))
        ga_non = np.array(ga_xyz) if len(ga_xyz) > 0 else np.empty((0, 4))
        github_non = np.array(github_xyz) if len(github_xyz) > 0 else np.empty((0, 4))
        smart_non = np.array(smart_xyz) if len(smart_xyz) > 0 else np.empty((0, 4))
        frontiers_num = len(ga_frontiers) + len(github_frontiers) + len(smart_frontiers)
        rec_degree_df.loc[len(rec_degree_df.index)] = [
            proj_name,
            len(ga_non) + len(ga_frontiers),
            len(ga_frontiers),
            len(github_non) + len(github_frontiers),
            len(github_frontiers),
            len(smart_non) + len(smart_frontiers),
            len(smart_frontiers),
            frontiers_num,
            len(ga_frontiers) / frontiers_num,
            len(github_frontiers) / frontiers_num,
            len(smart_frontiers) / frontiers_num
        ]
        draw_scatter(ax,
                     ga_frontiers[:, 0],
                     ga_frontiers[:, 1],
                     ga_frontiers[:, 2],
                     'r',
                     f'GA pareto frontiers: {len(ga_frontiers)}')
        draw_scatter(ax,
                     ga_non[:, 0],
                     ga_non[:, 1],
                     ga_non[:, 2],
                     'lightcoral',
                     f'GA normal: {len(ga_non)}')
        draw_scatter(ax,
                     github_frontiers[:, 0],
                     github_frontiers[:, 1],
                     github_frontiers[:, 2],
                     'gold',
                     f'Github caliber pareto frontiers: {len(github_frontiers)}')
        draw_scatter(ax,
                     github_non[:, 0],
                     github_non[:, 1],
                     github_non[:, 2],
                     'khaki',
                     f'Github caliber normal: {len(github_non)}')
        draw_scatter(ax,
                     smart_frontiers[:, 0],
                     smart_frontiers[:, 1],
                     smart_frontiers[:, 2],
                     'deepskyblue',
                     f'Smart baseline pareto frontiers: {len(smart_frontiers)}')
        draw_scatter(ax,
                     smart_non[:, 0],
                     smart_non[:, 1],
                     smart_non[:, 2],
                     'lightblue',
                     f'Smart baseline normal: {len(smart_non)}')
        ax.set_title(textwrap.fill(proj_name),
                     fontsize=14)
        ax.set_xlabel('Runtime')
        ax.set_ylabel('Price')
        ax.set_zlabel('Max failure rate')
        ax.legend()
        plt.savefig(f'{pareto_3d_best}/{proj_name}.png',
                    dpi=1000)
        plt.close()
    rec_degree_df.to_csv(f'{pareto_3d_best}/dat_summary_result.csv', sep=',', header=True, index=False)


def draw_baseline_line_graph():
    baseline_path = 'integration_dat_incl_cost/'
    baseline_line_graph_path = 'baseline_line_graph/'
    if not os.path.exists(baseline_line_graph_path):
        os.mkdir(baseline_line_graph_path)
    csvs = os.listdir(baseline_path)[1:]
    dfs = [pd.read_csv(f'{baseline_path}{csv}') for csv in csvs]
    proj_names = dfs[0].iloc[:, 0]
    bias = 17
    proj_idx = 0
    chp_github_runtime_idx = 7
    chp_github_price_idx = 8
    chp_github_max_fr_idx = 9
    chp_smart_runtime_idx = 13
    chp_smart_price_idx = 14
    chp_smart_max_fr_idx = 15
    for i, proj in proj_names.items():
        chp_github_tup = []
        chp_smart_tup = []
        fst_github_tup = []
        fst_smart_tup = []
        for df in dfs:
            if not np.isnan(df.iloc[i, chp_github_max_fr_idx]):
                chp_github_tup.append((df.iloc[i, chp_github_price_idx],
                                       df.iloc[i, chp_github_max_fr_idx]))
            if not np.isnan(df.iloc[i, chp_smart_max_fr_idx]):
                chp_smart_tup.append((df.iloc[i, chp_smart_price_idx],
                                      df.iloc[i, chp_smart_max_fr_idx]))
            if not np.isnan(df.iloc[i, chp_github_max_fr_idx + bias]):
                fst_github_tup.append((df.iloc[i, chp_github_runtime_idx + bias],
                                       df.iloc[i, chp_github_max_fr_idx + bias]))
            if not np.isnan(df.iloc[i, chp_smart_max_fr_idx + bias]):
                fst_smart_tup.append((df.iloc[i, chp_smart_runtime_idx + bias],
                                      df.iloc[i, chp_smart_max_fr_idx + bias]))
        chp_github_tup = sorted(chp_github_tup, key=lambda x: x[1])
        chp_smart_tup = sorted(chp_smart_tup, key=lambda x: x[1])
        fst_github_tup = sorted(fst_github_tup, key=lambda x: x[1])
        fst_smart_tup = sorted(fst_smart_tup, key=lambda x: x[1])
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Max failure rate')
        ax1.set_ylabel('Github caliber price', color=color)
        # ax1.plot(x, y1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('y2', color=color)
        # ax2.plot(x, y2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    # draw_integration_2d('incl', 'integration_dat_incl_cost/failure_rate_1.csv', 1)
    # draw_integration_2d('incl', 'integration_dat_incl_cost/failure_rate_0.csv', 0)
    # draw_pareto_3d('incl')
    # draw_integration_3d('incl')
    draw_baseline_line_graph()
