import copy
import os
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import numpy as np

scatter2d_path = 'integration_ga_scatter2d'
pareto3d_path = 'ga_pareto3d'
int_pareto3d_path = 'integration_ga_pareto3d'
int_fac_pareto2d_path = 'integration_ga_with_factors_pareto2d'
int_fac_bar_path = 'integration_ga_with_factors_bi_bar'

biases = {'chp': 0, 'chp_gh': 4, 'chp_smt': 10, 'fst': 17, 'fst_gh': 21, 'fst_smt': 27}
plt.rc('font', family='Times New Roman', weight='bold')


def draw_scatter(ax, x, y, z, c, label=None):
    ax.scatter(x, y, z, c=c, label=label)


def output_plot(ax,
                proj_name):
    ax.set_title(textwrap.fill(proj_name))
    ax.set_xlabel('Runtime')
    ax.set_ylabel('Price')
    ax.set_zlabel('Max failure rate')
    ax.legend()


def pareto_frontier_multi(is_avail,
                          dat):
    dat_len = len(dat)
    dominated = np.zeros(dat_len, dtype=bool)
    if is_avail:
        avail_dat = dat
    else:
        avail_dat = dat[:, :-1]
    for i in range(dat_len):
        for j in range(dat_len):
            if all(avail_dat[j] <= avail_dat[i]) and any(avail_dat[j] < avail_dat[i]):
                dominated[i] = True
                break
    return dat[~dominated]


def draw_pareto3d(modu):
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
        frontiers = pareto_frontier_multi(True,
                                          xyz)
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
                     'mediumpurple',
                     f'Pareto frontiers: {len(frontiers)}')
        draw_scatter(ax,
                     x_non,
                     y_non,
                     z_non,
                     'darkkhaki',
                     f'Normal: {len(xyz_non)}')
        output_plot(ax,
                    proj_name)
        plt.savefig(f'{pareto3d_path}/{proj_name}.svg')
        plt.close()


def draw_int_scatter2d(modu,
                       dat_name,
                       fr):
    subdir = f'{scatter2d_path}/{modu}_failure_rate_{fr}'
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    proj_idx = 0
    colors = ['mediumpurple', 'thistle', 'mediumseagreen', 'cyan', 'steelblue', 'darkkhaki']
    nodes = {'chp': 'Cheapest', 'chp_gh': 'Cheapest github_caliber', 'chp_smt': 'Cheapest smart baseline',
             'fst': 'Fastest', 'fst_gh': 'Fastest github caliber', 'fst_smt': 'Fastest smart baseline'}
    runtime_idx = 3
    price_idx = 4
    max_fr_idx = 5
    dat = pd.read_csv(dat_name)
    for _, row in dat.iterrows():
        proj_name = row[proj_idx]
        plt.title(textwrap.fill(proj_name))
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
                    c=colors)
        plt.xlabel('Runtime')
        plt.ylabel('Price')
        plt.legend(handles=[plt.scatter([], [], c=c) for c in colors],
                   labels=labels)
        plt.savefig(f'{subdir}/{proj_name}.svg')
        plt.close()


def draw_int_pareto3d(modu):
    def temp(x):
        return np.array(x) if len(x) > 0 else np.empty((0, 4))

    modu_map = {'incl': 'integration_incl_cost/',
                'excl': 'integration_excl_cost/',
                'bf': 'integration_bruteforce/'}
    dat_path = modu_map[modu]
    rec_degree_df = pd.DataFrame(None,
                                 columns=[
                                     'project',
                                     'ga_non_nan',
                                     'ga_pareto_front',
                                     'github_caliber_non_nan',
                                     'github_caliber_pareto_front',
                                     'smart_baseline_non_nan',
                                     'smart_baseline_pareto_front',
                                     'pareto_front_number',
                                     'ga_rate',
                                     'github_caliber_rate',
                                     'smart_baseline_rate'
                                 ])
    proj_idx = 0
    runtime_idx = 3
    price_idx = 4
    max_fr_idx = 5
    csvs = os.listdir(dat_path)[1:]
    dfs = [pd.read_csv(f'{dat_path}{csv}') for csv in csvs]
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
        frontiers = pareto_frontier_multi(False,
                                          np.array(xyz))
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
        ga_frontiers = temp(ga_frontiers)
        github_frontiers = temp(github_frontiers)
        smart_frontiers = temp(smart_frontiers)
        ga_non = temp(ga_xyz)
        github_non = temp(github_xyz)
        smart_non = temp(smart_xyz)
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
                     'mediumpurple',
                     f'GA pareto frontiers: {len(ga_frontiers)}')
        draw_scatter(ax,
                     ga_non[:, 0],
                     ga_non[:, 1],
                     ga_non[:, 2],
                     'thistle',
                     f'GA normal: {len(ga_non)}')
        draw_scatter(ax,
                     github_frontiers[:, 0],
                     github_frontiers[:, 1],
                     github_frontiers[:, 2],
                     'mediumseagreen',
                     f'Github caliber pareto frontiers: {len(github_frontiers)}')
        draw_scatter(ax,
                     github_non[:, 0],
                     github_non[:, 1],
                     github_non[:, 2],
                     'cyan',
                     f'Github caliber normal: {len(github_non)}')
        draw_scatter(ax,
                     smart_frontiers[:, 0],
                     smart_frontiers[:, 1],
                     smart_frontiers[:, 2],
                     'steelblue',
                     f'Smart baseline pareto frontiers: {len(smart_frontiers)}')
        draw_scatter(ax,
                     smart_non[:, 0],
                     smart_non[:, 1],
                     smart_non[:, 2],
                     'darkkhaki',
                     f'Smart baseline normal: {len(smart_non)}')
        output_plot(ax,
                    proj_name)
        plt.savefig(f'{int_pareto3d_path}/{proj_name}.svg')
        plt.close()
    rec_degree_df.to_csv(f'{int_pareto3d_path}/summary_result.csv', sep=',', header=True, index=False)


def draw_tread_graph():
    def draw_subplot(ax,
                     x,
                     y1,
                     y2,
                     chp_or_fst,
                     labels):
        title_map = {
            0: 'For cheapest',
            1: 'For fastest'
        }
        ax.plot(x, y1, 'o-', c='#5170d7', label=labels[0])
        ax.plot(x, y2, 'o-', c='#a87dc2', label=labels[1])
        ax.set_title(title_map[chp_or_fst])
        ax.set_xlabel('Failure rate')
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend()

    def normalize(dat):
        return dat / np.nanmean(dat)

    ig_path = 'integration_incl_cost/'
    csvs = os.listdir(ig_path)[1:]
    dfs = [pd.read_csv(f'{ig_path}{csv}') for csv in csvs]
    programs = dfs[0].iloc[:, 0]
    norm_fig, norm_axes = plt.subplots(1, 2, figsize=(10, 4))
    avg_chp = []
    avg_fst = []
    for i, prog in enumerate(programs):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        chp_tup = []
        fst_tup = []
        for j, df in enumerate(dfs):
            row = df.iloc[i, :]
            # runtime=3; price=4;
            fr = float(csvs[j].replace('.csv', '').split('_')[2])
            chp_tup.append((
                row[3],
                row[4],
                fr
            ))
            fst_tup.append((
                row[20],
                row[21],
                fr
            ))
        chp_tup = np.array(sorted(chp_tup, key=lambda x: x[2]))
        fst_tup = np.array(sorted(fst_tup, key=lambda x: x[2]))
        avg_chp.append(chp_tup)
        avg_fst.append(fst_tup)
        draw_subplot(axes[0],
                     chp_tup[:, 2],
                     normalize(chp_tup[:, 0]),
                     normalize(chp_tup[:, 1]),
                     0,
                     ['Runtime', 'Price'])
        draw_subplot(axes[1],
                     fst_tup[:, 2],
                     normalize(fst_tup[:, 0]),
                     normalize(fst_tup[:, 1]),
                     1,
                     ['Runtime', 'Price'])
        fig.suptitle(prog)
        # fig.subplots_adjust(wspace=0.5)
        plt.savefig(f'trend_plots/{prog}.svg')
        plt.close()

    avg_chp = np.nanmean(np.array(avg_chp), axis=0)
    avg_fst = np.nanmean(np.array(avg_fst), axis=0)
    draw_subplot(norm_axes[0],
                 avg_chp[:, 2],
                 normalize(avg_chp[:, 0]),
                 normalize(avg_chp[:, 1]),
                 0,
                 ['Average runtime', 'Average price'])
    draw_subplot(norm_axes[1],
                 avg_fst[:, 2],
                 normalize(avg_fst[:, 0]),
                 normalize(avg_fst[:, 1]),
                 1,
                 ['Average runtime', 'Average price'])
    norm_fig.suptitle('Average trend')
    # norm_fig.subplots_adjust(wspace=0.5)
    plt.savefig(f'trend_plots/unification.svg')
    plt.close()


# Focus on each project!
def draw_int_fac_graph():
    int_fac_path = 'integration_ga_with_factors/'
    csvs = os.listdir(int_fac_path)
    a = np.array([float(csv.replace('.csv', '')[1: csv.index('b')]) for csv in csvs])
    github_tradeoff = []
    smart_tradeoff = []

    def temp(x):
        return np.array(x) if len(x) > 0 else np.empty((0, 4))

    def pareto2d(ga,
                 gh,
                 smt,
                 prog):
        gh = [t for t in gh if not np.isnan(t[0])]
        smt = [t for t in smt if not np.isnan(t[0])]
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.invert_xaxis()
        ax.invert_yaxis()
        frontiers = pareto_frontier_multi(False,
                                          np.array(ga + gh + smt))
        ga_frontiers = []
        gh_frontiers = []
        smt_frontiers = []
        for frontier in frontiers:
            tup = tuple(frontier)
            if frontier[2] == 0:
                ga.remove(tup)
                ga_frontiers.append(tup)
            elif frontier[2] == 1:
                gh.remove(tup)
                gh_frontiers.append(tup)
            else:
                smt.remove(tup)
                smt_frontiers.append(tup)
        ga = temp(ga)
        gh = temp(gh)
        smt = temp(smt)
        ga_frontiers = temp(ga_frontiers)
        gh_frontiers = temp(gh_frontiers)
        smt_frontiers = temp(smt_frontiers)
        ax.scatter(ga_frontiers[:, 0],
                   ga_frontiers[:, 1],
                   alpha=0.5,
                   c='mediumpurple',
                   label=f'GA pareto frontiers: {len(np.unique(ga_frontiers))}({len(ga_frontiers)})')
        ax.scatter(ga[:, 0],
                   ga[:, 1],
                   alpha=0.5,
                   c='thistle',
                   label=f'GA normal: {len(np.unique(ga))}({len(ga)})')
        ax.scatter(gh_frontiers[:, 0],
                   gh_frontiers[:, 1],
                   alpha=0.5,
                   c='mediumseagreen',
                   label=f'Github caliber pareto frontiers: {len(np.unique(gh_frontiers))}({len(gh_frontiers)})')
        ax.scatter(gh[:, 0],
                   gh[:, 1],
                   alpha=0.5,
                   c='cyan',
                   label=f'Github caliber normal: {len(np.unique(gh))}({len(gh)})')
        ax.scatter(smt_frontiers[:, 0],
                   smt_frontiers[:, 1],
                   alpha=0.5,
                   c='steelblue',
                   label=f'Smart baseline pareto frontiers: {len(np.unique(smt_frontiers))}({len(smt_frontiers)})')
        ax.scatter(smt[:, 0],
                   smt[:, 1],
                   alpha=0.5,
                   c='darkkhaki',
                   label=f'Smart baseline normal: {len(np.unique(smt))}({len(smt)})')
        ax.set_title(textwrap.fill(prog))
        ax.set_xlabel('Runtime')
        ax.set_ylabel('Price')
        ax.legend(fontsize=5)
        plt.savefig(f'{int_fac_pareto2d_path}/{prog}.svg')
        plt.close()

    def sub_bar(ax,
                x,
                y1,
                y2,
                title):
        bar_width = 0.06
        ax.set_ylim([-1.5, 1.5])
        ax.set_xticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), )


        ax.yaxis.grid(True, linestyle='--',zorder=0)

        ax.bar(x, y1, color='#507dbc', width=bar_width, edgecolor='#04080f', label='Price')
        ax.bar(x, y2, color='#bbd1ea', width=bar_width, edgecolor='#04080f', label='Runtime')
        ax.plot(x, np.array([1 for _ in range(len(x))]), 'o-', color='#04080f')
        ax.plot(x, np.array([-1 for _ in range(len(x))]), 'o-', color='#04080f')

        ax.set_title(title, fontproperties='Times New Roman', size=12, weight='bold')
        # ax.set_ylabel('The ration compare to the baseline',fontproperties = 'Times New Roman',size = 13, weight='bold')

    def bi_bar(ga,
               gh,
               smt,
               prog):
        ga_vs_gh = np.array([(t1[0] / t2[0], -t1[1] / t2[1]) if not np.isnan(t2[0]) else (0, 0)
                             for t1, t2 in zip(ga, gh)])
        ga_vs_smt = np.array([(t1[0] / t2[0], -t1[1] / t2[1]) if not np.isnan(t2[0]) else (0, 0)
                              for t1, t2 in zip(ga, smt)])
        github_tradeoff.append(np.add(ga_vs_gh[:, 0], ga_vs_gh[:, 1]))
        smart_tradeoff.append(np.add(ga_vs_smt[:, 0], ga_vs_smt[:, 1]))
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0.1)
        sub_bar(ax1,
                a,
                ga_vs_gh[:, 0],
                ga_vs_gh[:, 1],
                'GitHub Caliber Baseline')
        sub_bar(ax2,
                a,
                ga_vs_smt[:, 0],
                ga_vs_smt[:, 1],
                'Smart Baseline')

        ax1.set_ylabel('The ratio compared to baseline', fontproperties='Times New Roman', size=12, weight='bold',
                       rotation=90)
        ax1.set_xlabel(r'The parameter a', fontproperties='Times New Roman', size=12, weight='bold')
        ax1.xaxis.set_label_coords(1, -0.075)
        # neg_ax1 = ax1.twinx()
        # ax1.set_ylabel('Price', fontproperties='Times New Roman', size=12, weight='bold', rotation=90)
        # ax1.yaxis.set_label_position('left')
        #
        # neg_ax1.set_ylabel('Runtime', fontproperties='Times New Roman', size=12, weight='bold', rotation=90)
        # neg_ax1.yaxis.set_label_position('left')
        #
        # ax1.yaxis.set_label_coords(-0.1, 0.65)
        # neg_ax1.yaxis.set_label_coords(-0.1, 0.25)

        legend = ax1.legend(edgecolor='none')
        legend.set_bbox_to_anchor((-0.05, 1.05))

        fig2.suptitle(prog, fontproperties='Times New Roman', size=12, weight='bold')
        plt.savefig(f'{int_fac_bar_path}/{prog}.svg')
        plt.close()

    dfs = [pd.read_csv(f'{int_fac_path}{csv}') for csv in csvs]
    runtime_idx = 3
    price_idx = 4
    gaps = {
        'gh': 5,
        'smt': 11
    }
    programs = dfs[0].iloc[:, 0]
    for i, proj in enumerate(programs):
        gene = []
        github = []
        smart = []
        for df in dfs:
            itm = df.iloc[i, :]
            gene.append((itm[runtime_idx],
                         itm[price_idx],
                         0))
            github.append((itm[runtime_idx + gaps['gh']],
                           itm[price_idx + gaps['gh']],
                           1))
            smart.append((itm[runtime_idx + gaps['smt']],
                          itm[price_idx + gaps['smt']],
                          2))
        # pareto2d(copy.deepcopy(gene),
        #          copy.deepcopy(github),
        #          copy.deepcopy(smart),
        #          proj)
        bi_bar(gene,
               github,
               smart,
               proj)
    github_tradeoff = np.nanmean(np.array(github_tradeoff), axis=0)
    smart_tradeoff = np.nanmean(np.array(smart_tradeoff), axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(a, github_tradeoff, 'o-', label='vs GitHub caliber')
    ax.plot(a, smart_tradeoff, 'o-', label='vs smart baseline')
    ax.set_title('Average tradeoff')
    ax.legend()
    plt.savefig(f'{int_fac_bar_path}/avg_trend_graph.svg')
    plt.close()


if __name__ == '__main__':
    # draw_int_scatter2d('incl', 'integration_incl_cost/failure_rate_1.csv', 1)
    # draw_int_scatter2d('incl', 'integration_incl_cost/failure_rate_0.csv', 0)
    # draw_pareto3d('incl')
    # draw_int_pareto3d('incl')
    # draw_tread_graph()
    draw_int_fac_graph()
