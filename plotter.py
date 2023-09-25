import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import numpy as np

matplotlib.use('TkAgg')

prefixes = {
    'chp': 'cheapest',
    'chp_gh': 'cheapest_github_caliber',
    'chp_smt': 'cheapest_smart_baseline',
    'fst': 'fastest',
    'fst_gh': 'fastest_github_caliber',
    'fst_smt': 'fastest_smart_baseline'
}
# plt.rc('font', family='Times New Roman', weight='bold')
plt.rc('font', family='Georgia')


def draw_scatter(ax, x, y, z, c, label=None):
    ax.scatter(x, y, z, c=c, label=label)


def output_plot(ax,
                proj_name):
    ax.set_title(textwrap.fill(proj_name))
    ax.set_xlabel('Runtime')
    ax.set_ylabel('Price')
    ax.set_zlabel('Max Failure Rate')
    ax.legend(fontsize=8)


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


def draw_integ_scatter2d(code,
                         fr):
    code_modu_map = {
        'ga': 'ga',
        'ig': 'ga_ig',
        'bf': 'bruteforce'
    }
    modu = code_modu_map[code]
    subdir = f'integ_fig/{modu}_scatter2d/failrate_{fr}'
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    colors = ['mediumpurple', 'thistle', 'mediumseagreen', 'cyan', 'steelblue', 'darkkhaki']
    nodes = {
        'chp': 'Cheapest',
        'chp_gh': 'Cheapest Github Caliber',
        'chp_smt': 'Cheapest Smart Baseline',
        'fst': 'Fastest',
        'fst_gh': 'Fastest Github Caliber',
        'fst_smt': 'Fastest Smart Baseline'
    }
    dat = pd.read_csv(f'integ_dat/{modu}/failrate_{fr}.csv').dropna()
    for _, row in dat.iterrows():
        proj_name = row['project']
        plt.title(textwrap.fill(proj_name))
        cnt = 0
        xy_fr_map = {}
        x_runtime = []
        y_price = []
        max_frs = []
        labels = []
        for key, prefix in prefixes.items():
            xi = row[f'{prefix}_runtime']
            yi = row[f'{prefix}_price']
            x_runtime.append(xi)
            y_price.append(yi)
            max_frs.append(row[f'{prefix}_max_failure_rate'])
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
                   labels=labels,
                   fontsize=8)
        plt.savefig(f'{subdir}/{proj_name}.pdf')
        plt.close()


def draw_integ_pareto3d(code):
    def reds(x):
        return np.array(x) if len(x) > 0 else np.empty((0, 4))

    def append_item(arr,
                    ser,
                    prefix,
                    mark):
        max_fr = ser[f'{prefix}_max_failure_rate']
        if np.isnan(max_fr):
            return
        arr.append((ser[f'{prefix}_runtime'],
                    ser[f'{prefix}_price'],
                    max_fr,
                    mark))

    code_modu_map = {
        'ga': 'ga',
        'ig': 'ga_ig',
        'bf': 'bruteforce'
    }
    modu = code_modu_map[code]
    subdir = f'integ_dat/{modu}'
    summary_df = pd.DataFrame(None,
                              columns=[
                                  'project',
                                  'normal_ga',
                                  'pareto_front_ga',
                                  'normal_github_caliber',
                                  'pareto_front_github_caliber',
                                  'normal_smart_baseline',
                                  'pareto_front_smart_baseline',
                                  'pareto_front_num',
                                  'ga_rate',
                                  'github_caliber_rate',
                                  'smart_baseline_rate'
                              ])
    csvs = os.listdir(subdir)
    dfs = [pd.read_csv(f'{subdir}/{csv}')
           for csv in csvs if csv.find('summary') == -1]
    proj_num = len(dfs[0])
    for i in range(proj_num):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()
        proj_name = dfs[0].iloc[i, :]['project']
        ga_xyz = []
        github_xyz = []
        smart_xyz = []
        # 0: ga; 1: github; 2: smart
        for df in dfs:
            ser = df.iloc[i, :]
            append_item(ga_xyz, ser, 'cheapest', 0)
            append_item(ga_xyz, ser, 'fastest', 0)
            append_item(github_xyz, ser, prefixes['chp_gh'], 1)
            append_item(github_xyz, ser, prefixes['fst_gh'], 1)
            append_item(smart_xyz, ser, prefixes['chp_smt'], 2)
            append_item(smart_xyz, ser, prefixes['fst_smt'], 2)
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
        ga_frontiers = reds(ga_frontiers)
        github_frontiers = reds(github_frontiers)
        smart_frontiers = reds(smart_frontiers)
        ga_non = reds(ga_xyz)
        github_non = reds(github_xyz)
        smart_non = reds(smart_xyz)
        frontiers_num = len(ga_frontiers) + len(github_frontiers) + len(smart_frontiers)
        summary_df.loc[len(summary_df.index)] = [
            proj_name,
            f'{len(ga_non) + len(ga_frontiers)}/12',
            len(ga_frontiers),
            f'{len(github_non) + len(github_frontiers)}/12',
            len(github_frontiers),
            f'{len(smart_non) + len(smart_frontiers)}/12',
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
                     f'GA Pareto Front: {len(ga_frontiers)}')
        draw_scatter(ax,
                     ga_non[:, 0],
                     ga_non[:, 1],
                     ga_non[:, 2],
                     'thistle',
                     f'GA Normal: {len(ga_non)}')
        draw_scatter(ax,
                     github_frontiers[:, 0],
                     github_frontiers[:, 1],
                     github_frontiers[:, 2],
                     'mediumseagreen',
                     f'Github Caliber Pareto Front: {len(github_frontiers)}')
        draw_scatter(ax,
                     github_non[:, 0],
                     github_non[:, 1],
                     github_non[:, 2],
                     'cyan',
                     f'Github Caliber Normal: {len(github_non)}')
        draw_scatter(ax,
                     smart_frontiers[:, 0],
                     smart_frontiers[:, 1],
                     smart_frontiers[:, 2],
                     'steelblue',
                     f'Smart Baseline Pareto Front: {len(smart_frontiers)}')
        draw_scatter(ax,
                     smart_non[:, 0],
                     smart_non[:, 1],
                     smart_non[:, 2],
                     'darkkhaki',
                     f'Smart Baseline Normal: {len(smart_non)}')
        output_plot(ax,
                    proj_name)
        plt.savefig(f'integ_fig/ga_pareto3d/{proj_name}.pdf')
        plt.close()
    summary_df.to_csv(f'integ_fig/ga_pareto3d/summary_result.csv', sep=',', header=True, index=False)


def draw_tread_graph():
    def draw_subplot(ax,
                     x,
                     y1,
                     y2,
                     chp_or_fst,
                     labels):
        title_map = {
            0: 'For Cheapest',
            1: 'For Fastest'
        }
        ax.plot(x, y1, 'o-', c='#84a98c', label=labels[0])
        ax.plot(x, y2, 'o-', c='#354f52', label=labels[1])
        ax.set_title(title_map[chp_or_fst], size=12)
        ax.set_xlabel('Failure Rate', size=12)
        ax.set_ylabel('Normalization Ratio', size=12)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(fontsize=8)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.yaxis.grid(True, linestyle='--', zorder=0)

    ig_path = 'integ_dat/ga'
    csvs = os.listdir(ig_path)
    dfs = [pd.read_csv(f'{ig_path}/{csv}')
           for csv in csvs if csv.find('summary') == -1]
    programs = dfs[0].iloc[:, 0]
    norm_fig, norm_axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    avg_chp = []
    avg_fst = []

    for i, prog in enumerate(programs):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        chp_tup = []
        fst_tup = []
        for j, df in enumerate(dfs):
            row = df.iloc[i, :]
            fr = float(csvs[j].replace('.csv', '').split('_')[1])
            chp_tup.append((
                row['cheapest_runtime'],
                row['cheapest_price'],
                fr
            ))
            fst_tup.append((
                row['fastest_runtime'],
                row['fastest_price'],
                fr
            ))
        chp_tup = np.array(sorted(chp_tup, key=lambda x: x[2]))
        fst_tup = np.array(sorted(fst_tup, key=lambda x: x[2]))
        chp_tup[:, 0] = chp_tup[:, 0] / np.nanmax(chp_tup[:, 0])
        chp_tup[:, 1] = chp_tup[:, 1] / np.nanmax(chp_tup[:, 1])
        fst_tup[:, 0] = fst_tup[:, 0] / np.nanmax(fst_tup[:, 0])
        fst_tup[:, 1] = fst_tup[:, 1] / np.nanmax(fst_tup[:, 1])
        avg_chp.append(chp_tup)
        avg_fst.append(fst_tup)
        draw_subplot(axes[0],
                     chp_tup[:, 2],
                     chp_tup[:, 0],
                     chp_tup[:, 1],
                     0,
                     ['Runtime', 'Price'])
        draw_subplot(axes[1],
                     fst_tup[:, 2],
                     fst_tup[:, 0],
                     fst_tup[:, 1],
                     1,
                     ['Runtime', 'Price'])
        fig.suptitle(prog)
        plt.savefig(f'integ_fig/ga_trend_graph/{prog}.pdf')
        plt.close()
    avg_chp = np.nanmean(np.array(avg_chp), axis=0)
    avg_fst = np.nanmean(np.array(avg_fst), axis=0)
    draw_subplot(norm_axes[0],
                 avg_chp[:, 2],
                 avg_chp[:, 0],
                 avg_chp[:, 1],
                 0,
                 ['Average Runtime Ratio', 'Average Price Ratio'])
    draw_subplot(norm_axes[1],
                 avg_fst[:, 2],
                 avg_fst[:, 0],
                 avg_fst[:, 1],
                 1,
                 ['Average Runtime Ratio', 'Average Price Ratio'])
    norm_fig.suptitle('Average Trend', size=12, weight='bold')
    plt.savefig(f'integ_fig/ga_trend_graph/unification.pdf')
    plt.close()


def sub_bar(ax,
            x,
            y1,
            y2,
            title,
            up_color='#457b9d',
            down_color='#cee5f2',
            bl_color='#1d3557',
            bar_width=0.035):
    ax.yaxis.grid(True, linestyle='--', zorder=0)
    ax.bar(x, y1, color=up_color, width=bar_width, edgecolor='#04080f', label='Runtime')
    ax.bar(x, y2, color=down_color, width=bar_width, edgecolor='#04080f', label='Price')
    ax.plot(x, np.array([1 for _ in range(len(x))]), 'o-', color=bl_color, markersize=4)
    ax.plot(x, np.array([-1 for _ in range(len(x))]), 'o-', color=bl_color, markersize=4)
    ax.set_title(title, size=12, weight='bold')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')


def draw_integ_as_graph(is_bar=False):
    csvs = [f for f in os.listdir('integ_dat') if os.path.isfile(os.path.join('integ_dat', f))]
    a = np.array([float(csv.replace('.csv', '')[csv.index('_a') + 2:]) for csv in csvs])
    a = sorted(a)
    a[0] = 0
    a[-1] = 1

    def reds(x):
        return np.array(x) if len(x) > 0 else np.empty((0, 4))

    def pareto2d(ga,
                 gh,
                 smt,
                 prog):
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
        ga = reds(ga)
        gh = reds(gh)
        smt = reds(smt)
        ga_frontiers = reds(ga_frontiers)
        gh_frontiers = reds(gh_frontiers)
        smt_frontiers = reds(smt_frontiers)
        ax.scatter(ga_frontiers[:, 0],
                   ga_frontiers[:, 1],
                   alpha=0.5,
                   c='mediumpurple',
                   label=f'GA Pareto Front: {len(np.unique(ga_frontiers))}({len(ga_frontiers)})')
        ax.scatter(ga[:, 0],
                   ga[:, 1],
                   alpha=0.5,
                   c='thistle',
                   label=f'GA Normal: {len(np.unique(ga))}({len(ga)})')
        ax.scatter(gh_frontiers[:, 0],
                   gh_frontiers[:, 1],
                   alpha=0.5,
                   c='mediumseagreen',
                   label=f'Github Caliber Pareto Front: {len(np.unique(gh_frontiers))}({len(gh_frontiers)})')
        ax.scatter(gh[:, 0],
                   gh[:, 1],
                   alpha=0.5,
                   c='cyan',
                   label=f'Github Caliber Normal: {len(np.unique(gh))}({len(gh)})')
        ax.scatter(smt_frontiers[:, 0],
                   smt_frontiers[:, 1],
                   alpha=0.5,
                   c='steelblue',
                   label=f'Smart Baseline Pareto Front: {len(np.unique(smt_frontiers))}({len(smt_frontiers)})')
        ax.scatter(smt[:, 0],
                   smt[:, 1],
                   alpha=0.5,
                   c='darkkhaki',
                   label=f'Smart Baseline Normal: {len(np.unique(smt))}({len(smt)})')
        ax.set_title(textwrap.fill(prog))
        ax.set_xlabel('Runtime')
        ax.set_ylabel('Price')
        ax.legend(fontsize=8)
        plt.savefig(f'integ_fig/ga_as_pareto2d/{prog}.pdf')
        plt.close()

    def set_parameters(ax1,
                       ax2):
        ax1.set_ylabel('The Ratio Compared to Baseline', size=12, weight='bold')
        ax2.set_ylabel('The Ratio Compared to Baseline', size=12, weight='bold')
        ax1.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax2.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax1.set_yticks([2, 1, 0, -1, -2])
        ax1.set_yticklabels([2, 1, 0, 1, 2])
        ax2.set_yticks([2, 1, 0, -1, -2])
        ax2.set_yticklabels([2, 1, 0, 1, 2])
        leg = ax1.legend(edgecolor='none')
        ax2.set_xlabel(r'The Parameter a', size=12, weight='bold')
        leg.set_bbox_to_anchor((0.05, 1.2))

    def bi_bar(prog,
               gh_rts,
               smt_rts):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0.1)
        sub_bar(ax1,
                a,
                gh_rts[:, 0],
                -gh_rts[:, 1],
                'GitHub Caliber Baseline')
        sub_bar(ax2,
                a,
                smt_rts[:, 0],
                -smt_rts[:, 1],
                'Smart Baseline')
        set_parameters(ax1, ax2)
        fig.suptitle(prog, size=16, weight='bold')
        plt.savefig(f'integ_fig/ga_as_bi_bar/{prog}.pdf')
        plt.close()

    def append_info(arr,
                    ser,
                    prefix,
                    code):
        runtime = ser[f'{prefix}_runtime']
        price = ser[f'{prefix}_price']
        if np.isnan(runtime):
            return
        arr.append((runtime, price, code))

    dfs = [pd.read_csv(f'integ_dat/ga_a{i}.csv') for i in a]
    programs = dfs[0].iloc[:, 0]
    is_reco_rt = True
    rts = []
    for i, proj in enumerate(programs):
        gene = []
        github = []
        smart = []
        # (runtime_rate, price_rate)
        github_rts = []
        smart_rts = []
        for j, df in enumerate(dfs):
            itm = df.iloc[i, :]
            gene.append((itm['runtime'],
                         itm['price'],
                         0))
            append_info(github, itm, 'github_caliber', 1)
            append_info(smart, itm, 'smart_baseline', 2)
            gh_runtime_rt = itm['github_caliber_runtime_rate']
            gh_price_rt = itm['github_caliber_price_rate']
            smt_runtime_rt = itm['smart_baseline_runtime_rate']
            smt_price_rt = itm['smart_baseline_price_rate']
            github_rts.append((gh_runtime_rt, gh_price_rt))
            smart_rts.append((smt_runtime_rt, smt_price_rt))
            if is_reco_rt:
                rts.append((df['github_caliber_runtime_rate'].dropna().mean(),
                            df['github_caliber_price_rate'].dropna().mean(),
                            df['smart_baseline_runtime_rate'].dropna().mean(),
                            df['smart_baseline_price_rate'].dropna().mean()))
        is_reco_rt = False
        if is_bar:
            bi_bar(proj,
                   np.array(github_rts),
                   np.array(smart_rts))
        else:
            pareto2d(gene,
                     github,
                     smart,
                     proj)
    if not is_bar:
        return
    rts = np.array(rts)
    # -------------------------------- tradeoff trend graph --------------------------------
    fig, pnl = plt.subplots(figsize=(10, 4))
    pnl.plot(a, rts[:, 0] * rts[:, 1], 'o-', label='vs GitHub Caliber', color='#b86f52', linewidth=2.5)
    pnl.plot(a, rts[:, 2] * rts[:, 3], 'o-', label='vs Smart Baseline', color='#f78764', linewidth=2.5)
    pnl.plot(a, [1 for _ in range(len(a))], '-.', color='#634133', linewidth=2)
    pnl.set_xlabel(r'The Parameter a')
    pnl.set_ylabel(r'Performance Increase')
    pnl.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    pnl.set_title('Average Tradeoff')
    pnl.spines['top'].set_color('none')
    pnl.spines['right'].set_color('none')
    pnl.yaxis.grid(True, linestyle='--', zorder=0)
    pnl.legend(fontsize=8)
    plt.savefig(f'integ_fig/ga_as_bi_bar/tradeoff_trend_graph.pdf')
    plt.close()
    # --------------------------------------- mean bi bar ---------------------------------------
    fig, (pnl1, pnl2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True, sharex=True)
    sub_bar(pnl1,
            a,
            rts[:, 0],
            -rts[:, 1],
            'Avg. vs GitHub Caliber')
    sub_bar(pnl2,
            a,
            rts[:, 2],
            -rts[:, 3],
            'Avg. vs Smart Baseline')
    set_parameters(pnl1, pnl2)
    fig.suptitle('Average Rate', size=16, weight='bold')
    plt.savefig(f'integ_fig/ga_as_bi_bar/avg_rate_graph.pdf')
    plt.close()


def draw_integ_proj_avg_rate_graph(goal_csv,
                                   sup_title,
                                   y1s,
                                   y1_labels,
                                   y2s,
                                   y2_labels):
    summary_per_proj_df = pd.read_csv(f'integ_dat/ga/{goal_csv}')
    gh_runtime_rts = summary_per_proj_df['github_caliber_avg_runtime_rate']
    gh_price_rts = summary_per_proj_df['github_caliber_avg_price_rate']
    smt_runtime_rts = summary_per_proj_df['smart_baseline_avg_runtime_rate']
    smt_price_rts = summary_per_proj_df['smart_baseline_avg_price_rate']
    gh_avg_runtime = np.nanmean(gh_runtime_rts)
    gh_avg_price = np.nanmean(gh_price_rts)
    smt_avg_runtime = np.nanmean(smt_runtime_rts)
    smt_avg_price = np.nanmean(smt_price_rts)
    gh_runtime_rts.loc[len(gh_runtime_rts)] = gh_avg_runtime
    gh_price_rts.loc[len(gh_price_rts)] = gh_avg_price
    smt_runtime_rts.loc[len(smt_runtime_rts)] = smt_avg_runtime
    smt_price_rts.loc[len(smt_price_rts)] = smt_avg_price
    projs = summary_per_proj_df['project']
    proj_id_map = {item['project'] + '_' + item['module']: item['id'] for _, item in
                   pd.read_csv('proj_info.csv').iterrows()}
    x = [proj_id_map[proj] for proj in projs]
    x.append('Tot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    sub_bar(ax1,
            [i for i in range(len(x))],
            gh_runtime_rts,
            -gh_price_rts,
            'Avg. vs GitHub Caliber',
            '#80727b',
            '#dbd2e0',
            '#37123c',
            0.75)
    sub_bar(ax2,
            [i for i in range(len(x))],
            smt_runtime_rts,
            -smt_price_rts,
            'Avg. vs Smart Baseline',
            '#80727b',
            '#dbd2e0',
            '#37123c',
            0.75)
    ax1.set_ylabel('The Ratio Compared to Baseline', size=12, weight='bold')
    ax2.set_ylabel('The Ratio Compared to Baseline', size=12, weight='bold')
    ax1.set_xticks([i for i in range(len(x))])
    ax2.set_xticks([i for i in range(len(x))])
    ax1.set_yticks(y1s)
    ax1.set_yticklabels(y1_labels)
    ax2.set_yticks(y2s)
    ax2.set_yticklabels(y2_labels)
    ax1.set_xticklabels(x)
    ax2.set_xticklabels(x)
    legend = ax1.legend(edgecolor='none')
    ax2.set_xlabel(r'Project Id', size=12, weight='bold')
    legend.set_bbox_to_anchor((0.05, 1.2))
    fig.suptitle(sup_title, size=16, weight='bold')
    plt.savefig(f'integ_fig/avg_rate_{goal_csv[8:-4]}_graph.pdf')
    plt.close()


if __name__ == '__main__':
    draw_integ_scatter2d('ga', 1)
    draw_integ_scatter2d('ga', 0)
    draw_integ_pareto3d('ga')
    draw_tread_graph()
    draw_integ_as_graph(True)
    draw_integ_proj_avg_rate_graph('summary_per_project_lower_price_goal.csv',
                                   'Average Rate with Lower Price Goal',
                                   [7, 6, 5, 4, 3, 2, 1, 0, -1],
                                   [7, 6, 5, 4, 3, 2, 1, 0, 1],
                                   [4, 3, 2, 1, 0, -1],
                                   [4, 3, 2, 1, 0, 1])
    draw_integ_proj_avg_rate_graph('summary_per_project_lower_runtime_goal.csv',
                                   'Average Rate with Lower Runtime Goal',
                                   [1, 0, -1, -2],
                                   [1, 0, 1, 2],
                                   [1, 0, -2, -4, -6, -8, -12],
                                   [1, 0, 2, 4, 6, 8, 12])
