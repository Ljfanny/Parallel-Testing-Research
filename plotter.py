import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import numpy as np

# matplotlib.use('TkAgg')

gh_max_fr = []
smt_max_fr = []
prefixes = {
    'chp': 'cheapest',
    'chp_gh': 'cheapest_github_baseline',
    'chp_smt': 'cheapest_smart_baseline',
    'fst': 'fastest',
    'fst_gh': 'fastest_github_baseline',
    'fst_smt': 'fastest_smart_baseline'
}
plt.rc('font', family='Georgia')
fr0_satisfied_projs = [
    'fastjson_dot',
    'commons-exec_dot',
    'incubator-dubbo_dubbo-remoting.dubbo-remoting-netty',
    'incubator-dubbo_dubbo-rpc.dubbo-rpc-dubbo',
    'rxjava2-extras_dot',
    'hutool_hutool-cron',
    'elastic-job-lite_dot',
    'elastic-job-lite_elastic-job-lite-core',
    'luwak_luwak',
    'fluent-logger-java_dot',
    'delight-nashorn-sandbox_dot',
    'http-request_dot',
    'spring-boot_dot',
    'retrofit_retrofit',
    'retrofit_retrofit-adapters.rxjava',
    'wro4j_wro4j-extensions']


def draw_scatter(ax, x, y, z, c, label=None):
    ax.scatter(x, y, z, c=c, label=label)


def output_plot(ax,
                proj_name):
    ax.set_title(textwrap.fill(proj_name))
    ax.set_xlabel('Running Time', size=12, weight='bold')
    ax.set_ylabel('Price', size=12, weight='bold')
    ax.set_zlabel('Max Flaky-Failure Rate', size=12, weight='bold')
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
        'chp_gh': 'Cheapest Github Baseline',
        'chp_smt': 'Cheapest Smart Baseline',
        'fst': 'Fastest',
        'fst_gh': 'Fastest Github Baseline',
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
        plt.xlabel('Running Time')
        plt.ylabel('Price')
        plt.legend(handles=[plt.scatter([], [], c=c) for c in colors],
                   labels=labels,
                   fontsize=8)
        plt.savefig(f'{subdir}/{proj_name}.pdf', bbox_inches='tight')
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
                                  'normal_github_baseline',
                                  'pareto_front_github_baseline',
                                  'normal_smart_baseline',
                                  'pareto_front_smart_baseline',
                                  'pareto_front_num',
                                  'ga_rate',
                                  'github_baseline_rate',
                                  'smart_baseline_rate'
                              ])
    csvs = os.listdir(subdir)
    dfs = [pd.read_csv(f'{subdir}/{csv}')
           for csv in csvs if csv.find('failrate') != -1]
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
                     f'Github Baseline Pareto Front: {len(github_frontiers)}')
        draw_scatter(ax,
                     github_non[:, 0],
                     github_non[:, 1],
                     github_non[:, 2],
                     'cyan',
                     f'Github Baseline Normal: {len(github_non)}')
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
        plt.savefig(f'integ_fig/ga_pareto3d/{proj_name}.pdf', bbox_inches='tight')
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
            0: 'For Price Optimization',
            1: 'For Running Time Optimization'
        }
        ax.plot(x, y1, 'o-', c='#84a98c', label=labels[0])
        ax.plot(x, y2, 'o-', c='#354f52', label=labels[1])
        ax.set_title(title_map[chp_or_fst], size=14, weight='bold')
        ax.set_xlabel('Flaky-Failure Rate', size=12, weight='bold')
        ax.set_ylabel('Normalization Ratio', size=12, weight='bold')
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.yaxis.grid(True, linestyle='--', zorder=0)
    ig_path = 'integ_dat/ga'
    csvs = os.listdir(ig_path)
    dfs = [pd.read_csv(f'{ig_path}/{csv}')
           for csv in csvs
           if csv.find('failrate') != -1]
    norm_fig, norm_axes = plt.subplots(1, 2, figsize=(10, 4.25), sharex=True, sharey=True)
    avg_chp = []
    avg_fst = []
    programs = dfs[4].dropna(subset=['cheapest_category']).iloc[:, 0]
    for i in programs.index:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.25))
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
        chp_tup[:, 0] = chp_tup[:, 0] / dfs[4].iloc[i]['cheapest_runtime']
        chp_tup[:, 1] = chp_tup[:, 1] / dfs[4].iloc[i]['cheapest_price']
        fst_tup[:, 0] = fst_tup[:, 0] / dfs[4].iloc[i]['fastest_runtime']
        fst_tup[:, 1] = fst_tup[:, 1] / dfs[4].iloc[i]['fastest_price']
        avg_chp.append(chp_tup)
        avg_fst.append(fst_tup)
        draw_subplot(axes[0],
                     chp_tup[:, 2],
                     chp_tup[:, 0],
                     chp_tup[:, 1],
                     0,
                     ['Running Time', 'Price'])
        draw_subplot(axes[1],
                     fst_tup[:, 2],
                     fst_tup[:, 0],
                     fst_tup[:, 1],
                     1,
                     ['Running Time', 'Price'])
        axes[0].legend(fontsize=8)
        fig.suptitle(programs[i])
        plt.savefig(f'integ_fig/ga_trend_graph/{programs[i]}.pdf', bbox_inches='tight')
        plt.close()
    avg_chp = np.nanmean(np.array(avg_chp), axis=0)
    avg_fst = np.nanmean(np.array(avg_fst), axis=0)
    draw_subplot(norm_axes[0],
                 avg_fst[:, 2],
                 avg_fst[:, 0],
                 avg_fst[:, 1],
                 1,
                 ['Avg. Running Time Ratio', 'Avg. Price Ratio'])
    draw_subplot(norm_axes[1],
                 avg_chp[:, 2],
                 avg_chp[:, 0],
                 avg_chp[:, 1],
                 0,
                 ['Avg. Running Time Ratio', 'Avg. Price Ratio'])
    norm_axes[0].legend(fontsize=8)
    print(avg_chp, avg_fst)
    # norm_fig.suptitle('Avg. Metric Ratio Trend of Acceptable Flaky-Failure Rate', size=12, weight='bold')
    plt.savefig(f'integ_fig/ga_trend_graph/unification.pdf', bbox_inches='tight')
    plt.close()


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
                   label=f'Github Baseline Pareto Front: {len(np.unique(gh_frontiers))}({len(gh_frontiers)})')
        ax.scatter(gh[:, 0],
                   gh[:, 1],
                   alpha=0.5,
                   c='cyan',
                   label=f'Github Baseline Normal: {len(np.unique(gh))}({len(gh)})')
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
        ax.set_title(textwrap.fill(prog), size=14, weight='bold')
        ax.set_xlabel('Running Time Ratio', size=12, weight='bold')
        ax.set_ylabel('Price Ratio', size=12, weight='bold')
        ax.legend(fontsize=8)
        plt.savefig(f'integ_fig/ga_as_pareto2d/{prog}.pdf', bbox_inches='tight')
        plt.close()

    def set_parameters(ax1,
                       ax2):
        ax1.set_ylabel('The Ratio Compared to Baseline', size=12, weight='bold')
        ax2.set_ylabel('The Ratio Compared to Baseline', size=12, weight='bold')
        ax1.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax2.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax1.set_yticks([2, 1, 0, -1, -2])
        ax1.set_yticklabels([2, 1, 0, 1, 2])
        ax2.set_yticks([2, 1, 0, -1, -2])
        ax2.set_yticklabels([2, 1, 0, 1, 2])
        ax2.set_xlabel(r'The Parameter a', size=12, weight='bold')

    def sub_bar(ax,
                x,
                y1,
                y2,
                title,
                up_color='#8d99ae',
                down_color='#e1e5f2',
                bl_color='#051923',
                bar_width=0.035):
        ax.yaxis.grid(True, linestyle='--', zorder=0)
        ax.bar(x, y1, color=up_color, width=bar_width, edgecolor='#04080f', label='Running Time Ratio')
        ax.bar(x, y2, color=down_color, width=bar_width, edgecolor='#04080f', label='Price Ratio')
        ax.plot(x, np.array([1 for _ in range(len(x))]), 'o-', color=bl_color, markersize=4)
        ax.plot(x, np.array([-1 for _ in range(len(x))]), 'o-', color=bl_color, markersize=4)
        ax.set_title(title, size=12, weight='bold')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

    def bi_bar(prog,
               gh_rts,
               smt_rts):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0.1)
        sub_bar(ax1,
                a,
                gh_rts[:, 0],
                -gh_rts[:, 1],
                'GitHub Baseline')
        sub_bar(ax2,
                a,
                smt_rts[:, 0],
                -smt_rts[:, 1],
                'Smart Baseline')
        set_parameters(ax1, ax2)
        # fig.suptitle(prog, size=16, weight='bold')
        plt.savefig(f'integ_fig/ga_as_bi_bar/{prog}.pdf', bbox_inches='tight')
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
    is_reco_rt = True
    rts = []
    for i, proj in enumerate(fr0_satisfied_projs):
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
            append_info(github, itm, 'github_baseline', 1)
            append_info(smart, itm, 'smart_baseline', 2)
            gh_runtime_rt = itm['github_baseline_runtime_rate']
            gh_price_rt = itm['github_baseline_price_rate']
            smt_runtime_rt = itm['smart_baseline_runtime_rate']
            smt_price_rt = itm['smart_baseline_price_rate']
            github_rts.append((gh_runtime_rt, gh_price_rt))
            smart_rts.append((smt_runtime_rt, smt_price_rt))
            if is_reco_rt:
                rts.append((df['github_baseline_runtime_rate'].dropna().mean(),
                            df['github_baseline_price_rate'].dropna().mean(),
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
    # pnl.plot(a, [1 for _ in range(len(a))], '-.', color='#bcb8b1', linewidth=2.25)
    pnl.plot(a, rts[:, 0] * rts[:, 1], 'o-', label='vs GitHub Baseline', color='#c89f9c', linewidth=2.5)
    pnl.plot(a, rts[:, 2] * rts[:, 3], 'o-', label='vs Smart Baseline', color='#9d8189', linewidth=2.5)
    print(rts[:, 0] * rts[:, 1], rts[:, 2] * rts[:, 3])
    pnl.set_xlabel(r'The Parameter a', size=12, weight='bold')
    pnl.set_ylabel(r'Performance Increase', size=12, weight='bold')
    pnl.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # pnl.set_title('Avg. Tradeoff Value over all Projects', size=16, weight='bold')
    pnl.spines['top'].set_color('none')
    pnl.spines['right'].set_color('none')
    pnl.yaxis.grid(True, linestyle='--', zorder=0)
    pnl.legend(fontsize=8)
    plt.savefig(f'integ_fig/ga_as_bi_bar/tradeoff_trend_graph.pdf', bbox_inches='tight')
    plt.close()
    # --------------------------------------- mean bi bar ---------------------------------------
    fig, (pnl1, pnl2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True, sharex=True)
    sub_bar(pnl1,
            a,
            rts[:, 0],
            -rts[:, 1],
            'vs GitHub Baseline')
    sub_bar(pnl2,
            a,
            rts[:, 2],
            -rts[:, 3],
            'vs Smart Baseline')
    set_parameters(pnl1, pnl2)
    pnl1.legend()
    print(rts)
    # fig.suptitle('Average Rate', size=16, weight='bold')
    plt.savefig(f'integ_fig/ga_as_bi_bar/avg_rate_graph.pdf', bbox_inches='tight')
    plt.close()


def draw_integ_proj_avg_rate_graph(goal_subdir,
                                   # sup_title,
                                   y1s,
                                   y1_labels,
                                   y2s,
                                   y2_labels):
    def extract_dat(prefix):
        failrate0_df = pd.read_csv(f'integ_dat/{goal_subdir}/failrate_0.csv').dropna()
        gh_runtime_rts = failrate0_df[f'{prefix}_github_baseline_runtime_rate']
        gh_price_rts = failrate0_df[f'{prefix}_github_baseline_price_rate']
        smt_runtime_rts = failrate0_df[f'{prefix}_smart_baseline_runtime_rate']
        smt_price_rts = failrate0_df[f'{prefix}_smart_baseline_price_rate']
        gh_max_fr.append(np.mean(failrate0_df[f'{prefix}_github_baseline_max_failure_rate']))
        smt_max_fr.append(np.mean(failrate0_df[f'{prefix}_smart_baseline_max_failure_rate']))
        return gh_runtime_rts, gh_price_rts, smt_runtime_rts, smt_price_rts

    def sub_double_bar(ax,
                       indexes,
                       y1,
                       y2,
                       y3,
                       y4,
                       title):
        bar_width = 0.53
        indexes = np.array(indexes)
        ax.bar(indexes - bar_width / 2 + 0.1, y1, color='sandybrown', width=bar_width, edgecolor='#04080f',
               label='Running Time Ratio with Running Time Optimization', hatch='//')
        ax.bar(indexes - bar_width / 2 + 0.1, y3, color='lavender', width=bar_width, edgecolor='#04080f',
               label='Price Ratio with Running Time Optimization', hatch='//')
        ax.bar(indexes + bar_width / 2 - 0.1, y2, color='lavender', width=bar_width, edgecolor='#04080f',
               label='Running Time Ratio with Price Optimization')
        ax.bar(indexes + bar_width / 2 - 0.1, y4, color='sandybrown', width=bar_width, edgecolor='#04080f',
               label='Price Ratio with Price Optimization')
        ax.plot(x, np.array([1 for _ in range(len(x))]), 'o-', color='#22223b', markersize=4)
        ax.plot(x, np.array([-1 for _ in range(len(x))]), 'o-', color='#22223b', markersize=4)
        ax.yaxis.grid(True, linestyle='--', zorder=0)
        ax.set_title(title, size=12, weight='bold')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

    tradeoff_per_proj_df = pd.DataFrame()
    gh_tm_runtime_rts, gh_tm_price_rts, smt_tm_runtime_rts, smt_tm_price_rts = extract_dat(
        'fastest')
    gh_pri_runtime_rts, gh_pri_price_rts, smt_pri_runtime_rts, smt_pri_price_rts = extract_dat(
        'cheapest')
    avg_idx = len(gh_pri_runtime_rts)
    proj_info_df = pd.read_csv('proj_info.csv')
    proj_id_map = {itm['project-module']: itm['id'].replace('P', 'M') for _, itm in proj_info_df.iterrows()}
    x = [proj_id_map[proj] for proj in fr0_satisfied_projs]
    tradeoff_per_proj_df['project'] = x
    tradeoff_per_proj_df['github_baseline_tradeoff_with_lowest_price'] = gh_pri_runtime_rts * gh_pri_price_rts
    tradeoff_per_proj_df['github_baseline_tradeoff_with_lowest_runtime'] = gh_tm_runtime_rts * gh_tm_price_rts
    tradeoff_per_proj_df['smart_baseline_tradeoff_with_lowest_price'] = smt_pri_runtime_rts * smt_pri_price_rts
    tradeoff_per_proj_df['smart_baseline_tradeoff_with_lowest_runtime'] = smt_tm_runtime_rts * smt_tm_price_rts
    gh_tm_runtime_rts[avg_idx] = np.mean(gh_tm_runtime_rts)
    gh_tm_price_rts[avg_idx] = np.mean(gh_tm_price_rts)
    gh_pri_runtime_rts[avg_idx] = np.mean(gh_pri_runtime_rts)
    gh_pri_price_rts[avg_idx] = np.mean(gh_pri_price_rts)
    smt_tm_runtime_rts[avg_idx] = np.mean(smt_tm_runtime_rts)
    smt_tm_price_rts[avg_idx] = np.mean(smt_tm_price_rts)
    smt_pri_runtime_rts[avg_idx] = np.mean(smt_pri_runtime_rts)
    smt_pri_price_rts[avg_idx] = np.mean(smt_pri_price_rts)
    x.append('Avg.')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    sub_double_bar(ax1,
                   [i for i in range(len(x))],
                   gh_tm_runtime_rts,
                   gh_pri_runtime_rts,
                   -gh_tm_price_rts,
                   -gh_pri_price_rts,
                   'vs GitHub Baseline')
    sub_double_bar(ax2,
                   [i for i in range(len(x))],
                   smt_tm_runtime_rts,
                   smt_pri_runtime_rts,
                   -smt_tm_price_rts,
                   -smt_pri_price_rts,
                   'vs Smart Baseline')
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
    ax2.set_xlabel(r'ID', size=12, weight='bold')
    # fig.suptitle(sup_title, size=16, weight='bold')
    ax1.legend()
    plt.savefig(f'integ_fig/avg_rate_{goal_subdir}_graph.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    draw_integ_scatter2d('ga', 1)
    draw_integ_scatter2d('ga', 0)
    draw_integ_pareto3d('ga')
    draw_tread_graph()
    draw_integ_as_graph(True)
    draw_integ_proj_avg_rate_graph('ga_ig',
                                   # 'Avg. Ratio for GASearch without Set-up Time',
                                   [8, 6, 4, 2, 0, -2],
                                   [8, 6, 4, 2, 0, 2],
                                   [8, 6, 4, 2, 0, -2, -4],
                                   [8, 6, 4, 2, 0, 2, 4])
    draw_integ_proj_avg_rate_graph('ga',
                                   # 'Avg. Ratio for GASearch with Set-up Time',
                                   [4, 3, 2, 1, 0, -1, -2],
                                   [4, 3, 2, 1, 0, 1, 2],
                                   [4, 3, 2, 1, 0, -1, -2],
                                   [4, 3, 2, 1, 0, 1, 2])
    print(np.mean(gh_max_fr),
          np.mean(smt_max_fr))
