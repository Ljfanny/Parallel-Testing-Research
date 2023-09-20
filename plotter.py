import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
import textwrap
import numpy as np

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
        'chp_gh': 'Cheapest github caliber',
        'chp_smt': 'Cheapest smart baseline',
        'fst': 'Fastest',
        'fst_gh': 'Fastest github caliber',
        'fst_smt': 'Fastest smart baseline'
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
                   labels=labels)
        plt.savefig(f'{subdir}/{proj_name}.svg')
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
           for csv in csvs if csv.find('_') != -1]
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
        plt.savefig(f'integ_fig/ga_pareto3d/{proj_name}.svg')
        plt.close()
    summary_df.to_csv(f'integ_fig/ga_pareto3d/summary_result.csv', sep=',', header=True, index=False)


def smooth_xy(x, y):
    xxyy = np.column_stack((x, y))[~np.isnan(y)]
    xx = xxyy[:, 0]
    yy = xxyy[:, 1]
    x_smooth = np.linspace(np.min(xx), np.max(xx), 300)
    y_smooth = make_interp_spline(xx, yy)(x_smooth)
    return x_smooth, y_smooth


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
        ax.plot(x, y1, 'o-', c='#84a98c', label=labels[0])
        ax.plot(x, y2, 'o-', c='#354f52', label=labels[1])
        # ax.scatter(x, y1, c='#84a98c')
        # ax.scatter(x, y2, c='#354f52')
        # smooth_x, smooth_y1 = smooth_xy(x, y1)
        # ax.plot(smooth_x, smooth_y1, c='#84a98c', label=labels[0])
        # smooth_x, smooth_y2 = smooth_xy(x, y2)
        # ax.plot(smooth_x, smooth_y2, c='#354f52', label=labels[1])
        ax.set_title(title_map[chp_or_fst], size=12)
        ax.set_xlabel('Failure rate', size=12)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.yaxis.grid(True, linestyle='--', zorder=0)

    ig_path = 'integ_dat/ga'
    csvs = os.listdir(ig_path)
    dfs = [pd.read_csv(f'{ig_path}/{csv}')
           for csv in csvs if csv.find('_') != -1]
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
        plt.savefig(f'integ_fig/ga_trend_graph/{prog}.svg')
        plt.close()
    avg_chp = np.nanmean(np.array(avg_chp), axis=0)
    avg_fst = np.nanmean(np.array(avg_fst), axis=0)
    draw_subplot(norm_axes[0],
                 avg_chp[:, 2],
                 avg_chp[:, 0],
                 avg_chp[:, 1],
                 0,
                 ['Average runtime', 'Average price'])
    draw_subplot(norm_axes[1],
                 avg_fst[:, 2],
                 avg_fst[:, 0],
                 avg_fst[:, 1],
                 1,
                 ['Average runtime', 'Average price'])
    norm_fig.suptitle('Average trend', size=12, weight='bold')
    plt.savefig(f'integ_fig/ga_trend_graph/unification.svg')
    plt.close()


# Focus on each project!
def draw_integ_as_graph():
    csvs = [f for f in os.listdir('integ_dat') if os.path.isfile(os.path.join('integ_dat', f))]
    a = np.array([float(csv.replace('.csv', '')[csv.index('_a') + 2:]) for csv in csvs])
    a = sorted(a)
    a[0] = 0
    a[-1] = 1
    github_reco = []
    smart_reco = []

    def reds(x):
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
        plt.savefig(f'integ_fig/ga_as_pareto2d/{prog}.svg')
        plt.close()

    def sub_bar(ax,
                x,
                y1,
                y2,
                title):
        bar_width = 0.035
        # ax.set_ylim([-1.5, 1.5])
        ax.set_xticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), )
        ax.yaxis.grid(True, linestyle='--', zorder=0)
        ax.bar(x, y1, color='#507dbc', width=bar_width, edgecolor='#04080f', label='Runtime')
        ax.bar(x, y2, color='#bbd1ea', width=bar_width, edgecolor='#04080f', label='Price')
        ax.plot(x, np.array([1 for _ in range(len(x))]), 'o-', color='#1d3557', markersize=4)
        ax.plot(x, np.array([-1 for _ in range(len(x))]), 'o-', color='#1d3557', markersize=4)
        ax.set_title(title, size=12, weight='bold')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

    def bi_bar(ga,
               gh,
               smt,
               prog):
        ga_vs_gh = np.array([(t1[0] / t2[0], -t1[1] / t2[1]) if not np.isnan(t2[0]) else (0, 0)
                             for t1, t2 in zip(ga, gh)])
        ga_vs_smt = np.array([(t1[0] / t2[0], -t1[1] / t2[1]) if not np.isnan(t2[0]) else (0, 0)
                              for t1, t2 in zip(ga, smt)])
        github_reco.append(ga_vs_gh)
        smart_reco.append(ga_vs_smt)
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True, sharex=True)
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

        ax1.set_ylabel('The ratio compared to baseline', size=12, weight='bold',
                       rotation=90)
        ax1.set_xlabel(r'The parameter a', size=12, weight='bold')
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
        fig2.suptitle(prog, size=12, weight='bold')
        plt.savefig(f'integ_fig/ga_as_bi_bar/{prog}.svg')
        plt.close()

    whe_draw_bar = True
    dfs = [pd.read_csv(f'integ_dat/ga_a{i}.csv') for i in a]
    programs = dfs[0].iloc[:, 0]
    for i, proj in enumerate(programs):
        gene = []
        github = []
        smart = []
        for df in dfs:
            itm = df.iloc[i, :]
            gene.append((itm['runtime'],
                         itm['price'],
                         0))
            github.append((itm['github_caliber_runtime'],
                           itm['github_caliber_price'],
                           1))
            smart.append((itm['smart_baseline_runtime'],
                          itm['smart_baseline_price'],
                          2))
        if whe_draw_bar:
            bi_bar(gene,
                   github,
                   smart,
                   proj)
        else:
            pareto2d(gene,
                     github,
                     smart,
                     proj)
    if not whe_draw_bar:
        return
    github_tradeoff = np.nanmean(np.array(github_reco).sum(axis=2), axis=0)
    smart_tradeoff = np.nanmean(np.array(smart_reco).sum(axis=2), axis=0)
    github_runtime = np.nanmean(np.array(github_reco), axis=0)[:, 0]
    github_price = np.nanmean(np.array(github_reco), axis=0)[:, 1]
    smart_runtime = np.nanmean(np.array(smart_reco), axis=0)[:, 0]
    smart_price = np.nanmean(np.array(smart_reco), axis=0)[:, 1]
    # ----------------------tradeoff trend graph----------------------
    fig, panel = plt.subplots(figsize=(5, 4))
    # panel.plot(a, github_tradeoff, 'o-', label='vs GitHub caliber', color='#8EA604')
    # panel.plot(a, smart_tradeoff, 'o-', label='vs smart baseline', color='#D76A03')
    panel.scatter(a, github_tradeoff, label='vs GitHub caliber', color='#8EA604')
    panel.scatter(a, smart_tradeoff, label='vs smart baseline', color='#D76A03')
    smooth_a, smooth_gh = smooth_xy(a, github_tradeoff)
    smooth_a, smooth_smt = smooth_xy(a, smart_tradeoff)
    panel.plot(smooth_a, smooth_gh, color='#8EA604')
    panel.plot(smooth_a, smooth_smt, color='#D76A03')
    panel.plot(a, [0 for _ in range(len(a))], 'o-', color='#354f52')
    panel.set_title('Average tradeoff')
    panel.spines['top'].set_color('none')
    panel.spines['right'].set_color('none')
    panel.yaxis.grid(True, linestyle='--', zorder=0)
    panel.legend()
    plt.savefig(f'integ_fig/ga_as_bi_bar/tradeoff_trend_graph.svg')
    plt.close()
    # ----------------------mean bi bar----------------------
    fig, (panel1, panel2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True, sharex=True)
    sub_bar(panel1,
            a,
            github_runtime,
            github_price,
            'Avg. vs GitHub caliber')
    sub_bar(panel2,
            a,
            smart_runtime,
            smart_price,
            'Avg. vs smart baseline')
    panel1.set_ylabel('The ratio compared to baseline', size=12, weight='bold',
                      rotation=90)
    panel1.set_xlabel(r'The parameter a', size=12, weight='bold')
    panel1.xaxis.set_label_coords(1, -0.075)
    legend = panel1.legend(edgecolor='none')
    legend.set_bbox_to_anchor((-0.05, 1.05))
    fig.suptitle('Average rate')
    plt.savefig(f'integ_fig/ga_as_bi_bar/avg_rate_graph.svg')
    plt.close()


if __name__ == '__main__':
    # draw_integ_scatter2d('ga', 1)
    # draw_integ_scatter2d('ga', 0)
    # draw_integ_pareto3d('ga')
    # draw_tread_graph()
    draw_integ_as_graph()
