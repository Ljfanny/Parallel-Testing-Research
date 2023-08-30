import os
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import numpy as np

pareto_path = 'pareto'
trade_off_path = 'trade_off'


def pareto_runtime_price(dat_name: str):
    proj_idx = 0
    colors = ['mediumpurple', 'thistle', 'lightsteelblue', 'darkseagreen', 'slategrey', 'darkkhaki']
    biases = {'chp': 0, 'chp_gh': 4, 'chp_smt': 10, 'fst': 17, 'fst_gh': 21, 'fst_smt': 27}
    nodes = {'chp': 'cheapest', 'chp_gh': 'cheapest_github_caliber', 'chp_smt': 'cheapest_smart_baseline',
             'fst': 'fastest', 'fst_gh': 'fastest_github_caliber', 'fst_smt': 'fastest_smart_baseline'}
    confs_idx = 2
    runtime_idx = 3
    price_idx = 4
    max_fr_idx = 5
    dat = pd.read_csv(dat_name)
    for _, row in dat.iterrows():
        proj_name = row[proj_idx]
        x_runtime = []
        y_price = []
        max_frs = []
        annotations = []
        for key, bias in biases.items():
            x_runtime.append(row[runtime_idx + bias])
            y_price.append(row[price_idx + bias])
            max_frs.append(row[max_fr_idx + bias])
            annotations.append(nodes[key])
        plt.scatter(x_runtime, y_price, s=75, c=colors)
        plt.title(textwrap.fill(proj_name), fontsize=14)
        plt.xlabel('Runtime')
        plt.ylabel('Price')
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.legend(handles=[plt.scatter([], [], c=c) for c in colors],
                   labels=annotations)
        # for x, y, fr in zip(x_runtime, y_price, max_frs):
        #     plt.text(x, y, round(fr, 2), ha='center')
        plt.savefig(f'{pareto_path}/{proj_name}.png', dpi=1000)
        plt.close()


def draw_trade_off():
    def get_x_y(df):
        # (runtime, price): (max failure rate, confs)
        mp = {}
        confs_idx = 2
        runtime_idx = 4
        price_idx = 5
        max_fr_idx = 7
        for _, row in df.iterrows():
            key = (row[runtime_idx], row[price_idx])
            if key not in mp.keys() or row[max_fr_idx] < mp[key][0]:
                mp[key] = (row[max_fr_idx], row[confs_idx])
        keys = np.array(list(mp.keys()))
        return keys[:, 0], keys[:, 1]

    def draw(subplot, x, y, title):
        colors = np.random.rand(len(x))
        subplot.scatter(x, y, c=colors)
        subplot.set_title(title)
        subplot.set_xlabel('Runtime')
        subplot.set_ylabel('Price')
        subplot.invert_xaxis()
        subplot.invert_yaxis()

    incl_path = 'ext_dat/incl_cost/'
    csvs = os.listdir(incl_path)
    category_column_name = 'machine_list_or_failure_rate_or_cheap_or_fast_category'
    for csv in csvs:
        proj_name = csv.replace('.csv', '')
        dat = pd.read_csv(incl_path + csv).iloc[:, 1:]
        fig = plt.figure(figsize=(10, 4.5))
        fig.suptitle(proj_name, fontsize=14)
        chp_df = dat.loc[dat[category_column_name].str.contains('cheap')]
        fst_df = dat.loc[dat[category_column_name].str.contains('fast')]
        chp_x, chp_y = get_x_y(chp_df)
        fst_x, fst_y = get_x_y(fst_df)
        draw(fig.add_subplot(121), chp_x, chp_y, 'cheap')
        draw(fig.add_subplot(122), fst_x, fst_y, 'fast')
        plt.subplots_adjust(wspace=0.3)
        fig.savefig(f'{trade_off_path}/{proj_name}.png', dpi=1000)
        plt.close()


if __name__ == '__main__':
    # pareto_runtime_price('integration_dat_ga.csv')
    draw_trade_off()
