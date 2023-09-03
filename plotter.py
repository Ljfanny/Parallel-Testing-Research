import os
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
import numpy as np

pareto_2d_path = 'pareto_2d'
pareto_3d_path = 'pareto_3d'


def pareto_frontier_multi(arr,
                          major):
    major_map = {'runtime': 0,
                 'price': 1,
                 'max_fr': 2}
    col = major_map[major]
    # Sort on first dimension
    arr = arr[arr[:, col].argsort()[::-1]]
    # Add first row to pareto_frontier
    pareto_frontiers = arr[0:1, :]
    # Test next row against the last row in pareto_frontier
    for row in arr[1:, :]:
        if sum([row[x] <= pareto_frontiers[-1][x] for x in range(len(row))]) == len(row):
            # If it is better on all features add the row to pareto_frontier
            pareto_frontiers = np.concatenate((pareto_frontiers, [row]))
    return pareto_frontiers


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
        # (runtime, price): max failure rate
        mp = {}
        for _, itm in df.iterrows():
            if np.isnan(itm[max_fr_idx]):
                continue
            key = (itm[runtime_idx], itm[price_idx])
            if key not in mp.keys() or mp[key] > itm[max_fr_idx]:
                mp[key] = itm[max_fr_idx]
        runtime_price_tup = np.array(list(mp.keys()))
        xyz = np.column_stack((runtime_price_tup,
                               np.array(list(mp.values()))))
        frontiers = pareto_frontier_multi(xyz,
                                          'runtime')
        x_front = frontiers[:, 0]
        y_front = frontiers[:, 1]
        z_front = frontiers[:, 2]
        # comment = ''
        for node in frontiers:
            mp.pop((node[0], node[1]))
            # comment = comment + f'x={node[0]}, y={node[1]}, z={node[2]}\n'
        keys = np.array(list(mp.keys()))
        x_non = keys[:, 0]
        y_non = keys[:, 1]
        z_non = np.array(list(mp.values()))
        ax.scatter(x_non, y_non, z_non,
                   s=75, c='royalblue')
        ax.scatter(x_front, y_front, z_front,
                   s=75, c='r', label='Pareto frontiers')
        ax.set_title(textwrap.fill(proj_name),
                     fontsize=14)
        ax.set_xlabel('Runtime')
        ax.set_ylabel('Price')
        ax.set_zlabel('Max failure rate')
        # ax.text(0, 0, 0, comment)
        ax.legend()
        plt.savefig(f'{pareto_3d_path}/{proj_name}.png',
                    dpi=1000)
        plt.close()


def draw_pareto_2d(dat_name,
                   fr):
    if not os.path.exists(f'{pareto_2d_path}/failure_rate_{fr}'):
        os.mkdir(f'{pareto_2d_path}/failure_rate_{fr}')
    proj_idx = 0
    colors = ['mediumpurple', 'thistle', 'lightsteelblue', 'darkseagreen', 'slategrey', 'darkkhaki']
    biases = {'chp': 0, 'chp_gh': 4, 'chp_smt': 10, 'fst': 17, 'fst_gh': 21, 'fst_smt': 27}
    nodes = {'chp': 'cheapest', 'chp_gh': 'cheapest_github_caliber', 'chp_smt': 'cheapest_smart_baseline',
             'fst': 'fastest', 'fst_gh': 'fastest_github_caliber', 'fst_smt': 'fastest_smart_baseline'}
    runtime_idx = 3
    price_idx = 4
    max_fr_idx = 5
    dat = pd.read_csv(dat_name)
    for _, row in dat.iterrows():
        proj_name = row[proj_idx]
        plt.title(textwrap.fill(proj_name),
                  fontsize=14)
        if np.isnan(row[max_fr_idx]):
            plt.savefig(f'{pareto_2d_path}/failure_rate_{fr}/{proj_name}.png',
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
        plt.scatter(x_runtime, y_price,
                    s=75, c=colors)
        plt.xlabel('Runtime')
        plt.ylabel('Price')
        plt.legend(handles=[plt.scatter([], [], c=c) for c in colors],
                   labels=labels)
        plt.savefig(f'{pareto_2d_path}/failure_rate_{fr}/{proj_name}.png',
                    dpi=1000)
        plt.close()


if __name__ == '__main__':
    # draw_pareto_2d('integration_dat_incl_cost/failure_rate_1.csv', 1)
    # draw_pareto_2d('integration_dat_incl_cost/failure_rate_0.csv', 0)
    draw_pareto_3d('incl')
