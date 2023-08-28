import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pareto_runtime_price(dat_path: str):
    confs_idx = 2
    runtime_idx = 4
    price_idx = 5
    max_fr_idx = 7
    plt.figure(figsize=(20, 40))
    csvs = os.listdir(dat_path)
    for i, csv in enumerate(csvs):
        subplot = plt.subplot(8, 4, i + 1)
        subplot.set_title(csv.replace('.csv', ''))
        # # (runtime,price): (max failure rate<minimum>,confs)
        mp = dict()
        df = pd.read_csv(dat_path + csv).iloc[:, 1:]
        for _, row in df.iterrows():
            key = (row[runtime_idx], row[price_idx])
            if key in mp.keys():
                max_fr = mp[key][0]
                if max_fr > row[max_fr_idx]:
                    mp[key] = (row[max_fr_idx], row[confs_idx])
            else:
                mp[key] = (row[max_fr_idx], row[confs_idx])
        keys = np.array(list(mp.keys()))
        x_runtimes = keys[:, 0]
        y_prices = keys[:, 1]
        subplot.plot(x_runtimes, y_prices, 'o')
        subplot.set_xlabel('Runtime')
        x_bias = (np.min(x_runtimes)+np.max(x_runtimes))/2
        y_bias = (np.min(y_prices)+np.max(y_prices))/2
        subplot.set_xlim(np.min(x_runtimes)-x_bias, np.max(x_runtimes)+x_bias)
        subplot.set_ylabel('Price')
        subplot.set_ylim(np.min(y_prices)-y_bias, np.max(y_prices)+y_bias)
        # for x, y in zip(x_runtimes, y_prices):
        #     subplot.text(x, y, round(mp[(x, y)][0], 2), ha='center')
    plt.show()


if __name__ == '__main__':
    pareto_runtime_price('ext_dat/incl_cost/')
