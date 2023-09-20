import os
from ast import literal_eval
import numpy as np
import pandas as pd

comp_path = 'contrast'


def comp(idx):
    idx_modu_map = {
        0: ['ga_a0', 'bruteforce_a0'],
        1: ['ga_a1', 'bruteforce_a1'],
        2: ['ga_a0', 'ga_a0_ig'],
        3: ['ga_a1', 'ga_a1_ig']
    }
    comp_dfs = [pd.DataFrame(None,
                             columns=['project',
                                      'category',
                                      'ga_confs',
                                      'ga_runtime',
                                      'ga_price',
                                      'ga_max_failure_rate',
                                      'ga_period',
                                      'bf_confs',
                                      'bf_runtime',
                                      'bf_price',
                                      'bf_max_failure_rate',
                                      'bf_period',
                                      'runtime_rate',
                                      'price_rate',
                                      'period_rate',
                                      'difference'
                                      ]) for _ in range(2)]
    summary_df = pd.DataFrame(None,
                              columns=['project',
                                       'total_num',
                                       'avg_runtime_rate',
                                       'avg_price_rate',
                                       'avg_period_rate',
                                       'different_confs_num',
                                       'difference_rate'
                                       ])

    def cmp(d1, d2):
        if len(d1) != len(d2):
            return False
        for k, v in d1.items():
            if k not in d2:
                return False
            if d2[k] != v:
                return False
        return True

    def get_info(ser):
        return ser['time_parallel'], ser['price'], ser['period']

    def reco_info(idx):
        subdir1 = idx_modu_map[idx][0]
        subdir2 = idx_modu_map[idx][0]
        csvs = os.listdir(f'ext_dat/{subdir1}')
        for csv in csvs:
            proj_name = csv.replace('.csv', '')
            ga = pd.read_csv(f'ext_dat/{subdir1}/{csv}').iloc[:24, :].dropna()
            bf = pd.read_csv(f'ext_dat/{subdir2}/{csv}').dropna()
            diff_cnt = 0
            tot_num = len(bf)
            tot_runtime_rate = 0
            tot_price_rate = 0
            tot_period_rate = 0
            for j in range(tot_num):
                ga_itm = ga.iloc[j, :]
                bf_itm = bf.iloc[j, :]
                diff = not cmp(literal_eval(ga_itm['confs']), literal_eval(bf_itm['confs']))
                ga_runtime, ga_price, ga_period = get_info(ga_itm)
                bf_runtime, bf_price, bf_period = get_info(bf_itm)
                runtime_rate = ga_runtime / bf_runtime
                price_rate = ga_price / bf_price
                period_rate = ga_period / bf_period
                if diff:
                    diff_cnt += 1
                comp_dfs[idx].loc[len(comp_dfs[idx].index)] = [
                    proj_name,
                    ga_itm['category'],
                    ga_itm['confs'],
                    ga_runtime,
                    ga_price,
                    ga_itm['max_failure_rate'],
                    ga_period,
                    bf_itm['confs'],
                    bf_runtime,
                    bf_price,
                    bf_itm['max_failure_rate'],
                    bf_period,
                    runtime_rate,
                    price_rate,
                    period_rate,
                    diff
                ]
            summary_df.loc[len(summary_df.index)] = [
                proj_name,
                tot_num,
                tot_runtime_rate / tot_num,
                tot_price_rate / tot_num,
                tot_period_rate / tot_num,
                diff_cnt,
                diff_cnt / tot_num
            ]

    reco_info(0)
    reco_info(1)
    comp_dfs[0].to_csv(f'{comp_path}/ga_bf_a0.csv', sep=',', header=True, index=False)
    comp_dfs[1].to_csv(f'{comp_path}/ga_bf_a1.csv', sep=',', header=True, index=False)
    summary_df.to_csv(f'{comp_path}/ga_bf_summary.csv', sep=',', header=True, index=False)


def comp_confs(a: str, b: str):
    id_subdir_map = {
        'ig': 'ga_ig',
        'non_ig': 'ga',
        'bf': 'bruteforce',
        'ga': 'ga_mach6'
    }
    comp_df = pd.DataFrame(None, columns=['project_fr',
                                          f'cheapest_{a}_category',
                                          f'cheapest_{a}_confs',
                                          f'cheapest_{b}_category',
                                          f'cheapest_{b}_confs',
                                          f'fastest_{a}_category',
                                          f'fastest_{a}_confs',
                                          f'fastest_{b}_category',
                                          f'fastest_{b}_confs'])
    a_path = f'integ_dat/{id_subdir_map[a]}'
    b_path = f'integ_dat/{id_subdir_map[b]}'
    csvs = os.listdir(a_path)
    for csv in csvs:
        if csv.find('_') == -1:
            continue
        fr = csv.replace('.csv', '').split('_')[1]
        a_df = pd.read_csv(f'{a_path}/{csv}').dropna()
        b_df = pd.read_csv(f'{b_path}/{csv}').dropna()
        num = len(a_df)
        for j in range(num):
            a_itm = a_df.iloc[j, :]
            b_itm = b_df.iloc[j, :]
            proj_name = a_itm['project']
            comp_df.loc[len(comp_df.index)] = [
                f'{proj_name}_{fr}',
                a_itm['cheapest_category'],
                a_itm['cheapest_confs'],
                b_itm['cheapest_category'],
                b_itm['cheapest_confs'],
                a_itm['fastest_category'],
                a_itm['fastest_confs'],
                b_itm['fastest_category'],
                b_itm['fastest_confs']
            ]
    comp_df.to_csv(f'{comp_path}/confs_{a}_{b}.csv', sep=',', header=True, index=False)


if __name__ == '__main__':
    comp()
    # comp_confs('non_ig', 'ig')
    # comp_confs('ga', 'bf')
