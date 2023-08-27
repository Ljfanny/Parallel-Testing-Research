import os

import pandas as pd


def comp_ga_bf():
    comp_df = pd.DataFrame(None,
                           columns=['project',
                                    'category',
                                    'GA_confs',
                                    'GA_runtime',
                                    'GA_price',
                                    'GA_max_failure_rate',
                                    'BF_confs',
                                    'BF_runtime',
                                    'BF_price',
                                    'BF_max_failure_rate',
                                    'runtime_rate(GA/BF)',
                                    'price_rate(GA/BF)'])
    confs_idx = 3
    runtime_idx = 5
    price_idx = 6
    max_fr_idx = 8
    ga_path = 'ext_dat/incl_cost/'
    bf_path = 'bruteforce_dat/'
    ga_csvs = os.listdir(ga_path)
    bf_csvs = os.listdir(bf_path)
    csv_num = len(bf_csvs)

    for i in range(csv_num):
        ga = pd.read_csv(ga_path + ga_csvs[i])
        bf = pd.read_csv(bf_path + bf_csvs[i])
        cnt = len(bf)
        for j in range(cnt):
            ga_itm = ga.loc[j]
            bf_itm = bf.loc[j]
            comp_df.loc[len(comp_df.index)] = [
                ga_itm[0],
                ga_itm[1],
                ga_itm[confs_idx],
                ga_itm[runtime_idx],
                ga_itm[price_idx],
                ga_itm[max_fr_idx],
                bf_itm[confs_idx],
                bf_itm[runtime_idx],
                bf_itm[price_idx],
                bf_itm[max_fr_idx],
                ga_itm[runtime_idx] / bf_itm[runtime_idx],
                ga_itm[price_idx] / bf_itm[price_idx]
            ]
    comp_df.to_csv('comparison.csv', sep=',', index=False, header=True)


def comp(a: str, b: str):
    temp = {'excl': 'excl_cost',
            'incl': 'incl_cost',
            'bruteforce': 'bruteforce',
            'ga': 'incl_cost'}
    chp_idx = 0
    fst_idx = 1

    fr_idx = 4
    proj_idx = 0
    chp_category_idx = 1
    chp_confs_idx = 2
    fst_category_idx = 10
    fst_confs_idx = 11
    comp_dfs = [pd.DataFrame(None, columns=['project_fr',
                                            f'{a}_category',
                                            f'{a}_confs',
                                            f'{b}_category',
                                            f'{b}_confs']) for _ in range(2)]
    csv_names = [f'comparison_price_of_{a}_{b}.csv', f'comparison_runtime_of_{a}_{b}.csv']
    a_path = f'integration_dat_{temp[a]}/'
    b_path = f'integration_dat_{temp[b]}/'
    a_csvs = sorted(os.listdir(a_path))
    b_csvs = sorted(os.listdir(b_path))
    for i in range(len(a_csvs)):
        fr = a_csvs[i].replace('.csv','').split('-')[fr_idx]
        a_df = pd.read_csv(a_path + a_csvs[i])
        b_df = pd.read_csv(b_path + b_csvs[i])
        for j in range(len(a_df)):
            a_itm = a_df.loc[j]
            b_itm = b_df.loc[j]
            comp_dfs[chp_idx].loc[len(comp_dfs[chp_idx].index)] = [
                a_itm[proj_idx] + '-' + fr,
                a_itm[chp_category_idx],
                a_itm[chp_confs_idx],
                b_itm[chp_category_idx],
                b_itm[chp_confs_idx]
            ]
            comp_dfs[fst_idx].loc[len(comp_dfs[fst_idx].index)] = [
                a_itm[proj_idx] + '-' + fr,
                a_itm[fst_category_idx],
                a_itm[fst_confs_idx],
                b_itm[fst_category_idx],
                b_itm[fst_confs_idx]
            ]
    comp_dfs[chp_idx].to_csv(csv_names[chp_idx], sep=',', index=False, header=True)
    comp_dfs[fst_idx].to_csv(csv_names[fst_idx], sep=',', index=False, header=True)


if __name__ == '__main__':
    comp('incl', 'excl')
    comp('bruteforce', 'ga')
