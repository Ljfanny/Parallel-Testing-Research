import os
from ast import literal_eval
import pandas as pd
from plotter import fr0_satisfied_projs
from addi_analyze import proj_list

import warnings
warnings.filterwarnings("ignore")

comp_path = 'contrast'


def comp_ga_bf():
    idx_modu_map = {
        0: ['ga_a0', 'bruteforce_a0'],
        1: ['ga_a1', 'bruteforce_a1']
    }
    proj_id_map = {item['project-module']: item['id'] for _, item in pd.read_csv('proj_info.csv').iterrows()}
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
                                      'difference'
                                      ]) for _ in range(2)]
    summary_dfs = [pd.DataFrame(None,
                                columns=['project',
                                         'total_num',
                                         'avg_runtime_rate',
                                         'avg_price_rate',
                                         'sum_period_rate',
                                         'different_confs_num',
                                         'difference_rate'
                                         ]) for _ in range(2)]

    comp_opt_df = pd.DataFrame(None,
                               columns=[
                                   'project_id',
                                   'period_ga_cheap',
                                   'period_bf_cheap',
                                   'avg_runtime_rate_cheap',
                                   'avg_price_rate_cheap',
                                   'period_ga_fast',
                                   'period_bf_fast',
                                   'avg_runtime_rate_fast',
                                   'avg_price_rate_fast'
                               ])
    comp_opt_df['period_ga_fast'] = comp_opt_df['period_ga_fast'].astype('float')
    comp_opt_df['period_bf_fast'] = comp_opt_df['period_bf_fast'].astype('float')
    comp_opt_df['avg_runtime_rate_fast'] = comp_opt_df['avg_runtime_rate_fast'].astype('float')
    comp_opt_df['avg_price_rate_fast'] = comp_opt_df['avg_price_rate_fast'].astype('float')

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

    def reco_info(idx,
                  is_fir):
        subdir1 = idx_modu_map[idx][0]
        subdir2 = idx_modu_map[idx][1]
        for i, csv in enumerate(fr0_satisfied_projs):
            proj_name = csv
            ga = pd.read_csv(f'ext_dat/{subdir1}/{csv}.csv').iloc[:24, :].dropna()
            ga = ga.reset_index(drop=True)
            bf = pd.read_csv(f'ext_dat/{subdir2}/{csv}.csv').dropna()
            bf = bf.reset_index(drop=True)
            diff_cnt = 0
            tot_num = len(bf)
            tot_runtime_rate = 0
            tot_price_rate = 0
            ga_tot_period = 0
            bf_tot_period = 0
            for j in range(tot_num):
                ga_itm = ga.iloc[j, :]
                bf_itm = bf.iloc[j, :]
                diff = not cmp(literal_eval(ga_itm['confs']), literal_eval(bf_itm['confs']))
                ga_runtime, ga_price, ga_period = get_info(ga_itm)
                bf_runtime, bf_price, bf_period = get_info(bf_itm)
                runtime_rate = ga_runtime / bf_runtime
                price_rate = ga_price / bf_price
                tot_runtime_rate += runtime_rate
                tot_price_rate += price_rate
                ga_tot_period += ga_period
                bf_tot_period += bf_period
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
                    diff
                ]
            summary_dfs[idx].loc[len(summary_dfs[idx].index)] = [
                proj_name,
                tot_num,
                tot_runtime_rate / tot_num,
                tot_price_rate / tot_num,
                ga_tot_period / bf_tot_period,
                diff_cnt,
                diff_cnt / tot_num
            ]
            if is_fir:
                comp_opt_df.loc[len(comp_opt_df.index)] = [
                    proj_id_map[proj_name],
                    ga_tot_period,
                    bf_tot_period,
                    tot_runtime_rate / tot_num,
                    tot_price_rate / tot_num,
                    0,
                    0,
                    0,
                    0
                ]
            else:
                comp_opt_df.loc[i, 'period_ga_fast'] = ga_tot_period
                comp_opt_df.loc[i, 'period_bf_fast'] = bf_tot_period
                comp_opt_df.loc[i, 'avg_runtime_rate_fast'] = tot_runtime_rate / tot_num
                comp_opt_df.loc[i, 'avg_price_rate_fast'] = tot_price_rate / tot_num

    reco_info(0,
              True)
    reco_info(1,
              False)
    comp_dfs[0].to_csv(f'{comp_path}/ga_bf_a0.csv', sep=',', header=True, index=False)
    comp_dfs[1].to_csv(f'{comp_path}/ga_bf_a1.csv', sep=',', header=True, index=False)
    summary_dfs[0].to_csv(f'{comp_path}/ga_bf_a0_summary.csv', sep=',', header=True, index=False)
    summary_dfs[1].to_csv(f'{comp_path}/ga_bf_a1_summary.csv', sep=',', header=True, index=False)
    comp_opt_df.to_csv(f'{comp_path}/ga_bf_period_comp_per_proj.csv', sep=',', header=True, index=False,
                       float_format='%.2f')


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


def comp_ga_nor_gre():
    comp_df = pd.DataFrame(None,
                           columns=['project_id',
                                    'period_ga_cheap',
                                    'period_bf_cheap',
                                    'runtime_rate_cheap',
                                    'price_rate_cheap',
                                    'period_ga_fast',
                                    'period_bf_fast',
                                    'runtime_rate_fast',
                                    'price_rate_fast'])

    def extract_dat(a):
        ga_gre_df = pd.read_csv(f'ga_a{a}.csv')
        for i, proj in enumerate(proj_list):
            ga_df = pd.read_csv(f'ext_dat/ga_a{a}/{proj}.csv')
            ga = ga_df.loc[ga_df['category'] == '4-0']
            ga_gre = ga_gre_df.loc[ga_gre_df['project'] == proj]
            if a == 0:
                comp_df.loc[len(comp_df.index)] = [
                    proj,
                    '<1ms',
                    ga_gre.iloc[0]['period'],
                    ga.iloc[0]['time_parallel'] / ga_gre.iloc[0]['time_parallel'],
                    ga.iloc[0]['price'] / ga_gre.iloc[0]['price'],
                    '<1ms',
                    0,
                    0,
                    0
                ]
            else:
                comp_df.loc[i, 'period_bf_fast'] = ga_gre.iloc[0]['period']
                comp_df.loc[i, 'runtime_rate_fast'] = ga.iloc[0]['time_parallel'] / ga_gre.iloc[0]['time_parallel']
                comp_df.loc[i, 'price_rate_fast'] = ga.iloc[0]['price'] / ga_gre.iloc[0]['price']
    extract_dat(0)
    extract_dat(1)
    comp_df.to_csv('ga_vs_greedy.csv', sep=',', header=True, index=False)


if __name__ == '__main__':
    comp_ga_bf()
    comp_confs('non_ig', 'ig')
    comp_confs('ga', 'bf')
    # comp_ga_nor_gre()
