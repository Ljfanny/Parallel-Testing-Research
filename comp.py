import os
from ast import literal_eval
import numpy as np
import pandas as pd

comp_path = 'comparison/'


def comp_ga_bf(is_total=False):
    def rec_info(df,
                 star,
                 ga_obj,
                 bf_obj):
        gap_map = {'confs': 0, 'runtime': 1, 'price': 2, 'max_fr': 3}
        df.loc[len(df.index)] = [
            ga_obj[proj_idx],
            ga_obj[star + gap_map['confs']],
            ga_obj[star + gap_map['runtime']],
            ga_obj[star + gap_map['price']],
            ga_obj[star + gap_map['max_fr']],
            bf_obj[star + gap_map['confs']],
            bf_obj[star + gap_map['runtime']],
            bf_obj[star + gap_map['price']],
            bf_obj[star + gap_map['max_fr']],
            ga_obj[star + gap_map['runtime']] / bf_obj[star + gap_map['runtime']],
            ga_obj[star + gap_map['price']] / bf_obj[star + gap_map['price']],
            ga_obj[star + gap_map['confs']] == bf_obj[star + gap_map['confs']]
        ]

    def cmp(d1, d2):
        if len(d1) != len(d2):
            return False
        for k, v in d1.items():
            if k not in d2:
                return False
            if d2[k] != v:
                return False
        return True

    if not is_total:
        comp_dfs = [[pd.DataFrame(None, columns=['project',
                                                 'ga_confs',
                                                 'ga_runtime',
                                                 'ga_price',
                                                 'ga_max_failure_rate',
                                                 'bf_confs',
                                                 'bf_runtime',
                                                 'bf_price',
                                                 'bf_max_failure_rate',
                                                 'runtime_rate',
                                                 'price_rate',
                                                 'is_the_same']) for _ in range(2)] for _ in range(6)]
        fr_idx_map = {'0': 0, '0.2': 1, '0.4': 2, '0.6': 3, '0.8': 4, '1': 5}
        chp_fst_idx_map = {'cheap': 0, 'fast': 1}
        ga_path = 'integration_dat_incl_cost_limit/'
        bf_path = 'integration_dat_bruteforce/'
        ga_csvs = os.listdir(ga_path)
        bf_csvs = os.listdir(bf_path)
        name_fr_idx = 2
        proj_idx = 0
        chp_star = 2
        fst_star = 19
        for i in range(len(ga_csvs)):
            ga = pd.read_csv(ga_path + ga_csvs[i])
            bf = pd.read_csv(bf_path + bf_csvs[i])
            fr = ga_csvs[i].replace('.csv', '').split('_')[name_fr_idx]
            cnt = len(bf)
            for j in range(cnt):
                ga_itm = ga.loc[j]
                bf_itm = bf.loc[j]
                rec_info(comp_dfs[fr_idx_map[fr]][chp_fst_idx_map['cheap']],
                         chp_star,
                         ga_itm,
                         bf_itm)
                rec_info(comp_dfs[fr_idx_map[fr]][chp_fst_idx_map['fast']],
                         fst_star,
                         ga_itm,
                         bf_itm)
        for fr, fr_idx in fr_idx_map.items():
            for cho, cho_idx in chp_fst_idx_map.items():
                comp_dfs[fr_idx][cho_idx].to_csv(f'{comp_path}/comparison_{fr}_{cho}_of_ga_bf.csv',
                                                 sep=',',
                                                 header=True,
                                                 index=False)
    else:
        comp_df = pd.DataFrame(None,
                               columns=['project',
                                        'category',
                                        'ga_confs',
                                        'ga_runtime',
                                        'ga_price',
                                        'ga_max_failure_rate',
                                        'bf_confs',
                                        'bf_runtime',
                                        'bf_price',
                                        'bf_max_failure_rate',
                                        'runtime_rate',
                                        'price_rate',
                                        'is_the_same'])
        rec_diff_df = pd.DataFrame(None,
                                   columns=['project',
                                            'same_cheap_confs_number',
                                            'same_fast_confs_number',
                                            'total_number',
                                            'same_rate',
                                            'avg_cheap_runtime_rate',
                                            'avg_cheap_price_rate',
                                            'avg_fast_runtime_rate',
                                            'avg_fast_price_rate'
                                            ])
        proj_idx = 0
        category_idx = 1
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
            proj_name = ga_csvs[i].replace('.csv', '')
            ga = pd.read_csv(ga_path + ga_csvs[i])
            bf = pd.read_csv(bf_path + bf_csvs[i])
            cnt = len(bf)
            chp_same_cnt = 0
            fst_same_cnt = 0
            chp_diff_runtime_rate = 0.0
            chp_diff_price_rate = 0.0
            fst_diff_runtime_rate = 0.0
            fst_diff_price_rate = 0.0
            for j in range(cnt):
                ga_itm = ga.loc[j]
                bf_itm = bf.loc[j]
                is_the_same = (np.isnan(ga_itm[max_fr_idx]) and np.isnan(bf_itm[max_fr_idx])) or cmp(literal_eval(ga_itm[confs_idx]), literal_eval(bf_itm[confs_idx]))
                runtime_rate = ga_itm[runtime_idx] / bf_itm[runtime_idx]
                price_rate = ga_itm[price_idx] / bf_itm[price_idx]
                if 'cheap' in ga_itm[category_idx]:
                    if is_the_same:
                        chp_same_cnt += 1
                    else:
                        chp_diff_runtime_rate += runtime_rate
                        chp_diff_price_rate += price_rate
                else:
                    if is_the_same:
                        fst_same_cnt += 1
                    else:
                        fst_diff_runtime_rate += runtime_rate
                        fst_diff_price_rate += price_rate
                comp_df.loc[len(comp_df.index)] = [
                    proj_name,
                    ga_itm[category_idx],
                    ga_itm[confs_idx],
                    ga_itm[runtime_idx],
                    ga_itm[price_idx],
                    ga_itm[max_fr_idx],
                    bf_itm[confs_idx],
                    bf_itm[runtime_idx],
                    bf_itm[price_idx],
                    bf_itm[max_fr_idx],
                    runtime_rate,
                    price_rate,
                    is_the_same
                ]
            diff_cnt = cnt - chp_same_cnt - fst_same_cnt
            rec_diff_df.loc[len(rec_diff_df.index)] = [
                proj_name,
                chp_same_cnt,
                fst_same_cnt,
                cnt,
                (chp_same_cnt + fst_same_cnt) / cnt,
                chp_diff_runtime_rate / (24 - chp_same_cnt) if chp_same_cnt < 24 else np.nan,
                chp_diff_price_rate / (24 - chp_same_cnt) if chp_same_cnt < 24 else np.nan,
                fst_diff_runtime_rate / (24 - fst_same_cnt) if fst_same_cnt < 24 else np.nan,
                fst_diff_price_rate / (24 - fst_same_cnt) if fst_same_cnt < 24 else np.nan
            ]
        rec_diff_df.to_csv(f'{comp_path}diff_confs_case_number_ga_bf.csv', sep=',', header=True, index=False)
        comp_df.to_csv(f'{comp_path}comparison_dat_of_ga_bf.csv', sep=',', header=True, index=False)


def comp(a: str, b: str):
    temp = {'excl': 'excl_cost',
            'incl': 'incl_cost',
            'bf': 'bruteforce',
            'ga': 'incl_cost_limit'}
    chp_idx = 0
    fst_idx = 1
    name_fr_idx = 2
    proj_idx = 0
    chp_category_idx = 1
    chp_confs_idx = 2
    fst_category_idx = 18
    fst_confs_idx = 19
    comp_dfs = [pd.DataFrame(None, columns=['project_fr',
                                            f'{a}_category',
                                            f'{a}_confs',
                                            f'{b}_category',
                                            f'{b}_confs']) for _ in range(2)]
    csv_names = [f'confs_comparison_{a}_{b}_for_price.csv', f'confs_comparison_{a}_{b}_for_runtime.csv']
    a_path = f'integration_dat_{temp[a]}/'
    b_path = f'integration_dat_{temp[b]}/'
    a_csvs = sorted(os.listdir(a_path))
    b_csvs = sorted(os.listdir(b_path))
    for i in range(len(a_csvs)):
        fr = a_csvs[i].replace('.csv', '').split('_')[name_fr_idx]
        a_df = pd.read_csv(a_path + a_csvs[i])
        b_df = pd.read_csv(b_path + b_csvs[i])
        num = min(len(a_df), len(b_df))
        for j in range(num):
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
    comp_dfs[chp_idx].to_csv(f'{comp_path}/{csv_names[chp_idx]}', sep=',', header=True, index=False)
    comp_dfs[fst_idx].to_csv(f'{comp_path}/{csv_names[fst_idx]}', sep=',', header=True, index=False)


def comp_period():
    ga_path = 'ext_dat/incl_cost/'
    bf_path = 'bruteforce_dat/'
    ga_csvs = os.listdir(ga_path)
    bf_csvs = os.listdir(bf_path)
    output_path = 'comparison/'
    output_csv_name = 'comparison_period_of_ga_bf.csv'
    df = pd.DataFrame(None,
                      columns=[
                          'project',
                          'ga_period',
                          'bf_period',
                          'period_rate'
                      ])
    period_idx = 9
    for i in range(len(ga_csvs)):
        ga_df = pd.read_csv(f'{ga_path}{ga_csvs[i]}')
        bf_df = pd.read_csv(f'{bf_path}{bf_csvs[i]}')
        num = min(len(ga_df), len(bf_df))
        proj = ga_csvs[i].replace('.csv', '')
        ga_period = 0
        bf_period = 0
        for j in range(num):
            ga_period += ga_df.iloc[j, period_idx]
            bf_period += bf_df.iloc[j, period_idx]
        df.loc[len(df.index)] = [
            proj,
            ga_period,
            bf_period,
            ga_period / bf_period
        ]
    df.to_csv(f'{output_path}{output_csv_name}', sep=',', header=True, index=False)


if __name__ == '__main__':
    # comp_ga_bf()
    # comp_ga_bf(True)
    # comp('incl', 'excl')
    # comp('ga', 'bf')
    comp_period()
