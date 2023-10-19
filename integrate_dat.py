import os
import numpy as np
import pandas as pd
from ast import literal_eval

beta = 25.993775 / 3600
frs = [0, 0.2, 0.4, 0.6, 0.8, 1]
alter_conf = '27CPU2Mem8GB.sh'


def record_df(df,
              idx,
              proj_name,
              chp=None,
              chp_gh=None,
              chp_smt=None,
              fst=None,
              fst_gh=None,
              fst_smt=None):
    def get_info(ser):
        return ser.loc['time_parallel'], ser.loc['price'], ser.loc['max_failure_rate']

    chp_single_conf_num = 0
    fst_single_conf_num = 0
    if chp is None:
        df.loc[len(df.index)] = [
            proj_name,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan
        ]
        return
    chp_runtime, chp_price, chp_fr = get_info(chp)
    chp_gh_runtime, chp_gh_price, chp_gh_fr = get_info(chp_gh)
    chp_smt_runtime, chp_smt_price, chp_smt_fr = get_info(chp_smt)
    fst_runtime, fst_price, fst_fr = get_info(fst)
    fst_gh_runtime, fst_gh_price, fst_gh_fr = get_info(fst_gh)
    fst_smt_runtime, fst_smt_price, fst_smt_fr = get_info(fst_smt)
    chp_gh_runtime_rate = chp_runtime / chp_gh_runtime
    chp_gh_price_rate = chp_price / chp_gh_price
    chp_smt_runtime_rate = chp_runtime / chp_smt_runtime
    chp_smt_price_rate = chp_price / chp_smt_price
    fst_gh_runtime_rate = fst_runtime / fst_gh_runtime
    fst_gh_price_rate = fst_price / fst_gh_price
    fst_smt_runtime_rate = fst_runtime / fst_smt_runtime
    fst_smt_price_rate = fst_price / fst_smt_price
    if len(literal_eval(chp['confs'])) == 1:
        chp_single_conf_num += 1
    if len(literal_eval(fst['confs'])) == 1:
        fst_single_conf_num += 1
    df.loc[len(df.index)] = [
        proj_name,
        chp['category'],
        chp['confs'],
        chp_runtime,
        chp_price,
        chp_fr,
        chp_gh['conf'],
        chp_gh_runtime,
        chp_gh_price,
        chp_gh_fr,
        chp_gh_runtime_rate,
        chp_gh_price_rate,
        chp_smt['conf'],
        chp_smt_runtime,
        chp_smt_price,
        chp_smt_fr,
        chp_smt_runtime_rate,
        chp_smt_price_rate,
        fst['category'],
        fst['confs'],
        fst_runtime,
        fst_price,
        fst_fr,
        fst_gh['conf'],
        fst_gh_runtime,
        fst_gh_price,
        fst_gh_fr,
        fst_gh_runtime_rate,
        fst_gh_price_rate,
        fst_smt['conf'],
        fst_smt_runtime,
        fst_smt_price,
        fst_smt_fr,
        fst_smt_runtime_rate,
        fst_smt_price_rate
    ]
    return chp_single_conf_num, fst_single_conf_num


def get_contrast(a,
                 subdir,
                 proj,
                 mach_num,
                 fr):
    df = pd.read_csv(f'baseline_dat/{subdir}/{proj}.csv')
    filter_dat = df.loc[df['num_machines'] == mach_num]
    filter_dat = (filter_dat.iloc[:, 1:]).reset_index(drop=True)
    github = filter_dat.loc[filter_dat['conf'] == alter_conf].iloc[0]
    filter_dat['fitness'] = a * (25.993775 / 3600) * filter_dat['time_parallel'] + (1 - a) * filter_dat['price']
    filter_dat.sort_values(by='fitness', inplace=True)
    smart = filter_dat.iloc[0, :]
    if smart['max_failure_rate'] > fr:
        filter_dat.sort_values(by='max_failure_rate', inplace=True)
        smart = filter_dat.iloc[0, :]
    return github, smart


def consider_fr(chp_dat_dir,
                fst_dat_dir,
                baseline_subdir,
                output_subdir,
                whe_mach6=False):
    df_num = 6
    tables = [
        'failrate_0',
        'failrate_0.2',
        'failrate_0.4',
        'failrate_0.6',
        'failrate_0.8',
        'failrate_1'
    ]
    fr_idx_map = {csv.split('_')[1]: idx for idx, csv in enumerate(tables)}
    dfs = [pd.DataFrame(None,
                        columns=['project',
                                 'cheapest_category',
                                 'cheapest_confs',
                                 'cheapest_runtime',
                                 'cheapest_price',
                                 'cheapest_max_failure_rate',
                                 'cheapest_github_baseline_conf',
                                 'cheapest_github_baseline_runtime',
                                 'cheapest_github_baseline_price',
                                 'cheapest_github_baseline_max_failure_rate',
                                 'cheapest_github_baseline_runtime_rate',
                                 'cheapest_github_baseline_price_rate',
                                 'cheapest_smart_baseline_conf',
                                 'cheapest_smart_baseline_runtime',
                                 'cheapest_smart_baseline_price',
                                 'cheapest_smart_baseline_max_failure_rate',
                                 'cheapest_smart_baseline_runtime_rate',
                                 'cheapest_smart_baseline_price_rate',
                                 'fastest_category',
                                 'fastest_confs',
                                 'fastest_runtime',
                                 'fastest_price',
                                 'fastest_max_failure_rate',
                                 'fastest_github_baseline_conf',
                                 'fastest_github_baseline_runtime',
                                 'fastest_github_baseline_price',
                                 'fastest_github_baseline_max_failure_rate',
                                 'fastest_github_baseline_runtime_rate',
                                 'fastest_github_baseline_price_rate',
                                 'fastest_smart_baseline_conf',
                                 'fastest_smart_baseline_runtime',
                                 'fastest_smart_baseline_price',
                                 'fastest_smart_baseline_max_failure_rate',
                                 'fastest_smart_baseline_runtime_rate',
                                 'fastest_smart_baseline_price_rate'])
           for _ in range(df_num)]
    output_dir = f'integ_dat/{output_subdir}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    chp_fs = os.listdir(chp_dat_dir)
    fst_fs = os.listdir(fst_dat_dir)
    a0_sum_single_conf = 0
    a1_sum_single_conf = 0
    for chp_f, fst_f in zip(chp_fs, fst_fs):
        if chp_f != fst_f:
            print(f'{chp_f} and {fst_f} are not matched!')
            break
        f = chp_f
        proj_name = f.replace('.csv', '')
        chp_df = pd.read_csv(f'{chp_dat_dir}/{f}')
        fst_df = pd.read_csv(f'{fst_dat_dir}/{f}')
        if whe_mach6:
            chp_arr = chp_df.iloc[0:24, 1:].dropna()
            fst_arr = fst_df.iloc[0:24, 1:].dropna()
        else:
            chp_arr = chp_df.iloc[:, 1:].dropna()
            fst_arr = fst_df.iloc[:, 1:].dropna()
        chp_fr_arr = [[] for _ in range(df_num)]
        fst_fr_arr = [[] for _ in range(df_num)]
        for i in range(len(chp_arr)):
            chp_itm = chp_arr.iloc[i, :]
            idx = fr_idx_map[chp_itm['category'].split('-')[1]]
            chp_fr_arr[idx].append(chp_itm)
            fst_fr_arr[idx].append(fst_arr.iloc[i, :])
        for i, arr in enumerate(zip(chp_fr_arr, fst_fr_arr)):
            chp = None
            fst = None
            if len(arr[0]) == 0:
                record_df(dfs[i],
                          i,
                          proj_name)
                continue
            chp_sort = sorted(arr[0], key=lambda x: x['price'])
            fst_sort = sorted(arr[1], key=lambda x: x['time_parallel'])
            chp_time = float('inf')
            chp_price = chp_sort[0]['price']
            fst_time = fst_sort[0]['time_parallel']
            fst_price = float('inf')
            for itm in chp_sort:
                if itm['price'] == chp_price:
                    if chp_time > itm['time_parallel']:
                        chp = itm
                        chp_time = itm['time_parallel']
                else:
                    break
            for itm in fst_sort:
                if itm['time_parallel'] == fst_time:
                    if fst_price > itm['price']:
                        fst = itm
                        fst_price = itm['price']
                else:
                    break
            chp_gh, chp_smt = get_contrast(0,
                                           baseline_subdir,
                                           proj_name,
                                           sum(literal_eval(chp['confs']).values()),
                                           0.2 * i)
            fst_gh, fst_smt = get_contrast(1,
                                           baseline_subdir,
                                           proj_name,
                                           sum(literal_eval(fst['confs']).values()),
                                           0.2 * i)
            chp_single_conf_num, fst_single_conf_num = record_df(dfs[i],
                                                                 i,
                                                                 proj_name,
                                                                 chp,
                                                                 chp_gh,
                                                                 chp_smt,
                                                                 fst,
                                                                 fst_gh,
                                                                 fst_smt)
            a0_sum_single_conf += chp_single_conf_num
            a1_sum_single_conf += fst_single_conf_num
    for i, df in enumerate(dfs):
        df.to_csv(f'{output_dir}/{tables[i]}.csv', sep=',', header=True, index=False)
    return a0_sum_single_conf, a1_sum_single_conf


def consider_ab(dat_dir,
                baseline_subdir,
                modu):
    output = f'integ_dat/{modu}.csv'
    df = pd.DataFrame(None,
                      columns=['project',
                               'category',
                               'confs',
                               'runtime',
                               'price',
                               'max_failure_rate',
                               'fitness',
                               'github_baseline_conf',
                               'github_baseline_runtime',
                               'github_baseline_price',
                               'github_baseline_max_failure_rate',
                               'github_baseline_score',
                               'github_baseline_runtime_rate',
                               'github_baseline_price_rate',
                               'github_baseline_score_rate',
                               'smart_baseline_conf',
                               'smart_baseline_runtime',
                               'smart_baseline_price',
                               'smart_baseline_max_failure_rate',
                               'smart_baseline_score',
                               'smart_baseline_runtime_rate',
                               'smart_baseline_price_rate',
                               'smart_baseline_score_rate'
                               ])
    fs = os.listdir(dat_dir)
    a = float(modu[modu.index('_') + 2:])
    b = 1 - a
    for f in fs:
        proj_name = f.replace('.csv', '')
        dat = pd.read_csv(f'{dat_dir}/{f}')
        dat = dat.loc[dat['category'].str.endswith('-0')].iloc[:, 1:].dropna()
        if dat.size == 0:
            continue
        dat.sort_values(by='fitness', inplace=True)
        itm = dat.iloc[0, :]
        gh, smt = get_contrast(a,
                               baseline_subdir,
                               proj_name,
                               sum(literal_eval(itm['confs']).values()),
                               0)
        gh_fit = a * beta * gh['time_parallel'] + b * gh['price']
        gh_fit_rate = itm['fitness'] / gh_fit
        smt_fit = a * beta * smt['time_parallel'] + b * smt['price']
        smt_fit_rate = itm['fitness'] / smt_fit
        df.loc[len(df.index)] = [
            proj_name,
            itm['category'],
            itm['confs'],
            itm['time_parallel'],
            itm['price'],
            itm['max_failure_rate'],
            itm['fitness'],
            gh['conf'],
            gh['time_parallel'],
            gh['price'],
            gh['max_failure_rate'],
            gh_fit,
            itm['time_parallel'] / gh['time_parallel'],
            itm['price'] / gh['price'],
            gh_fit_rate,
            smt['conf'],
            smt['time_parallel'],
            smt['price'],
            smt['max_failure_rate'],
            smt_fit,
            itm['time_parallel'] / smt['time_parallel'],
            itm['price'] / smt['price'],
            smt_fit_rate
        ]
    df.to_csv(f'{output}', sep=',', header=True, index=False)


if __name__ == '__main__':
    aes = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
           0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    print(consider_fr('ext_dat/ga_a0',
                      'ext_dat/ga_a1',
                      'non_ig',
                      'ga'))

    print(consider_fr('ext_dat/ga_a0_ig',
                      'ext_dat/ga_a1_ig',
                      'ig',
                      'ga_ig'))

    consider_fr('ext_dat/ga_a0',
                'ext_dat/ga_a1',
                'non_ig',
                'ga_mach6',
                True)

    consider_fr('ext_dat/bruteforce_a0',
                'ext_dat/bruteforce_a1',
                'non_ig',
                'bruteforce')
    for a in aes:
        consider_ab(f'ext_dat/ga_a{a}',
                    'non_ig',
                    f'ga_a{a}')
