import os
from ast import literal_eval
import numpy as np
import pandas as pd

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
        if ser.loc['max_failure_rate'] > frs[idx]:
            return np.nan, np.nan, np.nan
        return ser.loc['time_parallel'], ser.loc['price'], ser.loc['max_failure_rate']

    chp_nonnan_num = 0
    fst_nonnan_num = 0
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
    is_chp_gh = not np.isnan(chp_gh_fr)
    is_chp_smt = not np.isnan(chp_smt_fr)
    is_fst_gh = not np.isnan(fst_gh_fr)
    is_fst_smt = not np.isnan(fst_smt_fr)
    chp_gh_runtime_rate = chp_runtime / chp_gh_runtime
    chp_gh_price_rate = chp_price / chp_gh_price
    chp_smt_runtime_rate = chp_runtime / chp_smt_runtime
    chp_smt_price_rate = chp_price / chp_smt_price
    fst_gh_runtime_rate = fst_runtime / fst_gh_runtime
    fst_gh_price_rate = fst_price / fst_gh_price
    fst_smt_runtime_rate = fst_runtime / fst_smt_runtime
    fst_smt_price_rate = fst_price / fst_smt_price
    if len(literal_eval(chp['confs'])) == 1: chp_nonnan_num += 1
    if len(literal_eval(fst['confs'])) == 1: fst_nonnan_num += 1
    df.loc[len(df.index)] = [
        proj_name,
        chp['category'],
        chp['confs'],
        chp_runtime,
        chp_price,
        chp_fr,
        chp_gh['conf'] if is_chp_gh else np.nan,
        chp_gh_runtime,
        chp_gh_price,
        chp_gh_fr,
        chp_gh_runtime_rate,
        chp_gh_price_rate,
        chp_smt['conf'] if is_chp_smt else np.nan,
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
        fst_gh['conf'] if is_fst_gh else np.nan,
        fst_gh_runtime,
        fst_gh_price,
        fst_gh_fr,
        fst_gh_runtime_rate,
        fst_gh_price_rate,
        fst_smt['conf'] if is_fst_smt else np.nan,
        fst_smt_runtime,
        fst_smt_price,
        fst_smt_fr,
        fst_smt_runtime_rate,
        fst_smt_price_rate
    ]
    idx_proj_num_map[idx] += 1
    if is_chp_gh:
        idx_non_nan_baseline_num_map[idx][0] += 1
        summary_arr[idx][0] += chp_gh_runtime_rate
        summary_arr[idx][1] += chp_gh_price_rate
        summary_arr[idx][2] += 0 if chp_gh_price_rate >= 1 else 1
    if is_chp_smt:
        idx_non_nan_baseline_num_map[idx][1] += 1
        summary_arr[idx][3] += chp_smt_runtime_rate
        summary_arr[idx][4] += chp_smt_price_rate
        summary_arr[idx][5] += 0 if chp_smt_price_rate >= 1 else 1
    if is_fst_gh:
        idx_non_nan_baseline_num_map[idx][2] += 1
        summary_arr[idx][6] += fst_gh_runtime_rate
        summary_arr[idx][7] += fst_gh_price_rate
        summary_arr[idx][8] += 0 if fst_gh_runtime_rate >= 1 else 1
    if is_fst_smt:
        idx_non_nan_baseline_num_map[idx][3] += 1
        summary_arr[idx][9] += fst_smt_runtime_rate
        summary_arr[idx][10] += fst_smt_price_rate
        summary_arr[idx][11] += 0 if fst_smt_runtime_rate >= 1 else 1
    return chp_nonnan_num, fst_nonnan_num


def get_contrast(subdir,
                 proj,
                 mach_num):
    df = pd.read_csv(f'baseline_dat/{subdir}/{proj}.csv')
    filter_dat = df.loc[df['num_machines'] == mach_num]
    filter_dat = (filter_dat.iloc[:, 1:]).reset_index(drop=True)
    github_caliber = filter_dat.loc[filter_dat['conf'] == alter_conf].iloc[0]
    filter_dat.sort_values(by='max_failure_rate', inplace=True)
    return github_caliber, filter_dat.iloc[0, :]


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
    summary_df = pd.DataFrame(None,
                              columns=[
                                  'table_category',
                                  'non_nan_project_num',
                                  'cheapest_non_nan_github_caliber_num',
                                  'cheapest_avg_ga_vs_github_caliber_runtime_rate',
                                  'cheapest_avg_ga_vs_github_caliber_price_rate',
                                  'cheapest_ga_vs_github_caliber_better_num',
                                  'cheapest_non_nan_smart_baseline_num',
                                  'cheapest_avg_ga_vs_smart_baseline_runtime_rate',
                                  'cheapest_avg_ga_vs_smart_baseline_price_rate',
                                  'cheapest_ga_vs_smart_baseline_better_num',
                                  'fastest_non_nan_github_caliber_num',
                                  'fastest_avg_ga_vs_github_caliber_runtime_rate',
                                  'fastest_avg_ga_vs_github_caliber_price_rate',
                                  'fastest_ga_vs_github_caliber_better_num',
                                  'fastest_non_nan_smart_baseline_num',
                                  'fastest_avg_ga_vs_smart_baseline_runtime_rate',
                                  'fastest_avg_ga_vs_smart_baseline_price_rate',
                                  'fastest_ga_vs_smart_baseline_better_num',
                              ])
    dfs = [pd.DataFrame(None,
                        columns=['project',
                                 'cheapest_category',
                                 'cheapest_confs',
                                 'cheapest_runtime',
                                 'cheapest_price',
                                 'cheapest_max_failure_rate',
                                 'cheapest_github_caliber_conf',
                                 'cheapest_github_caliber_runtime',
                                 'cheapest_github_caliber_price',
                                 'cheapest_github_caliber_max_failure_rate',
                                 'cheapest_github_caliber_runtime_rate',
                                 'cheapest_github_caliber_price_rate',
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
                                 'fastest_github_caliber_conf',
                                 'fastest_github_caliber_runtime',
                                 'fastest_github_caliber_price',
                                 'fastest_github_caliber_max_failure_rate',
                                 'fastest_github_caliber_runtime_rate',
                                 'fastest_github_caliber_price_rate',
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
    a0_sum_nonnan = 0
    a1_sum_nonnan = 0
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
            chp_gh, chp_smt = get_contrast(baseline_subdir,
                                           proj_name,
                                           sum(literal_eval(chp['confs']).values()))
            fst_gh, fst_smt = get_contrast(baseline_subdir,
                                           proj_name,
                                           sum(literal_eval(fst['confs']).values()))
            a0_nonnan_num, a1_nonnan_num = record_df(dfs[i],
                                                     i,
                                                     proj_name,
                                                     chp,
                                                     chp_gh,
                                                     chp_smt,
                                                     fst,
                                                     fst_gh,
                                                     fst_smt)
            a0_sum_nonnan += a0_nonnan_num
            a1_sum_nonnan += a1_nonnan_num
    for i, df in enumerate(dfs):
        df.to_csv(f'{output_dir}/{tables[i]}.csv', sep=',', header=True, index=False)
        proj_num = idx_proj_num_map[i]
        summary_df.loc[len(summary_df.index)] = [
            tables[i],
            proj_num,
            idx_non_nan_baseline_num_map[i][0],
            summary_arr[i][0] / proj_num,
            summary_arr[i][1] / proj_num,
            summary_arr[i][2],
            idx_non_nan_baseline_num_map[i][1],
            summary_arr[i][3] / proj_num,
            summary_arr[i][4] / proj_num,
            summary_arr[i][5],
            idx_non_nan_baseline_num_map[i][2],
            summary_arr[i][6] / proj_num,
            summary_arr[i][7] / proj_num,
            summary_arr[i][8],
            idx_non_nan_baseline_num_map[i][3],
            summary_arr[i][9] / proj_num,
            summary_arr[i][10] / proj_num,
            summary_arr[i][11]
        ]
    summary_df.to_csv(f'{output_dir}/summary.csv', sep=',', header=True, index=False)
    return a0_sum_nonnan, a1_sum_nonnan


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
                               'github_caliber_conf',
                               'github_caliber_runtime',
                               'github_caliber_price',
                               'github_caliber_max_failure_rate',
                               'github_caliber_score',
                               'github_caliber_runtime_rate',
                               'github_caliber_price_rate',
                               'github_caliber_score_rate',
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
        gh, smt = get_contrast(baseline_subdir,
                               proj_name,
                               sum(literal_eval(itm['confs']).values()))
        is_gh = False if gh['max_failure_rate'] > 0 else True
        is_smt = False if smt['max_failure_rate'] > 0 else True
        gh_score = np.nan
        gh_rate = np.nan
        smt_score = np.nan
        smt_rate = np.nan
        if is_gh:
            gh_score = a * beta * gh['time_parallel'] + b * gh['price']
            gh_rate = itm['fitness'] / gh_score
        if is_smt:
            smt_score = a * beta * smt['time_parallel'] + b * smt['price']
            smt_rate = itm['fitness'] / smt_score
        df.loc[len(df.index)] = [
            proj_name,
            itm['category'],
            itm['confs'],
            itm['time_parallel'],
            itm['price'],
            itm['max_failure_rate'],
            itm['fitness'],
            gh['conf'] if is_gh else np.nan,
            gh['time_parallel'] if is_gh else np.nan,
            gh['price'] if is_gh else np.nan,
            gh['max_failure_rate'] if is_gh else np.nan,
            gh_score,
            itm['time_parallel'] / gh['time_parallel'] if is_gh else np.nan,
            itm['price'] / gh['price'] if is_gh else np.nan,
            gh_rate,
            smt['conf'] if is_smt else np.nan,
            smt['time_parallel'] if is_smt else np.nan,
            smt['price'] if is_smt else np.nan,
            smt['max_failure_rate'] if is_smt else np.nan,
            smt_score,
            itm['time_parallel'] / smt['time_parallel'] if is_smt else np.nan,
            itm['price'] / smt['price'] if is_smt else np.nan,
            smt_rate
        ]
    df.to_csv(f'{output}', sep=',', header=True, index=False)


def consider_per_proj(subdir,
                      prefix,
                      goal_csv):
    fr_tables = os.listdir(f'integ_dat/{subdir}')
    fr_dfs = [pd.read_csv(f'integ_dat/{subdir}/{fr_tb}') for fr_tb in fr_tables if fr_tb.find('summary') == -1]
    projs = fr_dfs[0].iloc[:, 0]
    summary_per_proj = pd.DataFrame(None,
                                    columns=[
                                        'project',
                                        'github_caliber_avg_runtime_rate',
                                        'github_caliber_avg_price_rate',
                                        'smart_baseline_avg_runtime_rate',
                                        'smart_baseline_avg_price_rate'
                                    ])
    for i, proj in enumerate(projs):
        gh_sum_runtime_rts = []
        smt_sum_runtime_rts = []
        gh_sum_price_rts = []
        smt_sum_price_rts = []
        for fr_df in fr_dfs:
            itm = fr_df.iloc[i, :]
            gh_sum_runtime_rts.append(itm[f'{prefix}_github_caliber_runtime_rate'])
            smt_sum_runtime_rts.append(itm[f'{prefix}_smart_baseline_runtime_rate'])
            gh_sum_price_rts.append(itm[f'{prefix}_github_caliber_price_rate'])
            smt_sum_price_rts.append(itm[f'{prefix}_smart_baseline_price_rate'])
        gh_avg_runtime_rate = np.nanmean(np.array(gh_sum_runtime_rts))
        smt_avg_runtime_rate = np.nanmean(np.array(smt_sum_runtime_rts))
        gh_avg_price_rate = np.nanmean(np.array(gh_sum_price_rts))
        smt_avg_price_rate = np.nanmean(np.array(smt_sum_price_rts))
        summary_per_proj.loc[len(summary_per_proj.index)] = [
            proj,
            gh_avg_runtime_rate,
            gh_avg_price_rate,
            smt_avg_runtime_rate,
            smt_avg_price_rate
        ]
    summary_per_proj.to_csv(f'integ_dat/{subdir}/{goal_csv}',
                            sep=',', header=True, index=False)


def get_avg_min_max_failrate():
    chp_pfx = 'cheapest'
    fst_pfx = 'fastest'
    gh_md = 'github_caliber'
    smt_md = 'smart_baseline'
    fr_col_nm = 'max_failure_rate'
    fr0_df = pd.read_csv('integ_dat/ga/failrate_0.csv').dropna(subset=[f'{chp_pfx}_category'])
    gh_frs = []
    smt_frs = []
    for index, low in fr0_df.iterrows():
        chp_gh, chp_smt = get_contrast('non_ig',
                                       low['project'],
                                       sum(literal_eval(low[f'{chp_pfx}_confs']).values()))
        fst_gh, fst_smt = get_contrast('non_ig',
                                       low['project'],
                                       sum(literal_eval(low[f'{fst_pfx}_confs']).values()))
        if np.isnan(low[f'{chp_pfx}_{gh_md}_{fr_col_nm}']):
            gh_frs.append(chp_gh[f'{fr_col_nm}'])
        if np.isnan(low[f'{fst_pfx}_{gh_md}_{fr_col_nm}']):
            gh_frs.append(fst_gh[f'{fr_col_nm}'])
        if np.isnan(low[f'{chp_pfx}_{smt_md}_{fr_col_nm}']):
            smt_frs.append(chp_smt[f'{fr_col_nm}'])
        if np.isnan(low[f'{fst_pfx}_{smt_md}_{fr_col_nm}']):
            smt_frs.append(fst_smt[f'{fr_col_nm}'])
    print(f'GitHub caliber avg. max failure rate: {np.mean(np.array(gh_frs))}')
    print(f'Smart baseline avg. max failure rate: {np.mean(np.array(smt_frs))}')


def get_same_conf_num():
    subdir = ['ga_a0', 'ga_a1', 'ga_a0_ig', 'ga_a1_ig']
    resu = [0, 0, 0, 0]
    tot = [0, 0, 0, 0]
    reco_df = pd.DataFrame(None,
                           columns=['project',
                                    'category'])
    for i, sub in enumerate(subdir):
        proj_csvs = os.listdir(f'ext_dat/{sub}')
        for csv in proj_csvs:
            proj = csv.replace('.csv', '')
            df = pd.read_csv(f'ext_dat/{sub}/{csv}').dropna()
            tot[i] += len(df)
            for _, item in df.iterrows():
                category = item['category']
                confs = literal_eval(item['confs'])
                if len(confs) == 1:
                    reco_df.loc[len(reco_df.index)] = [proj,
                                                       category]
                    resu[i] += 1
    print(f'GA with setup cost for a=0: {resu[0]}/{tot[0]}')
    print(f'GA with setup cost for a=1: {resu[1]}/{tot[1]}')
    print(f'GA without setup cost for a=0: {resu[2]}/{tot[2]}')
    print(f'GA without setup cost for a=0: {resu[3]}/{tot[3]}')
    reco_df.to_csv(f'same_confs_info.csv', sep=',', header=True, index=False)


if __name__ == '__main__':
    is_ab = False
    aes = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
           0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    modus = [f'ga_a{a}' for a in aes]
    if not is_ab:
        summary_arr = np.zeros((6, 12))
        idx_proj_num_map = {i: 0 for i in range(6)}
        idx_non_nan_baseline_num_map = {i: np.zeros(4) for i in range(6)}
    # Note: a0 = cheap; a1 = fast
    # print(consider_fr('ext_dat/ga_a0',
    #                   'ext_dat/ga_a1',
    #                   'non_ig',
    #                   'ga'))
    # consider_fr('ext_dat/ga_a0',
    #             'ext_dat/ga_a1',
    #             'non_ig',
    #             'ga_mach6',
    #             True)
    # consider_fr('ext_dat/bruteforce_a0',
    #             'ext_dat/bruteforce_a1',
    #             'non_ig',
    #             'bruteforce')
    print(consider_fr('ext_dat/ga_a0_ig',
                      'ext_dat/ga_a1_ig',
                      'ig',
                      'ga_ig'))
    # for md in modus:
    #     consider_ab(f'ext_dat/{md}',
    #                 'non_ig',
    #                 md)
    # consider_per_proj('ga',
    #                   'cheapest',
    #                   'summary_per_project_lower_price_goal.csv')
    # consider_per_proj('ga',
    #                   'fastest',
    #                   'summary_per_project_lower_runtime_goal.csv')
    # consider_per_proj('ga_ig',
    #                   'cheapest',
    #                   'summary_per_project_lower_price_goal.csv')
    # consider_per_proj('ga_ig',
    #                   'fastest',
    #                   'summary_per_project_lower_runtime_goal.csv')
    # get_avg_min_max_failrate()
    # get_same_conf_num()
