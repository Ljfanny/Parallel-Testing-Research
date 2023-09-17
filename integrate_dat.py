import os
from ast import literal_eval
import numpy as np
import pandas as pd

category_idx = 0
confs_idx = 2
time_idx = 4
price_idx = 5
max_fr_idx = 7
frs = [0, 0.2, 0.4, 0.6, 0.8, 1]
alter_conf = '27CPU2Mem8GB.sh'
baseline_paths = [
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/excl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/', 'baseline_dat/incl_cost/',
    'baseline_dat/incl_cost/'
]
dat_paths = [
    'bruteforce_dat/', 'ext_dat/incl_cost/',
    'ext_dat/excl_cost/', 'ext_dat/ga_a0.1b0.9/',
    'ext_dat/ga_a0.2b0.8/', 'ext_dat/ga_a0.3b0.7/',
    'ext_dat/ga_a0.4b0.6/', 'ext_dat/ga_a0.5b0.5/',
    'ext_dat/ga_a0.6b0.4/', 'ext_dat/ga_a0.7b0.3/',
    'ext_dat/ga_a0.8b0.2/', 'ext_dat/ga_a0.9b0.1/',
    'ext_dat/ga_a0.01b0.99/', 'ext_dat/ga_a0.02b0.98/',
    'ext_dat/ga_a0.03b0.97/', 'ext_dat/ga_a0.04b0.96/',
    'ext_dat/ga_a0.05b0.95/', 'ext_dat/ga_a0.06b0.94/',
    'ext_dat/ga_a0.07b0.93/', 'ext_dat/ga_a0.08b0.92/',
    'ext_dat/ga_a0.09b0.91/'
]
output_paths = [
    'integration_bruteforce/', 'integration_incl_cost/',
    'integration_excl_cost/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/', 'integration_ga_with_factors/',
    'integration_ga_with_factors/'
]
modus = {
    'bruteforce': 0,
    'incl': 1,
    'excl': 2,
    'a0.1b0.9': 3,
    'a0.2b0.8': 4,
    'a0.3b0.7': 5,
    'a0.4b0.6': 6,
    'a0.5b0.5': 7,
    'a0.6b0.4': 8,
    'a0.7b0.3': 9,
    'a0.8b0.2': 10,
    'a0.9b0.1': 11,
    'a0.01b0.99': 12,
    'a0.02b0.98': 13,
    'a0.03b0.97': 14,
    'a0.04b0.96': 15,
    'a0.05b0.95': 16,
    'a0.06b0.94': 17,
    'a0.07b0.93': 18,
    'a0.08b0.92': 19,
    'a0.09b0.91': 20
}


def record_df(df,
              idx,
              proj_name,
              chp=None,
              chp_gh=None,
              chp_smt=None,
              fst=None,
              fst_gh=None,
              fst_smt=None):
    def temp(x, y, z):
        return x / y if z else np.nan
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
    is_chp_gh = False if chp_gh[max_fr_idx] > frs[idx] else True
    is_chp_smt = False if chp_smt[max_fr_idx] > frs[idx] else True
    is_fst_gh = False if fst_gh[max_fr_idx] > frs[idx] else True
    is_fst_smt = False if fst_smt[max_fr_idx] > frs[idx] else True
    chp_time = chp[time_idx]
    chp_price = chp[price_idx]
    fst_time = fst[time_idx]
    fst_price = fst[price_idx]
    chp_runtime_rate_gh = temp(chp_time, chp_gh[time_idx], is_chp_gh)
    chp_price_rate_gh = temp(chp_price, chp_gh[price_idx], is_chp_gh)
    chp_runtime_rate_smt = temp(chp_time, chp_smt[time_idx], is_chp_smt)
    chp_price_rate_smt = temp(chp_price, chp_smt[price_idx], is_chp_smt)
    fst_runtime_rate_gh = temp(fst_time, fst_gh[time_idx], is_fst_gh)
    fst_price_rate_gh = temp(fst_price, fst_gh[price_idx], is_fst_gh)
    fst_runtime_rate_smt = temp(fst_time, fst_smt[time_idx], is_fst_smt)
    fst_price_rate_smt = temp(fst_price, fst_smt[price_idx], is_fst_smt)
    df.loc[len(df.index)] = [
        proj_name,
        chp[category_idx],
        chp[confs_idx],
        chp_time,
        chp_price,
        chp[max_fr_idx],
        chp_gh[confs_idx] if is_chp_gh else np.nan,
        chp_gh[time_idx] if is_chp_gh else np.nan,
        chp_gh[price_idx] if is_chp_gh else np.nan,
        chp_gh[max_fr_idx] if is_chp_gh else np.nan,
        chp_runtime_rate_gh,
        chp_price_rate_gh,
        chp_smt[confs_idx] if is_chp_smt else np.nan,
        chp_smt[time_idx] if is_chp_smt else np.nan,
        chp_smt[price_idx] if is_chp_smt else np.nan,
        chp_smt[max_fr_idx] if is_chp_smt else np.nan,
        chp_runtime_rate_smt,
        chp_price_rate_smt,
        fst[category_idx],
        fst[confs_idx],
        fst_time,
        fst_price,
        fst[max_fr_idx],
        fst_gh[confs_idx] if is_fst_gh else np.nan,
        fst_gh[time_idx] if is_fst_gh else np.nan,
        fst_gh[price_idx] if is_fst_gh else np.nan,
        fst_gh[max_fr_idx] if is_fst_gh else np.nan,
        fst_runtime_rate_gh,
        fst_price_rate_gh,
        fst_smt[confs_idx] if is_fst_smt else np.nan,
        fst_smt[time_idx] if is_fst_smt else np.nan,
        fst_smt[price_idx] if is_fst_smt else np.nan,
        fst_smt[max_fr_idx] if is_fst_smt else np.nan,
        fst_runtime_rate_smt,
        fst_price_rate_smt
    ]
    idx_proj_num_map[idx] += 1
    if is_chp_gh:
        idx_baseline_satis_map[idx][0] += 1
        rec_diff_arr[idx][0] += chp_runtime_rate_gh
        rec_diff_arr[idx][1] += chp_price_rate_gh
        rec_diff_arr[idx][2] += 0 if chp_price_rate_gh >= 1 else 1
    if is_chp_smt:
        idx_baseline_satis_map[idx][1] += 1
        rec_diff_arr[idx][3] += chp_runtime_rate_smt
        rec_diff_arr[idx][4] += chp_price_rate_smt
        rec_diff_arr[idx][5] += 0 if chp_price_rate_smt >= 1 else 1
    if is_fst_gh:
        idx_baseline_satis_map[idx][2] += 1
        rec_diff_arr[idx][6] += fst_runtime_rate_gh
        rec_diff_arr[idx][7] += fst_price_rate_gh
        rec_diff_arr[idx][8] += 0 if fst_runtime_rate_gh >= 1 else 1
    if is_fst_smt:
        idx_baseline_satis_map[idx][3] += 1
        rec_diff_arr[idx][9] += fst_runtime_rate_smt
        rec_diff_arr[idx][10] += fst_price_rate_smt
        rec_diff_arr[idx][11] += 0 if fst_runtime_rate_smt >= 1 else 1


def get_contrast(choice,
                 proj,
                 mach_num,
                 fr=None):
    baseline = baseline_paths[choice] + proj + '.csv'
    dat = pd.read_csv(baseline)
    cond = str(mach_num) + '-' + str(fr) + '-'
    filter_dat = dat.loc[dat['machine_list_or_failure_rate_or_cheap_or_fast_category'].str.find(cond) == 0]
    filter_dat = (filter_dat.iloc[:, 1:]).reset_index(drop=True)
    github_caliber = filter_dat.loc[filter_dat['confs'].str.contains(alter_conf)].iloc[0]
    filter_dat.sort_values(by='max_failure_rate', inplace=True)
    return github_caliber, filter_dat.iloc[0, :]


def consider_fr(modu,
                whe_machine_num_lim=False):
    df_num = 6
    csv_names = [
        'failure_rate_0',
        'failure_rate_0.2',
        'failure_rate_0.4',
        'failure_rate_0.6',
        'failure_rate_0.8',
        'failure_rate_1'
    ]
    fr_idx_map = {
        0: 0,
        0.2: 1,
        0.4: 2,
        0.6: 3,
        0.8: 4,
        1: 5
    }
    modu_idx = modus[modu]
    dat_path = dat_paths[modu_idx]
    rec_diff_df = pd.DataFrame(None,
                               columns=[
                                   'table_category',
                                   'satis_project_num',
                                   'satis_cheapest_github_caliber_num',
                                   'avg_runtime_cheapest_vs_github_caliber_rate',
                                   'avg_price_cheapest_vs_github_caliber_rate',
                                   'number_cases_cheapest_better_vs_github_caliber',
                                   'satis_cheapest_smart_baseline_num',
                                   'avg_runtime_cheapest_vs_smart_baseline_rate',
                                   'avg_price_cheapest_vs_smart_baseline_rate',
                                   'number_cases_cheapest_better_vs_smart_baseline',
                                   'satis_fastest_github_caliber_num',
                                   'avg_runtime_fastest_vs_github_caliber_rate',
                                   'avg_price_fastest_vs_github_caliber_rate',
                                   'number_cases_fastest_better_vs_github_caliber',
                                   'satis_fastest_smart_baseline_num',
                                   'avg_runtime_fastest_vs_smart_baseline_rate',
                                   'avg_price_fastest_vs_smart_baseline_rate',
                                   'number_cases_fastest_better_vs_smart_baseline'
                               ])
    dfs = [pd.DataFrame(None,
                        columns=['project',
                                 'cheapest_category',
                                 'cheapest_confs',
                                 'cheapest_runtime',
                                 'cheapest_price',
                                 'cheapest_max_failure_rate',
                                 'cheapest_github_caliber_confs',
                                 'cheapest_github_caliber_runtime',
                                 'cheapest_github_caliber_price',
                                 'cheapest_github_caliber_max_failure_rate',
                                 'cheapest_github_caliber_runtime_rate',
                                 'cheapest_github_caliber_price_rate',
                                 'cheapest_smart_baseline_confs',
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
                                 'fastest_github_caliber_confs',
                                 'fastest_github_caliber_runtime',
                                 'fastest_github_caliber_price',
                                 'fastest_github_caliber_max_failure_rate',
                                 'fastest_github_caliber_runtime_rate',
                                 'fastest_github_caliber_price_rate',
                                 'fastest_smart_baseline_confs',
                                 'fastest_smart_baseline_runtime',
                                 'fastest_smart_baseline_price',
                                 'fastest_smart_baseline_max_failure_rate',
                                 'fastest_smart_baseline_runtime_rate',
                                 'fastest_smart_baseline_price_rate'])
           for _ in range(df_num)]
    if whe_machine_num_lim:
        output = 'integration_incl_cost_limit/'
    else:
        output = output_paths[modu_idx]
    filenames = os.listdir(dat_path)
    for f in filenames:
        proj_name = f.replace('.csv', '')
        dat = pd.read_csv(dat_path + f).dropna()
        if whe_machine_num_lim:
            arr = dat.iloc[0:48, 1:].values
        else:
            arr = dat.iloc[:, 1:].values
        fr_cond_arr = [[] for _ in range(6)]
        for itm in arr:
            idx = fr_idx_map[float(itm[category_idx].split('-')[1])]
            fr_cond_arr[idx].append(itm)
        for i, fr_cond in enumerate(fr_cond_arr):
            chp = None
            fst = None
            if len(fr_cond) == 0:
                record_df(dfs[i],
                          i,
                          proj_name)
                continue
            chp_sort = sorted(fr_cond, key=lambda x: x[price_idx])
            fst_sort = sorted(fr_cond, key=lambda x: x[time_idx])
            chp_price = chp_sort[0][price_idx]
            fst_time = fst_sort[0][time_idx]
            chp_time = float('inf')
            fst_price = float('inf')
            for itm in chp_sort:
                if itm[price_idx] == chp_price:
                    if chp_time > itm[time_idx]:
                        chp = itm
                        chp_time = itm[time_idx]
                else:
                    break
            for itm in fst_sort:
                if itm[time_idx] == fst_time:
                    if fst_price > itm[price_idx]:
                        fst = itm
                        fst_price = itm[price_idx]
                else:
                    break
            chp_gh, chp_smt = get_contrast(modu_idx,
                                           proj_name,
                                           sum(literal_eval(chp[confs_idx]).values()),
                                           frs[i])
            fst_gh, fst_smt = get_contrast(modu_idx,
                                           proj_name,
                                           sum(literal_eval(fst[confs_idx]).values()),
                                           frs[i])
            record_df(dfs[i],
                      i,
                      proj_name,
                      chp,
                      chp_gh,
                      chp_smt,
                      fst,
                      fst_gh,
                      fst_smt)
    for i, df in enumerate(dfs):
        df.to_csv(f'{output}{csv_names[i]}.csv', sep=',', header=True, index=False)
        proj_num = idx_proj_num_map[i]
        rec_diff_df.loc[len(rec_diff_df.index)] = [
            csv_names[i],
            proj_num,
            idx_baseline_satis_map[i][0],
            rec_diff_arr[i][0] / proj_num,
            rec_diff_arr[i][1] / proj_num,
            rec_diff_arr[i][2],
            idx_baseline_satis_map[i][1],
            rec_diff_arr[i][3] / proj_num,
            rec_diff_arr[i][4] / proj_num,
            rec_diff_arr[i][5],
            idx_baseline_satis_map[i][2],
            rec_diff_arr[i][6] / proj_num,
            rec_diff_arr[i][7] / proj_num,
            rec_diff_arr[i][8],
            idx_baseline_satis_map[i][3],
            rec_diff_arr[i][9] / proj_num,
            rec_diff_arr[i][10] / proj_num,
            rec_diff_arr[i][11]
        ]
    rec_diff_df.to_csv(output + 'dat_summary_result.csv', sep=',', header=True, index=False)


def consider_factor_ab(modu):
    modu_idx = modus[modu]
    dat_path = dat_paths[modu_idx]
    output = output_paths[modu_idx]
    df = pd.DataFrame(None,
                      columns=['project',
                               'category',
                               'confs',
                               'runtime',
                               'price',
                               'score',
                               'max_failure_rate',
                               'github_caliber_confs',
                               'github_caliber_runtime',
                               'github_caliber_price',
                               'github_caliber_score',
                               'github_caliber_max_failure_rate',
                               'github_caliber_score_rate',
                               'smart_baseline_confs',
                               'smart_baseline_runtime',
                               'smart_baseline_price',
                               'smart_baseline_score',
                               'smart_baseline_max_failure_rate',
                               'smart_baseline_score_rate'])
    filenames = os.listdir(dat_path)
    a = float(modu[1:modu.index('b')])
    b = float(modu[modu.index('b') + 1:])
    for f in filenames:
        proj_name = f.replace('.csv', '')
        dat = pd.read_csv(dat_path + f)
        dat = (dat.iloc[:, 1:]).dropna()
        if dat.size == 0:
            continue
        dat.insert(len(dat.columns),
                   'score',
                   pd.Series(a * dat['time_parallel'] + b * dat['price']))
        dat.sort_values(by='score', inplace=True)
        itm = dat.iloc[0, :]
        gh, smt = get_contrast(modu_idx,
                               proj_name,
                               sum(literal_eval(itm[confs_idx]).values()),
                               0)
        is_gh = False if gh[max_fr_idx] > 0 else True
        is_smt = False if smt[max_fr_idx] > 0 else True
        gh_score = np.nan
        gh_rate = np.nan
        smt_score = np.nan
        smt_rate = np.nan
        if is_gh:
            gh_score = a * gh[time_idx] + b * gh[price_idx]
            gh_rate = itm[-1] / gh_score
        if is_smt:
            smt_score = a * smt[time_idx] + b * smt[price_idx]
            smt_rate = itm[-1] / smt_score
        df.loc[len(df.index)] = [
            proj_name,
            itm[category_idx],
            itm[confs_idx],
            itm[time_idx],
            itm[price_idx],
            itm[-1],
            itm[max_fr_idx],
            gh[confs_idx] if is_gh else np.nan,
            gh[time_idx] if is_gh else np.nan,
            gh[price_idx] if is_gh else np.nan,
            gh_score,
            gh[max_fr_idx] if is_gh else np.nan,
            gh_rate,
            smt[confs_idx] if is_smt else np.nan,
            smt[time_idx] if is_smt else np.nan,
            smt[price_idx] if is_smt else np.nan,
            smt_score,
            smt[max_fr_idx] if is_smt else np.nan,
            smt_rate
        ]
    df.to_csv(f'{output}{modu}.csv', sep=',', header=True, index=False)


if __name__ == '__main__':
    ex_ab = False
    if not ex_ab:
        rec_diff_arr = np.zeros((6, 12))
        idx_proj_num_map = {i: 0 for i in range(6)}
        idx_baseline_satis_map = {i: np.zeros(4) for i in range(6)}
    consider_fr(modu='incl')
    consider_fr(modu='excl')
    consider_fr(modu='incl',
                whe_machine_num_lim=True)
    # for key in modus.keys():
    #     if key.find('a') == -1:
    #         continue
    #     consider_factor_ab(key)
