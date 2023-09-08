import os
from ast import literal_eval
import numpy as np
import pandas as pd

category_idx = 0
confs_idx = 2
time_idx = 4
price_idx = 5
max_fr_idx = 7
rec_diff_arr = np.zeros((6, 12))
idx_proj_num_map = {i: 0 for i in range(6)}
idx_baseline_satis_map = {i: np.zeros(4) for i in range(6)}
frs = [0, 0.2, 0.4, 0.6, 0.8, 1]
alter_conf = '27CPU2Mem8GB.sh'
bl_paths = ['baseline_dat/incl_cost/', 'baseline_dat/incl_cost/', 'baseline_dat/excl_cost/']
dat_paths = ['bruteforce_dat/', 'ext_dat/incl_cost/', 'ext_dat/excl_cost/']
outputs = ['integration_dat_bruteforce/', 'integration_dat_incl_cost/', 'integration_dat_excl_cost/']


def record_df(df,
              idx,
              proj_name,
              chp=None,
              chp_gh=None,
              chp_smt=None,
              fst=None,
              fst_gh=None,
              fst_smt=None):
    if chp is None or fst is None:
        df.loc[len(df.index)] = [
            proj_name,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ]
        return
    is_chp_gh = False if chp_gh[max_fr_idx] > frs[idx] else True
    is_chp_smt = False if chp_smt[max_fr_idx] > frs[idx] else True
    is_fst_gh = False if fst_gh[max_fr_idx] > frs[idx] else True
    is_fst_smt = False if fst_smt[max_fr_idx] > frs[idx] else True
    idx_proj_num_map[idx] += 1
    chp_time = chp[time_idx]
    chp_price = chp[price_idx]
    fst_time = fst[time_idx]
    fst_price = fst[price_idx]
    chp_runtime_rate_gh = chp_time / chp_gh[time_idx] if is_chp_gh else np.nan
    chp_price_rate_gh = chp_price / chp_gh[price_idx] if is_chp_gh else np.nan
    chp_runtime_rate_smt = chp_time / chp_smt[time_idx] if is_chp_smt else np.nan
    chp_price_rate_smt = chp_price / chp_smt[price_idx] if is_chp_smt else np.nan
    fst_runtime_rate_gh = fst_time / fst_gh[time_idx] if is_fst_gh else np.nan
    fst_price_rate_gh = fst_price / fst_gh[price_idx] if is_fst_gh else np.nan
    fst_runtime_rate_smt = fst_time / fst_smt[time_idx] if is_fst_smt else np.nan
    fst_price_rate_smt = fst_price / fst_smt[price_idx] if is_fst_smt else np.nan
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
                 is_fr=False,
                 fr=None):
    baseline = bl_paths[choice] + proj + '.csv'
    dat = pd.read_csv(baseline)
    cond = str(mach_num) + '-'
    if is_fr:
        cond += str(fr) + '-'
    filter_dat = dat.loc[dat['machine_list_or_failure_rate_or_cheap_or_fast_category'].str.find(cond) == 0]
    filter_dat = (filter_dat.iloc[:, 1:]).reset_index(drop=True)
    github_caliber = filter_dat.loc[filter_dat['confs'].str.contains(alter_conf)].iloc[0]
    filter_dat.sort_values(by='max_failure_rate', inplace=True)
    return github_caliber, filter_dat.iloc[0, :]


def consider_fr(choice: int,
                is_limit=False):
    csv_names = [
        'failure_rate_0.csv',
        'failure_rate_0.2.csv',
        'failure_rate_0.4.csv',
        'failure_rate_0.6.csv',
        'failure_rate_0.8.csv',
        'failure_rate_1.csv']
    dfs = [pd.DataFrame(None, columns=['project',
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
                                       'fastest_smart_baseline_price_rate']) for _ in range(6)]
    rec_diff_df = pd.DataFrame(None,
                               columns=[
                                   'table_category',
                                   'satis_proj_num',
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
    fr_idx_map = {0: 0, 0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1: 5}
    dat_path = dat_paths[choice]
    if is_limit:
        output = 'integration_dat_incl_cost_limit/'
    else:
        output = outputs[choice]
    filenames = os.listdir(dat_path)
    for f in filenames:
        proj_name = f[:f.index('csv') - 1]
        dat = pd.read_csv(dat_path + f)
        if is_limit:
            arr = dat.iloc[0:48, 1:].values
        else:
            arr = dat.iloc[:, 1:].values
        fr_cond_arr = [[] for _ in range(6)]
        for itm in arr:
            if np.isnan(itm[max_fr_idx]):
                continue
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
            for itm in fst_sort:
                if itm[time_idx] == fst_time:
                    if fst_price > itm[price_idx]:
                        fst = itm
                        fst_price = itm[price_idx]
            chp_gh, chp_smt = get_contrast(choice,
                                           proj_name,
                                           sum(literal_eval(chp[confs_idx]).values()),
                                           True,
                                           frs[i])
            fst_gh, fst_smt = get_contrast(choice,
                                           proj_name,
                                           sum(literal_eval(fst[confs_idx]).values()),
                                           True,
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
        df.to_csv(output + csv_names[i], sep=',', header=True, index=False)
        proj_num = idx_proj_num_map[i]
        rec_diff_df.loc[len(rec_diff_df.index)] = [
            csv_names[i].replace('.csv', ''),
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


if __name__ == '__main__':
    modus = {'bruteforce': 0, 'incl': 1, 'excl': 2}
    # consider_fr(modus['bruteforce'])
    # consider_fr(modus['incl'])
    consider_fr(modus['excl'])
    # consider_fr(modus['incl'], True)
