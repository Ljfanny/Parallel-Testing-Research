import os
from ast import literal_eval
import numpy as np
import pandas as pd

category_idx = 0
confs_idx = 2
time_idx = 4
price_idx = 5
max_fr_idx = 7
alter_conf = '27CPU2Mem8GB.sh'
bl_paths = ['baseline_dat/incl_cost/', 'baseline_dat/incl_cost/', 'baseline_dat/excl_cost/']
dat_paths = ['bruteforce_dat/', 'ext_dat/incl_cost/', 'ext_dat/excl_cost/']
outputs = ['integration_dat_bruteforce/', 'integration_dat_incl_cost/', 'integration_dat_excl_cost/']


def record_df(df,
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
    chp_time = chp[time_idx]
    chp_price = chp[price_idx]
    fst_time = fst[time_idx]
    fst_price = fst[price_idx]
    df.loc[len(df.index)] = [
        proj_name,
        chp[category_idx],
        chp[confs_idx],
        chp_time,
        chp_price,
        chp[max_fr_idx],
        chp_gh[confs_idx],
        chp_gh[time_idx],
        chp_gh[price_idx],
        chp_gh[max_fr_idx],
        chp_time / chp_gh[time_idx],
        chp_price / chp_gh[price_idx],
        chp_smt[confs_idx],
        chp_smt[time_idx],
        chp_smt[price_idx],
        chp_smt[max_fr_idx],
        chp_time / chp_smt[time_idx],
        chp_price / chp_smt[price_idx],
        fst[category_idx],
        fst[confs_idx],
        fst_time,
        fst_price,
        fst[max_fr_idx],
        fst_gh[confs_idx],
        fst_gh[time_idx],
        fst_gh[price_idx],
        fst_gh[max_fr_idx],
        fst_time / fst_gh[time_idx],
        fst_price / fst_gh[price_idx],
        fst_smt[confs_idx],
        fst_smt[time_idx],
        fst_smt[price_idx],
        fst_smt[max_fr_idx],
        fst_time / fst_smt[time_idx],
        fst_price / fst_smt[price_idx]
    ]


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


def consider_fr(choice: int):
    csv_names = [
        'failure_rate_0.csv',
        'failure_rate_0.2.csv',
        'failure_rate_0.4.csv',
        'failure_rate_0.6.csv',
        'failure_rate_0.8.csv',
        'failure_rate_1.csv']
    frs = [0, 0.2, 0.4, 0.6, 0.8, 1]
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
    fr_idx_map = {0: 0, 0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1: 5}
    dat_path = dat_paths[choice]
    output = outputs[choice]
    filenames = os.listdir(dat_path)
    for f in filenames:
        proj_name = f[:f.index('csv') - 1]
        dat = pd.read_csv(dat_path + f)
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
                      proj_name,
                      chp,
                      chp_gh,
                      chp_smt,
                      fst,
                      fst_gh,
                      fst_smt)
    for i, df in enumerate(dfs):
        df.to_csv(output + csv_names[i], sep=',', header=True, index=False)


if __name__ == '__main__':
    modus = {'bruteforce': 0, 'incl': 1, 'excl': 2}
    consider_fr(modus['bruteforce'])
    consider_fr(modus['incl'])
    consider_fr(modus['excl'])
