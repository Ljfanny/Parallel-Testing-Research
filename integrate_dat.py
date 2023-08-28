import os
from ast import literal_eval
import pandas as pd

choice = 2
category_idx = 0
confs_idx = 2
time_idx = 4
price_idx = 5
csv_names = [
    'table_for_0.csv',
    'table_for_0.2.csv',
    'table_for_0.4.csv',
    'table_for_0.6.csv',
    'table_for_0.8.csv',
    'table_for_1.csv']
frs = [0, 0.2, 0.4, 0.6, 0.8, 1]
dfs = [pd.DataFrame(None,
                    columns=['project',
                             'cheapest_category',
                             'cheapest_confs',
                             'cheapest_runtime',
                             'cheapest_price',
                             'cheapest_github_caliber_confs',
                             'cheapest_github_caliber_runtime',
                             'cheapest_github_caliber_price',
                             'cheapest_github_caliber_runtime_rate',
                             'cheapest_github_caliber_price_saving',
                             'cheapest_smart_baseline_confs',
                             'cheapest_smart_baseline_runtime',
                             'cheapest_smart_baseline_price',
                             'cheapest_smart_baseline_runtime_rate',
                             'cheapest_smart_baseline_price_saving',
                             'fastest_category',
                             'fastest_confs',
                             'fastest_runtime',
                             'fastest_price',
                             'fastest_github_caliber_confs',
                             'fastest_github_caliber_runtime',
                             'fastest_github_caliber_price',
                             'fastest_github_caliber_runtime_rate',
                             'fastest_github_caliber_price_saving',
                             'fastest_smart_baseline_confs',
                             'fastest_smart_baseline_runtime',
                             'fastest_smart_baseline_price',
                             'fastest_smart_baseline_runtime_rate',
                             'fastest_smart_baseline_price_saving']) for _ in range(6)]
alter_conf = '27CPU2Mem8GB.sh'
fr_idx_map = {0: 0, 0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1: 5}
bl_paths = ['baseline_dat/incl_cost/', 'baseline_dat/incl_cost/', 'baseline_dat/excl_cost/']
dat_paths = ['bruteforce_dat/', 'ext_dat/incl_cost/', 'ext_dat/excl_cost/']
outputs = ['integration_dat_bruteforce/', 'integration_dat_incl_cost/', 'integration_dat_excl_cost/']


def get_contrast(proj: str, mach_num: int, fr: float):
    baseline = bl_paths[choice] + proj + '.csv'
    dat = pd.read_csv(baseline)
    cond = str(mach_num) + '-' + str(fr) + '-'
    filter_dat = dat.loc[dat['machine_list_or_failure_rate_or_cheap_or_fast_category'].str.contains(cond)]
    filter_dat = (filter_dat.iloc[:, 1:]).reset_index(drop=True)
    github_caliber = filter_dat.loc[filter_dat['confs'].str.contains(alter_conf)].iloc[0]
    filter_dat.sort_values(by='max_failure_rate', inplace=True)
    return github_caliber, filter_dat.iloc[0, :]


if __name__ == '__main__':
    dat_path = dat_paths[choice]
    output = outputs[choice]
    filenames = os.listdir(dat_path)
    for f in filenames:
        proj_name = f[:f.index('csv') - 1]
        df = pd.read_csv(dat_path + f)
        arr = df.iloc[:, 1:].values
        fr_cond_arr = [[] for _ in range(6)]
        for itm in arr:
            idx = fr_idx_map[float(itm[0].split('-')[1])]
            fr_cond_arr[idx].append(itm)
        for i, fr_cond in enumerate(fr_cond_arr):
            chp_cate = ''
            fst_cate = ''
            chp_conf = None
            fst_conf = None
            chp_sort = sorted(fr_cond, key=lambda x: x[price_idx])
            fst_sort = sorted(fr_cond, key=lambda x: x[time_idx])
            fst_time = fst_sort[0][time_idx]
            chp_price = chp_sort[0][price_idx]
            chp_time = float('inf')
            fst_price = float('inf')
            for itm in chp_sort:
                if itm[price_idx] == chp_price:
                    if chp_time > itm[time_idx]:
                        chp_time = itm[time_idx]
                        chp_conf = itm[confs_idx]
                        chp_cate = itm[category_idx]
            for itm in fst_sort:
                if itm[time_idx] == fst_time:
                    if fst_price > itm[price_idx]:
                        fst_price = itm[price_idx]
                        fst_conf = itm[confs_idx]
                        fst_cate = itm[category_idx]
            chp_gh, chp_smt = get_contrast(proj_name, sum(literal_eval(chp_conf).values()), frs[i])
            fst_gh, fst_smt = get_contrast(proj_name, sum(literal_eval(fst_conf).values()), frs[i])
            dfs[i].loc[len(dfs[i].index)] = [proj_name,
                                             chp_cate,
                                             chp_conf,
                                             chp_time,
                                             chp_price,
                                             chp_gh[confs_idx],
                                             chp_gh[time_idx],
                                             chp_gh[price_idx],
                                             chp_time / chp_gh[time_idx],
                                             1 - chp_price / chp_gh[price_idx],
                                             chp_smt[confs_idx],
                                             chp_smt[time_idx],
                                             chp_smt[price_idx],
                                             chp_time / chp_smt[time_idx],
                                             1 - chp_price / chp_smt[price_idx],
                                             fst_cate,
                                             fst_conf,
                                             fst_time,
                                             fst_price,
                                             fst_gh[confs_idx],
                                             fst_gh[time_idx],
                                             fst_gh[price_idx],
                                             1 - fst_time / fst_gh[time_idx],
                                             fst_price / fst_gh[price_idx],
                                             fst_smt[confs_idx],
                                             fst_smt[time_idx],
                                             fst_smt[price_idx],
                                             1 - fst_time / fst_smt[time_idx],
                                             fst_price / fst_smt[price_idx]]
    for i, df in enumerate(dfs):
        df.to_csv(output + csv_names[i], sep=',', float_format='%.2f', header=True, index=False)
