import os
from ast import literal_eval
import pandas as pd

choice = 0
category_idx = 0
conf_idx = 2
para_time_idx = 4
price_idx = 5
csv_names = [
    'table-of-proj-for-0.csv',
    'table-of-proj-for-0.2.csv',
    'table-of-proj-for-0.4.csv',
    'table-of-proj-for-0.6.csv',
    'table-of-proj-for-0.8.csv',
    'table-of-proj-for-1.csv']
frs = [0, 0.2, 0.4, 0.6, 0.8, 1]
dfs = [pd.DataFrame(None,
                    columns=['project',
                             'cheapest_category',
                             'cheapest_confs',
                             'cheapest_price',
                             'cheapest_runtime',
                             'cheapest_baseline_confs',
                             'cheapest_baseline_price',
                             'cheapest_baseline_runtime',
                             'cheapest_price_saving',
                             'cheapest_runtime_rate',
                             'fastest_category',
                             'fastest_confs',
                             'fastest_price',
                             'fastest_runtime',
                             'fastest_baseline_confs',
                             'fastest_baseline_price',
                             'fastest_baseline_runtime',
                             'fastest_price_rate',
                             'fastest_runtime_saving'])
       for _ in range(6)]
fr_idx_map = {0: 0, 0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1: 5}
bl_paths = ['baseline_dat/incl_cost/', 'baseline_dat/incl_cost/', 'baseline_dat/excl_cost/']
dat_paths = ['bruteforce_dat/', 'ext_dat/incl_cost/', 'ext_dat/excl_cost/']
outputs = ['integration_dat_bruteforce/', 'integration_dat_incl_cost/', 'integration_dat_excl_cost/']
alter_conf = '27CPU2Mem8GB.sh'


def get_baseline(proj: str, mach_num: int, fr: float):
    baseline = bl_paths[choice] + proj + '.csv'
    dat = pd.read_csv(baseline)
    cond = str(mach_num) + '-' + str(fr) + '-'
    filter_dat = dat.loc[dat['machine_list_or_failure_rate_or_cheap_or_fast_category'].str.contains(cond) &
                         dat['confs'].str.contains(alter_conf)]
    # filter_dat.sort_values(by='max_failure_rate', inplace=True)
    # conf, price, runtime
    return filter_dat.iloc[0, 3], filter_dat.iloc[0, 5], filter_dat.iloc[0, 6]


if __name__ == '__main__':
    dat_path = dat_paths[choice]
    output = outputs[choice]
    filenames = os.listdir(dat_path)
    for f in filenames:
        proj_name = f[:f.index('csv')-1]
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
            fst_sort = sorted(fr_cond, key=lambda x: x[para_time_idx])
            fst_time = fst_sort[0][para_time_idx]
            chp_price = chp_sort[0][price_idx]
            chp_time = float('inf')
            fst_price = float('inf')
            for itm in chp_sort:
                if itm[price_idx] == chp_price:
                    if chp_time > itm[para_time_idx]:
                        chp_time = itm[para_time_idx]
                        chp_conf = itm[conf_idx]
                        chp_cate = itm[category_idx]
            for itm in fst_sort:
                if itm[para_time_idx] == fst_time:
                    if fst_price > itm[price_idx]:
                        fst_price = itm[price_idx]
                        fst_conf = itm[conf_idx]
                        fst_cate = itm[category_idx]
            chp_bl, chp_bl_time, chp_bl_price = get_baseline(proj_name, sum(literal_eval(chp_conf).values()), frs[i])
            fst_bl, fst_bl_time, fst_bl_price = get_baseline(proj_name, sum(literal_eval(fst_conf).values()), frs[i])
            dfs[i].loc[len(dfs[i].index)] = [proj_name,
                                             chp_cate,
                                             chp_conf,
                                             chp_price,
                                             chp_time,
                                             chp_bl,
                                             chp_bl_price,
                                             chp_bl_time,
                                             1 - chp_price/chp_bl_price,
                                             chp_time/chp_bl_time,
                                             fst_cate,
                                             fst_conf,
                                             fst_price,
                                             fst_time,
                                             fst_bl,
                                             fst_bl_price,
                                             fst_bl_time,
                                             fst_price/fst_bl_price,
                                             1 - fst_time/fst_bl_time]
    for i, df in enumerate(dfs):
        df.to_csv(output + csv_names[i], sep=',', index=False, header=True)
