import os
import copy
from ast import literal_eval

import pandas as pd

category_idx = 0
conf_idx = 2
para_time_idx = 4
price_idx = 5
sheets_name = [
    'table-of-proj-for-0',
    'table-of-proj-for-0.2',
    'table-of-proj-for-0.4',
    'table-of-proj-for-0.6',
    'table-of-proj-for-0.8',
    'table-of-proj-for-1'
]
frs = [0, 0.2, 0.4, 0.6, 0.8, 1]
dfs = [pd.DataFrame(None,
                    columns=['project_fr',
                             'cheapest_price',
                             'cheapest_runtime',
                             'cheapest_conf',
                             'cheapest_category',
                             'cheapest_baseline_conf',
                             'cheapest_price_baseline',
                             'cheapest_price_saving',
                             'fastest_price',
                             'fastest_runtime',
                             'fastest_conf',
                             'fastest_category',
                             'fastest_baseline_conf',
                             'fastest_runtime_baseline',
                             'fastest_runtime_saving'])
       for _ in range(6)]
fr_idx_map = {0: 0, 0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1: 5}
choice = 'excl_cost'
baseline_path = f'baseline_dat/{choice}/'
col_name = 'machine_list_or_failure_rate_or_cheap_or_fast_category'


def get_baseline(modu: str, proj: str, mach_num: int, fr: float):
    baseline = baseline_path + proj + '.csv'
    dat = pd.read_csv(baseline)
    cond = str(mach_num) + '-' + str(fr) + '-'
    filter_dat = copy.deepcopy(dat.loc[dat[col_name].str.contains(cond)])
    filter_dat.sort_values(by='max_failure_rate', inplace=True)
    return filter_dat.iloc[0, 3], filter_dat.iloc[0, 6] if modu == 'cheap' else filter_dat.iloc[0, 5]


if __name__ == '__main__':
    res_path = f'ext_dat/{choice}/'
    resu = f'integration_dat_{choice}.xlsx'
    filenames = os.listdir(res_path)
    for f in filenames:
        proj_name = f[:f.index('csv')-1]
        df = pd.read_csv(res_path + f)
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
            chp_bl, chp_bl_price = get_baseline('cheap', proj_name, sum(literal_eval(chp_conf).values()), frs[i])
            fst_bl, fst_bl_time = get_baseline('fast', proj_name, sum(literal_eval(fst_conf).values()), frs[i])
            dfs[i].loc[len(dfs[i].index)] = [proj_name + '-' + str(frs[i]),
                                             chp_price,
                                             chp_time,
                                             chp_conf,
                                             chp_cate,
                                             chp_bl,
                                             chp_bl_price,
                                             '{:.2%}'.format(1 - chp_price / chp_bl_price),
                                             fst_price,
                                             fst_time,
                                             fst_conf,
                                             fst_cate,
                                             fst_bl,
                                             fst_bl_time,
                                             '{:.2%}'.format(1 - fst_time / fst_bl_time)]
    writer = pd.ExcelWriter(resu)
    for i, df in enumerate(dfs):
        df.to_excel(writer, sheet_name=sheets_name[i])
    writer.save()
