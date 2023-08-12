import os
import pandas as pd

cate_idx = 0
conf_idx = 2
para_time_idx = 4
price_idx = 5
res_path = 'ext_dat/'
output = 'integration_dat.xlsx'
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
                             'fastest_price',
                             'fastest_runtime',
                             'fastest_conf',
                             'fastest_category'])
       for _ in range(6)]
fr_idx_map = {0: 0, 0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1: 5}

if __name__ == '__main__':
    filenames = os.listdir(res_path)
    for f in filenames:
        proj_name = f[:f.index('.')]
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
                        chp_cate = itm[cate_idx]
            for itm in fst_sort:
                if itm[para_time_idx] == fst_time:
                    if fst_price > itm[price_idx]:
                        fst_price = itm[price_idx]
                        fst_conf = itm[conf_idx]
                        fst_cate = itm[cate_idx]
            dfs[i].loc[len(dfs[i].index)] = [proj_name + '-' + str(frs[i]),
                                             chp_price,
                                             chp_time,
                                             chp_conf,
                                             chp_cate,
                                             fst_price,
                                             fst_time,
                                             fst_conf,
                                             fst_cate]
    writer = pd.ExcelWriter(output)
    for i, df in enumerate(dfs):
        df.to_excel(writer, sheet_name=sheets_name[i])
    writer.save()
