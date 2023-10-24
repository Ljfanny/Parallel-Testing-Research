import json, time
import pandas as pd
import numpy as np
from ast import literal_eval
from itertools import product
from preproc import preproc, conf_prc_map

proj_list = [
    'incubator-dubbo_dubbo-remoting.dubbo-remoting-netty',
    'hutool_hutool-cron'
]


def iter_alloc(a,
               subdir):
    def cal_price(mach_tm_dict):
        price = 0
        for key, per_runtime in mach_tm_dict.items():
            per_price = per_runtime * conf_prc_map[mach_arr[key]] / 3600
            price += per_price
        return price

    reco_df = pd.DataFrame(None,
                           columns=['project',
                                    'category',
                                    'confs',
                                    'time_seq',
                                    'time_parallel',
                                    'price',
                                    'period'])
    for proj in proj_list:
        resu_path = f'ext_dat/{subdir}/{proj}.csv'
        df = pd.read_csv(resu_path)
        df = df.loc[df['category'] == '4-0']
        avg_tm_dict = preproc(proj)
        with open(f'setup_time_rec/{proj}', 'r') as f:
            setup_tm_dict = json.load(f)
        confs_dict = literal_eval(df.iloc[0]['confs'])
        n = len(avg_tm_dict.keys())
        arr = np.array([0, 1, 2, 3])
        prods = product(arr, repeat=n)
        mach_arr = []
        for ky, val in confs_dict.items():
            for _ in range(val):
                mach_arr.append(ky)
        tmp_dict = {}
        for ky, val in avg_tm_dict.items():
            tmp_dict[ky] = {}
            for itm in val:
                for i in range(4):
                    if itm[0] == mach_arr[i]:
                        if itm[3] == 0:
                            tmp_dict[ky][i] = itm
                        else:
                            tmp_dict[ky][i] = None
        idx_tst_map = {i: tst for i, tst in enumerate(list(tmp_dict.keys()))}
        mini = float('inf')
        mini_mach_time_dict = {}
        t1 = time.time()
        for prod in prods:
            is_match_fr = True
            mach_test_dict = {i: [] for i in range(4)}
            mach_time_dict = {i: setup_tm_dict[mach_arr[i]] for i in range(4)}
            for idx, mach_id in enumerate(prod):
                tst = idx_tst_map[idx]
                itm = tmp_dict[tst][mach_id]
                if itm is None:
                    is_match_fr = False
                    break
                mach_test_dict[mach_id].append(tst)
                mach_time_dict[mach_id] += itm[2]
            if a == 1:
                runtm = max(mach_time_dict.values())
                if is_match_fr and mini >= runtm:
                    mini = runtm
                    mini_mach_time_dict = mach_time_dict
            else:
                prc = cal_price(mach_time_dict)
                if is_match_fr and mini >= prc:
                    mini = prc
                    mini_mach_time_dict = mach_time_dict
        t2 = time.time()

        reco_df.loc[len(reco_df.index)] = [
            proj,
            '4-0',
            confs_dict,
            sum(mini_mach_time_dict.values()),
            max(mini_mach_time_dict.values()),
            cal_price(mini_mach_time_dict),
            t2 - t1
        ]
    reco_df.to_csv(f'{subdir}.csv', sep=',', header=True, index=False)


if __name__ == '__main__':
    iter_alloc(0,
               'ga_a0')
    iter_alloc(1,
               'ga_a1')
