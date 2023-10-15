import json, time
import pandas as pd
import numpy as np
from ast import literal_eval
from itertools import product
from preproc import preproc, conf_prc_map


def iter_alloc(a,
               subdir):
    def cal_price(mach_tm_dict):
        price = 0
        for key, per_runtime in mach_tm_dict.items():
            per_price = per_runtime * conf_prc_map[mach_arr[key]] / 3600
            price += per_price
        return price

    proj_list = [
        'carbon-apimgt_analyzer-modules.org.wso2.carbon.apimgt.throttling.siddhi.extension',
        'esper_examples.rfidassetzone',
        'fluent-logger-java_dot',
        'hutool_hutool-cron',
        'incubator-dubbo_dubbo-remoting.dubbo-remoting-netty'
    ]
    reco_df = pd.DataFrame(None,
                           columns=['project',
                                    'category',
                                    'confs',
                                    'time_seq',
                                    'time_parallel',
                                    'price',
                                    'test_allocation_record',
                                    'period'])
    for proj in proj_list:
        resu_path = f'ext_dat/{subdir}/{proj}.csv'
        df = pd.read_csv(resu_path)
        df = df.loc[df['category'] == '2-0']
        avg_tm_dict = preproc(proj)
        with open(f'setup_time_rec/{proj}', 'r') as f:
            setup_tm_dict = json.load(f)
        confs_dict = literal_eval(df.iloc[0]['confs'])
        n = len(avg_tm_dict.keys())
        arr = np.array([0, 1])
        prods = np.array(list(product(arr, repeat=n)))
        if len(confs_dict) == 1:
            mach_arr = [list(confs_dict.keys())[0], list(confs_dict.keys())[0]]
        else:
            mach_arr = list(confs_dict.keys())
        tmp_dict = {}
        for ky, val in avg_tm_dict.items():
            tmp_dict[ky] = {}
            for itm in val:
                if itm[0] == mach_arr[0]:
                    if itm[3] == 0:
                        tmp_dict[ky][0] = itm
                    else:
                        tmp_dict[ky][0] = None
                if itm[0] == mach_arr[1]:
                    if itm[3] == 0:
                        tmp_dict[ky][1] = itm
                    else:
                        tmp_dict[ky][1] = None
        idx_tst_map = {i: tst for i, tst in enumerate(list(tmp_dict.keys()))}
        mini = float('inf')
        mini_mach_test_dict = {}
        mini_mach_time_dict = {}
        t1 = time.time()
        for prod in prods:
            is_match_fr = True
            mach_test_dict = {0: [], 1: []}
            mach_time_dict = {0: setup_tm_dict[mach_arr[0]], 1: setup_tm_dict[mach_arr[1]]}
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
                    mini_mach_test_dict = mach_test_dict
                    mini_mach_time_dict = mach_time_dict
            else:
                prc = cal_price(mach_time_dict)
                if is_match_fr and mini >= prc:
                    mini = prc
                    mini_mach_test_dict = mach_test_dict
                    mini_mach_time_dict = mach_time_dict
        t2 = time.time()
        reco_df.loc[len(reco_df.index)] = [
            proj,
            '2-0',
            confs_dict,
            sum(mini_mach_time_dict.values()),
            max(mini_mach_time_dict.values()),
            cal_price(mini_mach_time_dict),
            mini_mach_test_dict,
            t2 - t1
        ]
    reco_df.to_csv(f'{subdir}.csv', sep=',', header=True, index=False)


if __name__ == '__main__':
    # iter_alloc(0,
    #            'ga_a0')
    iter_alloc(1,
               'ga_a1')
