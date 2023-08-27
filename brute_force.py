import copy
import os
import time
from itertools import combinations

import pandas as pd

from analy import sel_modu, proj_names, load_setup_time_map, Individual, mapping, confs_num, avail_confs
from preproc import preproc


if __name__ == '__main__':
    bf_dat_path = 'brute_force_dat/'
    idx_conf_map = {k: v for k, v in enumerate(avail_confs)}
    num_of_machine = [1, 2, 4, 6, 8, 10, 12]
    pct_of_failure_rate = [0, 0.2, 0.4, 0.6, 0.8, 1]
    chp_or_fst = ['cheap', 'fast']
    for proj_name in proj_names:
        csv_name = bf_dat_path + proj_name + '.csv'
        if os.path.exists(csv_name):
            df = pd.read_csv(csv_name)
        else:
            df = pd.DataFrame(None,
                              columns=['project',
                                       'machine_list_or_failure_rate_or_cheap_or_fast_category',
                                       'num_confs',
                                       'confs',
                                       'time_seq',
                                       'time_parallel',
                                       'price',
                                       'min_failure_rate',
                                       'max_failure_rate']
                              )
        avg_tm_dict = preproc(proj_name)
        setup_tm_dict = load_setup_time_map(proj_name, False)
        for mach_num in num_of_machine:
            for pct in pct_of_failure_rate:
                for cho in chp_or_fst:
                    temp_rec_dict = {}
                    combs = combinations(range(mach_num * confs_num), mach_num)
                    mini = float('inf')
                    mini_tup = ()
                    tt = 0
                    st = time.time()
                    for comb in combs:
                        arr = [i % confs_num for i in comb]
                        tup = mapping(arr)
                        if tup in temp_rec_dict.keys():
                            continue
                        time_seq, time_parallel, price, min_fr, max_fr, mach_test_dict, unmatched_fr_test = \
                            sel_modu(arr, pct, cho, avg_tm_dict, setup_tm_dict)
                        ind = Individual(copy.deepcopy(arr), time_seq,
                                         time_parallel, price, min_fr, max_fr,
                                         mach_test_dict, unmatched_fr_test)
                        temp_rec_dict[tup] = ind
                        if cho == 'cheap':
                            ind.score = price
                        else:
                            ind.score = time_parallel
                        if ind.score < mini:
                            mini = ind.score
                            mini_tup = tup
                        tt = time.time() - st
                        if tt >= 1800:
                            break
                    category = str(mach_num) + '-' + str(pct) + '-' + cho
                    print('--------------------   ' + proj_name + '-' + category + '   --------------------')
                    print(f'[INFO] Total time: {tt} s')
                    mini_ind = temp_rec_dict[mini_tup]
                    mini_ind.print_ind()
                    mini_ind.record_ind('brute_force', proj_name, category, df)
        df.to_csv(csv_name, sep=',', index=False, header=True)
