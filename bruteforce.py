import copy
import os
import time
from itertools import combinations

import pandas as pd

from analy import sel_modu, load_setup_time_map, Individual, mapping, confs_num, avail_confs
from preproc import preproc

proj_names = [
    # 'activiti_dot',
    # 'assertj-core_dot',
    # 'carbon-apimgt_analyzer-modules.org.wso2.carbon.apimgt.throttling.siddhi.extension',
    # 'commons-exec_dot'

    # 'db-scheduler_dot',
    # 'delight-nashorn-sandbox_dot',
    # 'elastic-job-lite_dot',
    # 'elastic-job-lite_elastic-job-lite-core'

    # 'esper_examples.rfidassetzone',
    # 'fastjson_dot',
    # 'fluent-logger-java_dot',
    # 'handlebars.java_dot'

    # 'hbase_dot',
    # 'http-request_dot',
    # 'httpcore_dot',
    # 'hutool_hutool-cron'

    # 'incubator-dubbo_dubbo-remoting.dubbo-remoting-netty',
    # 'incubator-dubbo_dubbo-rpc.dubbo-rpc-dubbo',
    # 'logback_dot',
    # 'luwak_luwak'

    # 'ninja_dot',
    # 'noxy_noxy-discovery-zookeeper',
    # 'okhttp_dot',
    # 'orbit_dot'

    # 'retrofit_retrofit-adapters.rxjava',
    # 'retrofit_retrofit',
    # 'rxjava2-extras_dot',
    # 'spring-boot_dot'

    'timely_server',
    'wro4j_wro4j-extensions',
    'yawp_yawp-testing.yawp-testing-appengine',
    'zxing_dot'
]

if __name__ == '__main__':
    bf_dat_path = 'bruteforce_dat/'
    idx_conf_map = {k: v for k, v in enumerate(avail_confs)}
    num_of_machine = [1, 2, 4, 6]
    pct_of_failure_rate = [0, 0.2, 0.4, 0.6, 0.8, 1]
    chp_or_fst = ['cheap', 'fast']
    for proj_name in proj_names:
        csv_name = bf_dat_path + proj_name + '.csv'
        if os.path.exists(csv_name):
            df = pd.read_csv(csv_name)
        else:
            df = pd.DataFrame(None,
                              columns=['project',
                                       'category',
                                       'num_confs',
                                       'confs',
                                       'time_seq',
                                       'time_parallel',
                                       'price',
                                       'min_failure_rate',
                                       'max_failure_rate',
                                       'period']
                              )
        avg_tm_dict = preproc(proj_name)
        setup_tm_dict = load_setup_time_map(proj_name,
                                            False)
        for mach_num in num_of_machine:
            for pct in pct_of_failure_rate:
                for cho in chp_or_fst:
                    t1 = time.time()
                    temp_rec_dict = {}
                    combs = combinations(range(mach_num * confs_num),
                                         mach_num)
                    mini = float('inf')
                    mini_tup = ()
                    st = time.time()
                    for comb in combs:
                        arr = [i % confs_num for i in comb]
                        tup = mapping(arr)
                        if tup in temp_rec_dict.keys():
                            continue
                        time_seq, time_para, price, min_fr, max_fr, mach_test_dict = sel_modu(arr,
                                                                                              pct,
                                                                                              cho,
                                                                                              avg_tm_dict,
                                                                                              setup_tm_dict)
                        ind = Individual(copy.deepcopy(arr),
                                         time_seq,
                                         time_para,
                                         price,
                                         min_fr,
                                         max_fr,
                                         mach_test_dict)
                        if ind.max_fr <= pct:
                            if cho == 'cheap':
                                ind.score = price
                            else:
                                ind.score = time_para
                        else:
                            ind.score = float('inf')
                        if ind.score <= mini:
                            mini = ind.score
                            mini_tup = tup
                        temp_rec_dict[tup] = ind
                    t2 = time.time()
                    tt = t2 - t1
                    category = str(mach_num) + '-' + str(pct) + '-' + cho
                    print('--------------------   ' + proj_name + '-' + category + '   --------------------')
                    mini_ind = temp_rec_dict[mini_tup]
                    mini_ind.print_ind(tt)
                    mini_ind.record_ind('bruteforce',
                                        proj_name,
                                        category,
                                        df,
                                        tt)
        df.to_csv(csv_name, sep=',', header=True, index=False)
