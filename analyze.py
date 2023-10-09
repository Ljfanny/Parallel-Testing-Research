import os
import random
import json
import time
from itertools import combinations
from pprint import pprint, pformat
from deap import base, creator, tools

import numpy as np
import pandas as pd

from preproc import preproc, conf_prc_map

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list,
               fitness=creator.FitnessMin,
               time_seq=np.nan,
               time_para=np.nan,
               price=np.nan,
               min_fr=np.nan,
               max_fr=np.nan,
               mach_test_dict={})

random.seed(0)
resu_path = 'ext_dat'
setup_rec_path = 'setup_time_rec'
tst_alloc_rec_path = 'test_allocation_rec'
proj_names = [
    'activiti_dot',
    'assertj-core_dot',
    'carbon-apimgt_analyzer-modules.org.wso2.carbon.apimgt.throttling.siddhi.extension',
    'commons-exec_dot',
    'db-scheduler_dot',
    'delight-nashorn-sandbox_dot',
    'elastic-job-lite_dot',
    'elastic-job-lite_elastic-job-lite-core',
    'esper_examples.rfidassetzone',
    'fastjson_dot',
    'fluent-logger-java_dot',
    'handlebars.java_dot',
    'hbase_dot',
    'http-request_dot',
    'httpcore_dot',
    'hutool_hutool-cron',
    'incubator-dubbo_dubbo-remoting.dubbo-remoting-netty',
    'incubator-dubbo_dubbo-rpc.dubbo-rpc-dubbo',
    'logback_dot',
    'luwak_luwak',
    'ninja_dot',
    'noxy_noxy-discovery-zookeeper',
    'okhttp_dot',
    'orbit_dot',
    'retrofit_retrofit-adapters.rxjava',
    'retrofit_retrofit',
    'rxjava2-extras_dot',
    'spring-boot_dot',
    'timely_server',
    'wro4j_wro4j-extensions',
    'yawp_yawp-testing.yawp-testing-appengine',
    'zxing_dot'
]
confs_num = 12
avail_confs = [
    '19CPU010Mem1GB.sh',
    '20CPU010Mem2GB.sh',
    '21CPU025Mem2GB.sh',
    '22CPU05Mem2GB.sh',
    '23CPU05Mem4GB.sh',
    '24CPU1Mem4GB.sh',
    '25CPU1Mem8GB.sh',
    '26CPU2Mem4GB.sh',
    '27CPU2Mem8GB.sh',
    '28CPU2Mem16GB.sh',
    '29CPU4Mem8GB.sh',
    '01noThrottling.sh']
versions = [f'v{i}' for i in range(12)]
conf_idx_map = {k: v for v, k in enumerate(avail_confs)}
idx_conf_map = {k: v for k, v in enumerate(avail_confs)}
thrott_conf_idx = 0
outer_round_idx = 1
avg_time_idx = 2
failure_rate_idx = 3
price_idx = 4

a = 0
fr = 0
hist = {}
tests = []
conf_candidates = []
min_conf_candidate_runtime_idx_tup_list = []
setup_tm_dict = {}


def load_setup_cost_file(proj: str,
                         cond: bool):
    global setup_tm_dict
    setup_tm_dict.clear()
    if cond:
        setup_tm_dict = {k: float(0) for k in avail_confs}
        return
    file = f'{setup_rec_path}/{proj}'
    with open(file, 'r') as f:
        setup_tm_dict = json.load(f)


def mapping(machs):
    conf_list = [0 for _ in range(confs_num)]
    for m in machs:
        conf_list[m] += 1
    return tuple(conf_list)


def anal_machs(machs):
    mach_arr = []
    mach_test_dict = {}
    mach_time_dict = {}
    num_arr = list(mapping(machs))
    bool_arr = [False if num <= 1 else True for num in num_arr]
    for i, idx in enumerate(machs):
        conf = idx_conf_map[idx]
        setup_tm = setup_tm_dict[conf]
        if bool_arr[idx]:
            num_arr[idx] -= 1
            cur = (idx, num_arr[idx])
            mach_arr.append(cur)
        else:
            cur = (idx, -1)
            mach_arr.append(cur)
        mach_test_dict[cur] = []
        mach_time_dict[cur] = setup_tm
    return mach_arr, mach_test_dict, mach_time_dict


def get_fit(mach_time_dict):
    price = 0
    b = 1 - a
    beta = 25.993775 / 3600
    for mach, per_runtime in mach_time_dict.items():
        per_price = per_runtime * conf_prc_map[idx_conf_map[mach[0]]] / 3600
        price += per_price
    fitness = a * beta * max(mach_time_dict.values()) + b * price
    return fitness, price


def org_info(avg_tm_dict):
    global tests, conf_candidates, min_conf_candidate_runtime_idx_tup_list
    tests.clear()
    conf_candidates.clear()
    min_conf_candidate_runtime_idx_tup_list.clear()
    cnt = 0
    for t, info in avg_tm_dict.items():
        tests.append(t)
        conf_candidates.append(info)
        min_runtime = min(np.array(info)[:, 2].astype('float'))
        min_conf_candidate_runtime_idx_tup_list.append((min_runtime, cnt))
        cnt += 1


def eval_sche(ind):
    mach_ky = mapping(ind[:])
    if mach_ky in hist.keys():
        ind.min_fr = hist[mach_ky].min_fr
        ind.max_fr = hist[mach_ky].max_fr
        ind.price = hist[mach_ky].price
        ind.time_para = hist[mach_ky].time_para
        ind.time_seq = hist[mach_ky].time_seq
        ind.mach_test_dict = hist[mach_ky].mach_test_dict
        ind.fitness.values = hist[mach_ky].fitness.values
        return
    mach_arr, mach_test_dict, mach_time_dict = anal_machs(ind[:])
    min_fr = 100
    max_fr = 0
    sorted_tup_list = sorted(min_conf_candidate_runtime_idx_tup_list, key=lambda x: x[0], reverse=True)
    for tup in sorted_tup_list:
        idx = tup[1]
        key = tests[idx]
        val = conf_candidates[idx]
        conf_inner_idx_map = {conf_idx_map[cand[thrott_conf_idx]]: i for i, cand in enumerate(val)}
        conv_calc_map = {mach: conf_inner_idx_map[mach[0]] for mach in mach_arr}
        tst = f'{key[0]}#{key[1]}'
        min_fitness = float('inf')
        min_mach = None
        min_mach_fr = 0
        for mach in mach_arr:
            inner_idx = conv_calc_map[mach]
            cur_fr = val[inner_idx][failure_rate_idx]
            cur_avg_tm = val[inner_idx][avg_time_idx]
            if cur_fr > fr:
                continue
            mach_time_dict[mach] += cur_avg_tm
            fitness, _ = get_fit(mach_time_dict)
            if fitness < min_fitness:
                min_fitness = fitness
                min_mach = mach
                min_mach_fr = cur_fr
            mach_time_dict[mach] -= cur_avg_tm
        if min_mach is None:
            ind.fitness.values = (float('inf'),)
            hist[mach_ky] = ind
            return
        mach_test_dict[min_mach].append(tst)
        mach_time_dict[min_mach] += val[conv_calc_map[min_mach]][avg_time_idx]
        if min_mach_fr > max_fr:
            max_fr = min_mach_fr
        if min_mach_fr < min_fr:
            min_fr = min_mach_fr
    ind.max_fr = max_fr
    ind.min_fr = min_fr
    ind.time_para = max(mach_time_dict.values())
    ind.time_seq = sum(mach_time_dict.values())
    ind.mach_test_dict = mach_test_dict
    fit, ind.price = get_fit(mach_time_dict)
    ind.fitness.values = (fit,)
    hist[mach_ky] = ind


def eval_bf(ind):
    mach_ky = mapping(ind[:])
    if mach_ky in hist.keys():
        ind.min_fr = hist[mach_ky].min_fr
        ind.max_fr = hist[mach_ky].max_fr
        ind.price = hist[mach_ky].price
        ind.time_para = hist[mach_ky].time_para
        ind.time_seq = hist[mach_ky].time_seq
        ind.mach_test_dict = hist[mach_ky].mach_test_dict
        ind.fitness.values = hist[mach_ky].fitness.values
        return
    mach_arr, mach_test_dict, mach_time_dict = anal_machs(ind[:])
    min_fr = 100
    max_fr = 0
    sorted_tup_list = sorted(min_conf_candidate_runtime_idx_tup_list, key=lambda x: x[0], reverse=True)
    for tup in sorted_tup_list:
        idx = tup[1]
        key = tests[idx]
        val = conf_candidates[idx]
        conf_inner_idx_map = {conf_idx_map[cand[thrott_conf_idx]]: i for i, cand in enumerate(val)}
        conv_calc_map = {mach: conf_inner_idx_map[mach[0]] for mach in mach_arr}
        tst = f'{key[0]}#{key[1]}'
        min_fitness = float('inf')
        min_mach = None
        min_mach_fr = 0
        for mach in mach_arr:
            inner_idx = conv_calc_map[mach]
            cur_fr = val[inner_idx][failure_rate_idx]
            cur_avg_tm = val[inner_idx][avg_time_idx]
            if cur_fr > fr:
                continue
            mach_time_dict[mach] += cur_avg_tm
            fitness, _ = get_fit(mach_time_dict)
            if fitness < min_fitness:
                min_fitness = fitness
                min_mach = mach
                min_mach_fr = cur_fr
            mach_time_dict[mach] -= cur_avg_tm
        if min_mach is None:
            ind.fitness.values = (float('inf'),)
            hist[mach_ky] = ind
            return
        mach_test_dict[min_mach].append(tst)
        mach_time_dict[min_mach] += val[conv_calc_map[min_mach]][avg_time_idx]
        if min_mach_fr > max_fr:
            max_fr = min_mach_fr
        if min_mach_fr < min_fr:
            min_fr = min_mach_fr
    ind.max_fr = max_fr
    ind.min_fr = min_fr
    ind.time_para = max(mach_time_dict.values())
    ind.time_seq = sum(mach_time_dict.values())
    ind.mach_test_dict = mach_test_dict
    fit, ind.price = get_fit(mach_time_dict)
    ind.fitness.values = (fit,)
    hist[mach_ky] = ind


def ga(gene_len):
    pop = toolbox.population(n=100)
    list(map(toolbox.evaluate, pop))
    pop = sorted(pop, key=lambda x: x.fitness.values[0])
    if gene_len == 1:
        return pop[0]
    for g in range(25):
        pop = sorted(pop, key=lambda x: x.fitness.values[0])
        offspring = list(map(toolbox.clone, pop[:35]))
        for chd1, chd2 in zip(pop[::2], pop[1::2]):
            toolbox.mate(chd1, chd2, 0.5)
        for gen in pop:
            toolbox.mutate(gen)
        pop = sorted(pop, key=lambda x: x.fitness.values[0])
        pop[:] = offspring + list(map(toolbox.clone, pop[:65]))
        list(map(toolbox.evaluate, pop))
    return pop[0]


def bruteforce(gene_len):
    combs = combinations(range(gene_len * confs_num),
                         gene_len)
    mini = float('inf')
    mini_ind = None
    for comb in combs:
        machs = [i % confs_num for i in comb]
        ind = creator.Individual(machs)
        eval_sche(ind)
        if ind.fitness.values[0] <= mini:
            mini = ind.fitness.values[0]
            mini_ind = ind
    return mini_ind


def print_ind(ind,
              period):
    print(f'Period: {period}')
    fit = ind.fitness.values[0]
    if fit == float('inf'):
        print(f'Maximum failure rate: {ind.max_fr}')
        return
    print('Machine list: ')
    pprint({(k, i): mapping(ind[:])[i] for i, k in enumerate(avail_confs)})
    print(f'Time seq: {ind.time_seq}')
    print(f'Time parallel: {ind.time_para}')
    print(f'Price: {ind.price}')
    print(f'Minimum failure rate: {ind.min_fr}')
    print(f'Maximum failure rate: {ind.max_fr}')
    print(f'Score: {fit}')
    print(ind.mach_test_dict.keys())


def record_ind(ind,
               subdir,
               proj,
               cg,
               df,
               period):
    dis_folder = f'{tst_alloc_rec_path}/{subdir}/{proj}'
    fit = ind.fitness.values[0]
    if not os.path.exists(dis_folder):
        os.makedirs(dis_folder)
    if fit == float('inf'):
        df.loc[len(df.index)] = [
            proj,
            cg,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            period
        ]
        with open(f'{dis_folder}/category{cg}', 'w'):
            pass
        return
    num_tup = mapping(ind[:])
    confs = set(ind[:])
    conf_num_map = {idx_conf_map[k]: num_tup[k] for k in confs}
    df.loc[len(df.index)] = [
        proj,
        cg,
        int(len(confs)),
        conf_num_map,
        ind.time_seq,
        ind.time_para,
        ind.price,
        ind.min_fr,
        ind.max_fr,
        fit,
        period
    ]
    temp_dict = {idx_conf_map[k[0]] if k[1] == -1
                 else f'{idx_conf_map[k[0]]}:{versions[k[1]]}': v for k, v in ind.mach_test_dict.items()}
    with open(f'{dis_folder}/category{cg}', 'w') as f:
        f.write(pformat(temp_dict))


if __name__ == '__main__':
    # a = 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1
    prog_start = time.time()
    a = 0
    group_ky = 'non-ig'
    groups_map = {
        'non-ig': ['', False],
        'ig': ['_ig', True]
    }
    num_of_machine = [1, 2, 4, 6, 8, 10, 12]
    pct_of_failure_rate = [0, 0.2, 0.4, 0.6, 0.8, 1]
    sub = f'ga_a{a}{groups_map[group_ky][0]}'
    for proj_name in proj_names:
        ext_dat_df = pd.DataFrame(None,
                                  columns=['project',
                                           'category',
                                           'num_confs',
                                           'confs',
                                           'time_seq',
                                           'time_parallel',
                                           'price',
                                           'min_failure_rate',
                                           'max_failure_rate',
                                           'fitness',
                                           'period']
                                  )
        ext_dat_df['num_confs'] = ext_dat_df['num_confs'].astype(int)
        org_info(preproc(proj_name))
        load_setup_cost_file(proj_name,
                             groups_map[group_ky][1])
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 0, 11)
        toolbox.register("mate", tools.cxUniform)
        toolbox.register("evaluate", eval_sche)
        for mach_num in num_of_machine:
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, mach_num)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("mutate", tools.mutUniformInt, low=0, up=11, indpb=1 / mach_num)
            for fr in pct_of_failure_rate:
                hist.clear()
                t1 = time.time()
                # best_ind = bruteforce(mach_num)
                best_ind = ga(mach_num)
                t2 = time.time()
                tt = t2 - t1
                category = f'{mach_num}-{fr}'
                print(f'-------------------- {proj_name}-{category} --------------------')
                print_ind(best_ind,
                          tt)
                record_ind(best_ind,
                           sub,
                           proj_name,
                           category,
                           ext_dat_df,
                           tt)
        resu_sub_path = f'{resu_path}/{sub}'
        if not os.path.exists(resu_sub_path):
            os.mkdir(resu_sub_path)
        ext_dat_df.to_csv(f'{resu_sub_path}/{proj_name}.csv', sep=',', header=True, index=False)
    print(f'[Total time] {time.time() - prog_start} s')
