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

iters = 50
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list,
               fitness=creator.FitnessMin,
               time_seq=np.nan,
               time_para=np.nan,
               price=np.nan,
               min_fr=np.nan,
               max_fr=np.nan,
               pro_fr=np.nan,
               mach_ts_dict={})

random.seed(0)
resu_path = 'rich_RQ5'
base_path = 'baseline'
setup_rec_path = 'setup_time_record'
proj_names = [
    # 'activiti_dot',
    # 'fastjson_dot',
    # 'commons-exec_dot',
    # 'httpcore_dot',
    # 'incubator-dubbo_dubbo-remoting.dubbo-remoting-netty',
    # 'incubator-dubbo_dubbo-rpc.dubbo-rpc-dubbo',
    # 'rxjava2-extras_dot',
    # 'elastic-job-lite_dot',
    # 'elastic-job-lite_elastic-job-lite-core',
    # 'luwak_luwak',
    # 'fluent-logger-java_dot',
    # 'delight-nashorn-sandbox_dot',
    # 'handlebars.java_dot',
    # 'assertj-core_dot',
    'db-scheduler_dot',
    'http-request_dot',
    'timely_server',
    'ninja_dot',
    'orbit_dot',
    'logback_dot',
    'spring-boot_dot',
    'retrofit_retrofit',
    'retrofit_retrofit-adapters.rxjava',
    'wro4j_wro4j-extensions',
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
hist = {}
tests = []
conf_candidates = []
min_conf_candidate_runtime_idx_tup_list = []
setup_tm_dict = {}
tri_tm_dict = {}
tri_fr_dict = {}


def load_setup(proj: str,
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
    mach_ts_dict = {}
    mach_tm_dict = {}
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
        mach_ts_dict[cur] = []
        mach_tm_dict[cur] = setup_tm
    return mach_arr, mach_ts_dict, mach_tm_dict


def get_fit(mach_tm_dict):
    price = 0
    b = 1 - a
    beta = 25.993775 / 3600
    for mach, per_runtime in mach_tm_dict.items():
        per_price = per_runtime * conf_prc_map[idx_conf_map[mach[0]]] / 3600
        price += per_price
    runtime = max(mach_tm_dict.values())
    fitness = a * beta * runtime + b * price
    return fitness, price


def organize_info(avg_tm_dict):
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


def fill_tri_info(avg_tm_dict):
    global tri_tm_dict, tri_fr_dict
    tri_tm_dict.clear()
    tri_fr_dict.clear()
    for t, info in avg_tm_dict.items():
        tri_tm_dict[t] = {itm[0]: itm[2] for itm in info}
        tri_fr_dict[t] = {itm[0]: itm[3] for itm in info}


def eva_schedule(ind):
    mach_ky = mapping(ind[:])
    if mach_ky in hist.keys():
        ind.price = hist[mach_ky].price
        ind.time_para = hist[mach_ky].time_para
        ind.mach_ts_dict = hist[mach_ky].mach_ts_dict
        ind.fitness.values = hist[mach_ky].fitness.values
        return
    mach_arr, mach_ts_dict, mach_tm_dict = anal_machs(ind[:])
    sorted_tup_list = sorted(min_conf_candidate_runtime_idx_tup_list, key=lambda x: x[0], reverse=True)
    for tup in sorted_tup_list:
        idx = tup[1]
        key = tests[idx]
        val = conf_candidates[idx]
        conf_inner_idx_map = {conf_idx_map[cand[thrott_conf_idx]]: i for i, cand in enumerate(val)}
        conv_calc_map = {mach: conf_inner_idx_map[mach[0]] for mach in mach_arr}
        tst = f'{key[0]}#{key[1]}'
        min_fitness = float('inf')
        min_time_para = float('inf')
        min_mach = None
        for mach in mach_arr:
            inner_idx = conv_calc_map[mach]
            cur_avg_tm = val[inner_idx][avg_time_idx]
            mach_tm_dict[mach] += cur_avg_tm
            fitness, _ = get_fit(mach_tm_dict)
            if (fitness < min_fitness) or (fitness == min_fitness and max(mach_tm_dict.values()) < min_time_para):
                min_fitness = fitness
                min_time_para = max(mach_tm_dict.values())
                min_mach = mach
            mach_tm_dict[mach] -= cur_avg_tm
        mach_tm_dict[min_mach] += val[conv_calc_map[min_mach]][avg_time_idx]
        mach_ts_dict[min_mach].append(tst)
    ind.time_para = max(mach_tm_dict.values())
    ind.mach_ts_dict = mach_ts_dict
    fit, ind.price = get_fit(mach_tm_dict)
    ind.fitness.values = (fit,)
    hist[mach_ky] = ind


def recalculate_ind(ind):
    _, _, mach_tm_dict = anal_machs(ind[:])
    min_fr = 100
    max_fr = 0
    mul_rt = 1
    if ind.fitness.values[0] == float('inf'):
        return
    for mach, test_set in ind.mach_ts_dict.items():
        conf = idx_conf_map[mach[0]]
        for test in test_set:
            temp = test.split('#')
            key = (temp[0], temp[1])
            mach_tm_dict[mach] += tri_tm_dict[key][conf]
            cur_fr = tri_fr_dict[key][conf]
            mul_rt *= 1 - cur_fr
            if min_fr > cur_fr:
                min_fr = cur_fr
            if max_fr < cur_fr:
                max_fr = cur_fr
    ind.min_fr = min_fr
    ind.max_fr = max_fr
    ind.pro_fr = 1 - mul_rt
    ind.time_para = max(mach_tm_dict.values())
    ind.time_seq = sum(mach_tm_dict.values())
    fit, ind.price = get_fit(mach_tm_dict)
    ind.fitness.values = (fit,)


def ga(gene_len):
    pop = toolbox.population(n=88)
    for i in range(confs_num):
        pop.append(creator.Individual([i for _ in range(gene_len)]))
    list(map(toolbox.evaluate, pop))
    pop = sorted(pop, key=lambda x: (x.fitness.values[0], x.time_para))
    if gene_len == 1:
        return pop[0]
    for _ in range(iters):
        offspring = list(map(toolbox.clone, pop[:35]))
        for chd1, chd2 in zip(pop[::2], pop[1::2]):
            toolbox.mate(chd1, chd2, 0.5)
        for gen in pop:
            toolbox.mutate(gen)
        list(map(toolbox.evaluate, pop))
        pop = sorted(pop, key=lambda x: (x.fitness.values[0], x.time_para))
        pop[:] = offspring + list(map(toolbox.clone, pop[:65]))
        pop = sorted(pop, key=lambda x: (x.fitness.values[0], x.time_para))
    return pop[0]


def bruteforce(gene_len):
    combs = combinations(range(gene_len * confs_num),
                         gene_len)
    mini = float('inf')
    mini_ind = None
    for comb in combs:
        machs = [i % confs_num for i in comb]
        ind = creator.Individual(machs)
        eva_schedule(ind)
        if ind.fitness.values[0] <= mini:
            mini = ind.fitness.values[0]
            mini_ind = ind
    recalculate_ind(mini_ind)
    return mini_ind


def reco_base(proj,
              df,
              gene_len):
    base_pop = []
    for i in range(confs_num):
        base_pop.append(creator.Individual([i for _ in range(gene_len)]))
    list(map(toolbox.evaluate, base_pop))
    for ind in base_pop:
        recalculate_ind(ind)
        df.loc[len(df.index)] = [
            proj,
            gene_len,
            idx_conf_map[ind[:][0]],
            ind.time_seq,
            ind.time_para,
            ind.price,
            ind.min_fr,
            ind.max_fr,
            ind.pro_fr,
            ind.fitness.values[0]]


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
    print(f'Probability failure rate: {ind.pro_fr}')
    print(f'Fitness: {fit}')
    print(ind.mach_ts_dict.keys())


def record_ind(ind,
               proj,
               cg,
               df,
               period):
    fit = ind.fitness.values[0]
    if fit == float('inf'):
        df.loc[len(df.index)] = [
            proj,
            cg,
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            period
        ]
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
        ind.pro_fr,
        fit,
        period
    ]


if __name__ == '__main__':
    # a = 0, 0.05, 0.1, 0.15, 0.2, 0.25,
    #     0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
    #     0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
    #     0.9, 0.95, 1
    prog_start = time.time()
    a = 0.5
    round_cnt = 30
    group_ky = 'ig'
    groups_map = {
        'non_ig': ['', False],
        'ig': ['_ig', True]
    }
    num_of_machine = [1, 2, 4, 6, 8, 10, 12]
    sub = f'rnd{round_cnt}'
    tot_test_num = 0
    for proj_name in proj_names:
        ext_dat_df = pd.DataFrame(None,
                                  columns=['project', 'machines_num',
                                           'confs_num', 'confs',
                                           'time_seq', 'time_parallel',
                                           'price',
                                           'min_failure_rate',
                                           'max_failure_rate',
                                           'probability_failure_rate',
                                           'fitness', 'period'])
        base_df = pd.DataFrame(None,
                               columns=['project', 'machines_num',
                                        'conf', 'time_seq',
                                        'time_parallel', 'price',
                                        'min_failure_rate',
                                        'max_failure_rate',
                                        'probability_failure_rate',
                                        'fitness'])
        avg_tm_dict = preproc(f'preproc_10/{round_cnt}', proj_name)
        if avg_tm_dict is None: continue
        organize_info(avg_tm_dict)
        fill_tri_info(preproc('preproc_300', proj_name))
        load_setup(proj_name,
                   groups_map[group_ky][1])
        tot_test_num += len(tests)
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 0, 11)
        toolbox.register("mate", tools.cxUniform)
        toolbox.register("evaluate", eva_schedule)
        for mach_num in num_of_machine:
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, mach_num)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("mutate", tools.mutUniformInt, low=0, up=11, indpb=1 / mach_num)
            # ---------------- Calculate Baseline ----------------
            # hist.clear()
            # reco_base(proj_name,
            #           base_df,
            #           mach_num)
            # ------------------------ End -----------------------
            hist.clear()
            t1 = time.time()
            # best_ind = bruteforce(mach_num)
            best_ind = ga(mach_num)
            recalculate_ind(best_ind)
            t2 = time.time()
            tt = t2 - t1
            category = mach_num
            print(f'-------------------- {proj_name}-{mach_num} --------------------')
            print_ind(best_ind,
                      tt)
            record_ind(best_ind,
                       proj_name,
                       category,
                       ext_dat_df,
                       tt)
        resu_sub_path = f'{resu_path}/{sub}'
        if not os.path.exists(resu_sub_path):
            os.mkdir(resu_sub_path)
        ext_dat_df.to_csv(f'{resu_sub_path}/{proj_name}.csv', sep=',', header=True, index=False)
        # base_sub_path = f'{base_path}/{group_ky}/a{a}'
        # if not os.path.exists(base_sub_path):
        #     os.mkdir(base_sub_path)
        # base_df.to_csv(f'{base_sub_path}/{proj_name}.csv', sep=',', header=True, index=False)
    print(f'[Total time] {time.time() - prog_start} s')
    print(f'[Test num] {tot_test_num}')
