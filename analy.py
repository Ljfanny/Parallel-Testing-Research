import copy
import os
import random
import json
import time
from pprint import pprint, pformat

import numpy as np
import pandas as pd

from preproc import preproc, conf_prc_map

random.seed(0)
resu_path = 'ext_dat/'
setup_rec_path = 'setup_time_rec/'
tst_alloc_rec_path = 'test_allocation_rec/'
baseline_path = 'baseline_dat/'
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
versions = ['v' + str(i) for i in range(12)]
err_dat = 'error_tests.csv'
conf_idx_map = {k: v for v, k in enumerate(avail_confs)}
idx_conf_map = {k: v for k, v in enumerate(avail_confs)}
thrott_conf_idx = 0
outer_round_idx = 1
avg_time_idx = 2
failure_rate_idx = 3
price_idx = 4


def load_setup_time_map(proj: str,
                        cond: bool):
    if cond:
        return {k: float(0) for k in avail_confs}
    file = setup_rec_path + proj
    with open(file, 'r') as f:
        setup_time_map = json.load(f)
    return setup_time_map


def analysis_machs(machs: list,
                   setup_tm_dict: dict):
    price = 0
    mach_arr = []
    mach_test_dict = {}
    mach_time_dict = {}
    multi_dict = {}
    conf_machs_map = {k: [] for k in set(machs)}
    num_arr = list(mapping(machs))
    bool_arr = [False if num <= 1 else True for num in num_arr]
    for i, idx in enumerate(machs):
        conf = idx_conf_map[idx]
        setup_tm = setup_tm_dict[conf]
        prc = setup_tm * conf_prc_map[conf] / 3600
        if bool_arr[idx]:
            num_arr[idx] -= 1
            if idx not in multi_dict.keys():
                multi_dict[idx] = []
            cur = (idx, num_arr[idx])
            mach_arr.append(cur)
            conf_machs_map[idx].append(num_arr[idx])
        else:
            cur = (idx, -1)
            mach_arr.append(cur)
        mach_test_dict[cur] = []
        mach_time_dict[cur] = setup_tm
        price += prc
    return price, mach_arr, mach_test_dict, mach_time_dict, multi_dict, conf_machs_map


def get_fst_alloc(machs: list,
                  fr: float,
                  avg_tm_dict: dict,
                  setup_tm_dict: dict):
    price, mach_arr, mach_test_dict, mach_time_dict, _, _ = analysis_machs(machs,
                                                                           setup_tm_dict)
    confs = set(machs)
    min_fr = 100
    max_fr = 0
    for key, val in avg_tm_dict.items():
        tst = key[0] + '#' + key[1]
        min_para_time = float('inf')
        min_mach = None
        cur_fr = 0
        tmp_map = {}
        for item in val:
            thrott_conf = conf_idx_map[item[thrott_conf_idx]]
            if thrott_conf not in confs:
                continue
            tmp_map[thrott_conf] = [item[avg_time_idx], item[price_idx], item[failure_rate_idx]]
        fr_arr = list(zip(*tmp_map.values()))[-1]
        for m in mach_arr:
            idx = m[0]
            arr = tmp_map[idx]
            if mach_time_dict[m] + arr[0] < min_para_time and arr[2] <= fr:
                min_para_time = mach_time_dict[m] + arr[0]
                min_mach = m
                cur_fr = arr[2]
        if min_mach is None:
            min_idx = [k for k, v in tmp_map.items() if v[2] == min(fr_arr)][0]
            cur_fr = tmp_map[min_idx][2]
            filtered_dict = {k: v for k, v in mach_time_dict.items() if k[0] == min_idx}
            min_mach = [k for k, v in mach_time_dict.items() if v == min(filtered_dict.values())][0]
        min_conf = min_mach[0]
        mach_test_dict[min_mach].append(tst)
        mach_time_dict[min_mach] += tmp_map[min_conf][0]
        if cur_fr > max_fr:
            max_fr = cur_fr
        if cur_fr < min_fr:
            min_fr = cur_fr
        price += tmp_map[min_conf][1]
    time_para = max(mach_time_dict.values())
    time_seq = sum(mach_time_dict.values())
    return time_seq, time_para, price, min_fr, max_fr, mach_test_dict


def get_chp_alloc(machs: list,
                  fr: float,
                  avg_tm_dict: dict,
                  setup_tm_dict: dict):
    price, mach_arr, mach_test_dict, mach_time_dict, multi_dict, conf_machs_map = analysis_machs(machs,
                                                                                                 setup_tm_dict)
    confs = set(machs)
    min_fr = 100
    max_fr = 0
    for key, val in avg_tm_dict.items():
        tst = key[0] + '#' + key[1]
        mini = float('inf')
        mini_conf = -1
        mini_time = 0
        mini_price = 0
        for item in val:
            thrott_conf = conf_idx_map[item[thrott_conf_idx]]
            cur_fr = item[failure_rate_idx]
            if thrott_conf not in confs or cur_fr > fr:
                continue
            if item[price_idx] < mini:
                mini = item[price_idx]
                mini_conf = thrott_conf
                mini_price = item[price_idx]
                mini_time = item[avg_time_idx]
                if cur_fr > max_fr:
                    max_fr = cur_fr
                if cur_fr < min_fr:
                    min_fr = cur_fr
        if mini_conf == -1:
            tmp_list = sorted(val, key=lambda x: x[failure_rate_idx])
            i = 0
            while True:
                if conf_idx_map[tmp_list[i][thrott_conf_idx]] in confs:
                    break
                else:
                    i += 1
            item = tmp_list[i]
            mini_conf = conf_idx_map[item[thrott_conf_idx]]
            mini_price = item[price_idx]
            mini_time = item[avg_time_idx]
            cur_fr = item[failure_rate_idx]
            if cur_fr > max_fr:
                max_fr = cur_fr
            if cur_fr < min_fr:
                min_fr = cur_fr
        ky = (mini_conf, -1)
        if ky in mach_test_dict.keys():
            mach_test_dict[ky].append(tst)
            mach_time_dict[ky] += mini_time
        else:
            multi_dict[mini_conf].append([tst, mini_time])
        price += mini_price
    for key, val in multi_dict.items():
        for tup in val:
            tst = tup[0]
            tm = tup[1]
            min_para_time = float('inf')
            min_ver = -1
            for ver in conf_machs_map[key]:
                ky = (key, ver)
                if mach_time_dict[ky] + tm < min_para_time:
                    min_para_time = mach_time_dict[ky] + tm
                    min_ver = ver
            min_mac = (key, min_ver)
            mach_time_dict[min_mac] += tm
            mach_test_dict[min_mac].append(tst)
    time_para = max(mach_time_dict.values())
    time_seq = sum(mach_time_dict.values())
    return time_seq, time_para, price, min_fr, max_fr, mach_test_dict


def sel_modu(machs: list,
             fr: float,
             modu: str,
             avg_tm_dict: dict,
             setup_tm_dict: dict):
    if modu == 'cheap':
        return get_chp_alloc(machs,
                             fr,
                             avg_tm_dict,
                             setup_tm_dict)
    else:
        return get_fst_alloc(machs,
                             fr,
                             avg_tm_dict,
                             setup_tm_dict)


# ----------------------------------------- Genetic algorithm --------------------------------------------
class Individual:

    def __init__(self,
                 machs,
                 time_seq,
                 time_para,
                 price,
                 min_fr,
                 max_fr,
                 mach_test_dict):
        self.machs = machs
        self.time_seq = time_seq
        self.time_para = time_para
        self.price = price
        self.min_fr = min_fr
        self.max_fr = max_fr
        self.mach_test_dict = mach_test_dict
        self.score = 0

    def print_ind(self,
                  period):
        print(f'Period: {period}')
        if self.score == float('inf'):
            print(f'Maximum failure rate: {self.max_fr}')
            return
        print('Machine list: ')
        pprint({(k, i): mapping(self.machs)[i] for i, k in enumerate(avail_confs)})
        print(f'Time seq: {self.time_seq}')
        print(f'Time parallel: {self.time_para}')
        print(f'Price: {self.price}')
        print(f'Minimum failure rate: {self.min_fr}')
        print(f'Maximum failure rate: {self.max_fr}')
        print('Testing the correspondence between the machine and the test set: ')
        print(self.mach_test_dict.keys())

    def record_ind(self,
                   subdir,
                   proj,
                   cg,
                   df,
                   period):
        dis_folder = tst_alloc_rec_path + subdir + '/' + proj
        if not os.path.exists(dis_folder):
            os.makedirs(dis_folder)
        if self.score == float('inf'):
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
                period
            ]
            with open(dis_folder + '/category' + cg, 'w'):
                pass
            return
        num_tup = mapping(self.machs)
        confs = set(self.machs)
        conf_num_map = {idx_conf_map[k]: num_tup[k] for k in confs}
        df.loc[len(df.index)] = [
            proj,
            cg,
            len(confs),
            conf_num_map,
            self.time_seq,
            self.time_para,
            self.price,
            self.min_fr,
            self.max_fr,
            period
        ]
        temp_dict = {idx_conf_map[k[0]] if k[1] == -1
                     else idx_conf_map[k[0]] + ':' + versions[k[1]]: v for k, v in self.mach_test_dict.items()}
        with open(dis_folder + '/category' + cg, 'w') as f:
            f.write(pformat(temp_dict))


def mapping(machs: list):
    conf_list = [0 for _ in range(confs_num)]
    for m in machs:
        conf_list[m] += 1
    return tuple(conf_list)


class GA:
    def __init__(self,
                 fr,
                 modu,
                 avg_tm_dict,
                 setup_tm_dict,
                 pop_size,
                 gene_length,
                 max_iter):
        self.fr = fr
        self.modu = modu
        self.avg_tm_dict = avg_tm_dict
        self.setup_tm_dict = setup_tm_dict
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.max_iter = max_iter
        self.memo = {}
        self.population = []

    def init_pop(self):
        for i in range(confs_num):
            machs = [i for _ in range(self.gene_length)]
            self.population.append(machs)
            self.maintain_memo(machs)
        rest_num = self.pop_size - confs_num
        for _ in range(rest_num):
            machs = [random.choice(range(confs_num)) for _ in range(self.gene_length)]
            self.population.append(machs)
            self.maintain_memo(machs)

    def selection(self):
        pop = sorted(self.population, key=lambda chd: self.memo[mapping(chd)].score)
        return pop[:int(0.2 * len(pop))]

    def crossover(self,
                  parents):
        cnt = self.pop_size - len(parents)
        temp = 0
        i = 0
        end = self.gene_length - 1
        children = []
        while i < cnt:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            if p1 != p2 or temp == 10:
                pt = random.randint(1, end)
                machs1 = p1[0:pt] + p2[pt:]
                machs2 = p2[0:pt] + p1[pt:]
                children.append(machs1)
                children.append(machs2)
                self.maintain_memo(machs1)
                self.maintain_memo(machs2)
                i += 2
                temp = 0
            else:
                temp += 1
        return children

    def mutation(self,
                 children):
        chd_num = len(children)
        for i in range(chd_num):
            mut_chance = True if random.random() < float(1) / len(children[i]) else False
            if mut_chance:
                pos = random.randint(0, self.gene_length - 1)
                children[i][pos] = random.choice(range(confs_num))
                self.maintain_memo(children[i])

    def run(self):
        for i in range(self.max_iter):
            if self.gene_length > 1:
                parents = self.selection()
                children = self.crossover(parents)
                self.mutation(children)
                self.population = parents + children
            else:
                self.mutation(self.population)

    def print_best(self,
                   period):
        pop = sorted(self.population, key=lambda chd: self.memo[mapping(chd)].score)
        ind = self.memo[mapping(pop[0])]
        ind.print_ind(period)

    def record_best(self,
                    subdir,
                    proj,
                    cg,
                    period):
        pop = sorted(self.population, key=lambda chd: self.memo[mapping(chd)].score)
        ind = self.memo[mapping(pop[0])]
        ind.record_ind(subdir,
                       proj,
                       cg,
                       ext_dat_df,
                       period)

    def maintain_memo(self,
                      machs: list):
        conf_tup = mapping(machs)
        if conf_tup not in self.memo.keys():
            time_seq, time_para, price, min_fr, max_fr, mach_test_dict = sel_modu(machs,
                                                                                  self.fr,
                                                                                  self.modu,
                                                                                  self.avg_tm_dict,
                                                                                  self.setup_tm_dict)
            new_ind = Individual(copy.deepcopy(machs),
                                 time_seq,
                                 time_para,
                                 price,
                                 min_fr,
                                 max_fr,
                                 mach_test_dict)
            if max_fr <= self.fr:
                if self.modu == 'cheap':
                    new_ind.score = price
                else:
                    new_ind.score = time_para
            else:
                new_ind.score = float('inf')
            self.memo[conf_tup] = new_ind


def record_baseline(subdir: str,
                    proj: str,
                    df,
                    obj):
    cg = str(obj.gene_length) + '-' + str(obj.fr) + '-' + obj.modu
    for i in range(confs_num):
        ind = obj.memo[mapping([i for _ in range(obj.gene_length)])]
        ind.record_ind(subdir,
                       proj,
                       cg,
                       df)


if __name__ == '__main__':
    num_of_machine = [1, 2, 4, 6, 8, 10, 12]
    pct_of_failure_rate = [0, 0.2, 0.4, 0.6, 0.8, 1]
    chp_or_fst = ['cheap', 'fast']

    whether_to_ignore_setup_cost = True
    modus = {True: 'excl_cost', False: 'incl_cost'}
    sub = modus[whether_to_ignore_setup_cost]

    dat_df_csv_path = baseline_path + sub
    if not os.path.exists(dat_df_csv_path):
        os.mkdir(dat_df_csv_path)
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
                                           'period']
                                  )
        dat_df_csv = dat_df_csv_path + '/' + proj_name + '.csv'
        whether_baseline_exist = os.path.exists(dat_df_csv)
        baseline_dat_df = copy.deepcopy(ext_dat_df)
        preproc_proj_dict = preproc(proj_name)
        preproc_mvn_dict = load_setup_time_map(proj_name,
                                               whether_to_ignore_setup_cost)
        for mach_num in num_of_machine:
            for pct in pct_of_failure_rate:
                for cho in chp_or_fst:
                    t1 = time.time()
                    ga = GA(fr=pct,
                            modu=cho,
                            avg_tm_dict=preproc_proj_dict,
                            setup_tm_dict=preproc_mvn_dict,
                            pop_size=100,
                            gene_length=mach_num,
                            max_iter=100)
                    ga.init_pop()
                    ga.run()
                    t2 = time.time()
                    tt = t2 - t1
                    if not whether_baseline_exist:
                        record_baseline('baseline_' + sub, proj_name,
                                        baseline_dat_df,
                                        ga)
                    category = str(mach_num) + '-' + str(pct) + '-' + cho
                    print('--------------------   ' + proj_name + '-' + category + '   --------------------')
                    ga.print_best(tt)
                    ga.record_best(sub,
                                   proj_name,
                                   category,
                                   tt)
        if not os.path.exists(resu_path + sub):
            os.mkdir(resu_path + sub)
        csv_name = resu_path + sub + '/' + proj_name + '.csv'
        ext_dat_df.to_csv(csv_name, sep=',', header=True, index=False)
        if not whether_baseline_exist:
            baseline_dat_df.to_csv(dat_df_csv, sep=',', header=True, index=False)
