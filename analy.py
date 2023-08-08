import os
import random
import heapq
import json
import pandas as pd
import numpy as np
import copy
from solver_time import OptTimeSolver
from pprint import pprint, pformat

random.seed(0)

dir_path = '../raft-csvs/'
res_path = 'preprocessed_data_results/'
idx_tst_path = 'index_test_map/'
np_path = 'np_solver_arr/'
distributed_cond_path = 'test_distribution_cond/'
unmatched_path = 'failure_rate_unmatched_tests/'
proj_names = [
    # 'activiti_dot',
    # 'assertj-core_dot',
    'carbon-apimgt_analyzer-modules.org.wso2.carbon.apimgt.throttling.siddhi.extension',
    'commons-exec_dot',
    'db-scheduler_dot',
    'delight-nashorn-sandbox_dot',
    'elastic-job-lite_dot',
    'elastic-job-lite_elastic-job-lite-core',
    'esper_examples.rfidassetzone',
    # 'fastjson_dot',
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
    # 'spring-boot_dot',
    'timely_server',
    'wro4j_wro4j-extensions',
    'yawp_yawp-testing.yawp-testing-appengine',
    'zxing_dot'
]

confs_num = 12
proj = ''
err_dat = 'error_tests.csv'
config_prc_map = {}
config_idx_map = {}


def df_to_map(df, keyName, valueName):
    res = {}
    for idx, row in df.iterrows():
        res[row[keyName]] = row[valueName]
    return res


# rec_dict: key=(classname, methodname, throttConf); value=[outerRound, numOfPassing, timeOfSum]
def ext_dat(csv_dir, proj_name, rec_dict):
    global config_prc_map
    dat = pd.read_csv(csv_dir + proj_name + '.csv', low_memory=False)
    config_prc_df = pd.read_csv('config_price.csv')
    config_prc_map = df_to_map(config_prc_df, 'config', 'price_hour')
    config_lst = config_prc_map.keys()
    for _, row in dat.iterrows():
        throttConf = row['throttConf']
        if throttConf not in config_lst or row['methodname'].__class__ != str:
            continue
        tm = row['time']
        if tm.__class__ == str and tm.find(','):
            tm = float(row['time'].replace(',', ''))
        key = (row['classname'], row['methodname'], throttConf)
        if key in rec_dict.keys():
            rec_dict[key][0] += 1
            if row['result'] != 'pass':
                continue
            rec_dict[key][1] += 1
            rec_dict[key][2] += tm
        else:
            if row['result'] != 'pass':
                rec_dict[key] = [1, 0, 0]
                continue
            rec_dict[key] = [1, 1, tm]


thrott_conf_idx = 0
outer_round_idx = 1
avg_time_idx = 2
failure_rate_idx = 3
price_idx = 4


# avg_tm_dict: key=(classname, methodname); value=[throttConf, outerRound, avgTime, failureRate, price]
def cal_dat(proj_name, rec_dict):
    avg_tm_dict = {}
    idx_tst_dict = {}
    tst_idx_dict = {}
    tst_tm_arr = []
    idx = 0
    errs = []
    for key, val in rec_dict.items():
        tst = (key[0], key[1])
        if val[1] != 0:
            avg_time = val[2] / val[1]
        else:
            err = [proj_name, key[0] + '#' + key[1], key[2] + '\n']
            errs.append(err)
            continue
        failure_rate = 1 - val[1] / val[0]
        prc = config_prc_map[key[2]] * val[2] / 3600
        conf_idx = config_idx_map[key[2]]
        if tst not in avg_tm_dict.keys():
            avg_tm_dict[tst] = [[key[2], val[0], avg_time, failure_rate, prc]]
            # 添加idx->test的map
            idx_tst_dict[idx] = tst
            tst_idx_dict[tst] = idx
            tst_tm_arr.append([0 for _ in range(confs_num)])
            tst_tm_arr[idx][conf_idx] = avg_time
            idx += 1
        else:
            avg_tm_dict[tst].append([key[2], val[0], avg_time, failure_rate, prc])
            tst_tm_arr[tst_idx_dict[tst]][conf_idx] = avg_time
    for err in errs:
        with open(err_dat, 'a') as f:
            f.write(','.join(err))
    return avg_tm_dict, idx_tst_dict, tst_tm_arr


def analysis_machs_list(machs: list):
    config_idx_dict = {}
    idx_config_dict = {}
    idx_num_dict = {}
    idx = 0
    for m in machs:
        if m in config_idx_dict.keys():
            idx_num_dict[config_idx_dict[m]] += 1
            continue
        config_idx_dict[m] = idx
        idx_config_dict[idx] = m
        idx_num_dict[idx] = 1
        idx += 1
    return idx, config_idx_dict, idx_config_dict, idx_num_dict


versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12']


def get_alloc(a: float, b: float, machs: list, fr: float, avg_tm_dict: dict):
    config_num, config_idx_dict, idx_config_dict, idx_num_dict = analysis_machs_list(machs)
    unmatched_fr_test = {}
    idx_time_list_dict = {}
    idx_sum_time_dict = {}
    configs = config_idx_dict.keys()
    mach_test_dict = {}
    mach_time_dict = {}
    min_fr = 100
    max_fr = 0
    for key, val in idx_num_dict.items():
        if val == 1:
            mach_test_dict[idx_config_dict[key]] = []
            mach_time_dict[idx_config_dict[key]] = 0
            continue
        tmp = val - 1
        while tmp >= 0:
            mach_test_dict[idx_config_dict[key] + ':' + versions[tmp]] = []
            mach_time_dict[idx_config_dict[key] + ':' + versions[tmp]] = 0
            tmp -= 1
    price = 0
    time_seq = 0
    for key, val in avg_tm_dict.items():
        node = key[0] + '#' + key[1]
        mini = float('inf')
        mini_conf = ''
        mini_time = 0
        mini_price = 0
        for item in val:
            thrott_conf = item[thrott_conf_idx]
            cur_fr = item[failure_rate_idx]
            if thrott_conf not in configs or cur_fr > fr:
                continue
            cost = a * item[price_idx] + b * item[avg_time_idx]
            if cost < mini:
                mini = cost
                mini_conf = thrott_conf
                mini_price = item[price_idx]
                mini_time = item[avg_time_idx]
                if cur_fr > max_fr:
                    max_fr = cur_fr
                if cur_fr < min_fr:
                    min_fr = cur_fr
        if mini_conf == '':
            tmp_list = sorted(val, key=lambda x: a * x[price_idx] + b * x[avg_time_idx])
            i = 0
            while True:
                if tmp_list[i][thrott_conf_idx] in configs:
                    break
                else:
                    i += 1
            item = tmp_list[i]
            mini_conf = item[thrott_conf_idx]
            mini_price = item[price_idx]
            mini_time = item[avg_time_idx]
            cur_fr = item[failure_rate_idx]
            unmatched_fr_test[node] = cur_fr
            if cur_fr > max_fr:
                max_fr = cur_fr
            if cur_fr < min_fr:
                min_fr = cur_fr
        mini_conf_idx = config_idx_dict[mini_conf]
        if idx_num_dict[mini_conf_idx] == 1:
            mach_test_dict[mini_conf].append(node)
            mach_time_dict[mini_conf] += mini_time
        else:
            if mini_conf_idx in idx_time_list_dict.keys():
                idx_time_list_dict[mini_conf_idx].append([node, mini_time])
                idx_sum_time_dict[mini_conf_idx] += mini_time
            else:
                idx_time_list_dict[mini_conf_idx] = [[node, mini_time]]
                idx_sum_time_dict[mini_conf_idx] = mini_time
        time_seq += mini_time
        price += mini_price
    for key, val in idx_time_list_dict.items():
        conf = idx_config_dict[key]
        num = idx_num_dict[key]
        avg = idx_sum_time_dict[key] / num
        sort_time_list = sorted(val, key=lambda x: x[1])
        i = 0
        idx = num - 1
        while idx >= 0 and i < len(val):
            conf_ver = conf + ':' + versions[idx]
            tst = sort_time_list[i][0]
            tm = sort_time_list[i][1]
            if mach_time_dict[conf_ver] + tm <= avg:
                mach_time_dict[conf_ver] += tm
                mach_test_dict[conf_ver].append(tst)
                i += 1
            else:
                idx -= 1
    time_parallel = max(mach_time_dict.values())
    return time_seq, time_parallel, price, min_fr, max_fr, mach_test_dict, unmatched_fr_test


def machs_fr_lim(a: float, b: float, machs: list, fr: float, modu: str, avg_tm_dict: dict,
                 tst_tm_arr=None, idx_tst_dict=None):
    if modu == 'cheap' or modu == 'fast_seq':
        return get_alloc(a, b, machs, fr, avg_tm_dict)
    else:
        return get_alloc_z3(a, b, machs, fr, avg_tm_dict, tst_tm_arr, idx_tst_dict)


def dijkstra(graph, s):
    dist = {v: float('inf') for v in graph}
    dist[s] = 0
    queue = [(0, s)]
    while queue:
        cur_dist, cur_v = heapq.heappop(queue)
        if cur_dist > dist[cur_v]: continue
        for neighbor, weight in graph[cur_v].items():
            new_dist = cur_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))
    return dist


# -------------------------------------------------Genetic algorithm----------------------------------------------------
class Individual:

    def __init__(self, machs: list, time_seq: float, time_parallel: float, price: float, min_fr: float, max_fr: float,
                 mach_test_dict: dict, unmatched_fr_test: dict):
        self.machs = machs
        self.time_seq = time_seq
        self.time_parallel = time_parallel
        self.price = price
        self.min_fr = min_fr
        self.max_fr = max_fr
        self.mach_test_dict = mach_test_dict
        self.unmatched_fr_test = unmatched_fr_test
        self.score = 0


class GA:

    def __init__(self, a, b, fr, modu, avail_configs, avg_tm_dict, pop_size, gene_length, max_iter,
                 tst_tm_arr=None, idx_tst_dict=None):
        self.a = a
        self.b = b
        self.fr = fr
        self.modu = modu
        self.avail_configs_num = confs_num
        self.avail_configs = avail_configs
        self.avg_tm_dict = avg_tm_dict
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.max_iter = max_iter
        self.memo = {}
        self.population = []
        self.tst_tm_arr = tst_tm_arr
        self.idx_tst_dict = idx_tst_dict

    def init_pop(self):
        for _ in range(self.pop_size):
            machs = [random.choice(self.avail_configs) for _ in range(self.gene_length)]
            self.population.append(machs)
            self.maintain_memo(machs)

    # a * price + b * time
    def cal_score(self, ind: Individual):
        ind.score = 1 - (self.a * ind.price + self.b * ind.time_seq)

    def selection(self):
        pop = sorted(self.population, key=lambda ind: self.memo[self.chg(ind)].score, reverse=True)
        return pop[:int(0.2 * len(pop))]

    def crossover(self, parents):
        cnt = self.pop_size - len(parents)
        tmp = 0
        i = 0
        end = self.gene_length - 1
        children = []
        while i < cnt:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            if p1 != p2 or tmp == 10:
                pt = random.randint(1, end)
                machs1 = p1[0:pt] + p2[pt:]
                machs2 = p2[0:pt] + p1[pt:]
                children.append(machs1)
                children.append(machs2)
                self.maintain_memo(machs1)
                self.maintain_memo(machs2)
                i += 2
                tmp = 0
            else:
                tmp += 1
        return children

    def mutation(self, children):
        chd_num = len(children)
        for i in range(chd_num):
            pt = random.randint(0, self.gene_length - 1)
            cfg = random.choice(self.avail_configs)
            children[i][pt] = cfg
            self.maintain_memo(children[i])

    def run(self):
        for i in range(self.max_iter):
            parents = self.selection()
            children = self.crossover(parents)
            self.mutation(children)
            self.population = parents + children

    def print_best(self):
        pop = sorted(self.population, key=lambda ind: self.memo[self.chg(ind)].score, reverse=True)
        ind = self.memo[self.chg(pop[0])]
        print('Machine list: ' + ', '.join(ind.machs))
        print('Time seq: ' + str(ind.time_seq))
        print('Time parallel: ' + str(ind.time_parallel))
        print('Price: ' + str(ind.price))
        print('Minimum failure rate: ' + str(ind.min_fr))
        print('Maximum failure rate: ' + str(ind.max_fr))
        print('Testing the correspondence between the machine and the test set: ')
        print(ind.mach_test_dict.keys())
        print('Failure rate unmatched tests: ')
        pprint(ind.unmatched_fr_test)

    def rec_best(self, proj_name, cate):
        global ext_dat_df
        pop = sorted(self.population, key=lambda ind: self.memo[self.chg(ind)].score, reverse=True)
        ind = self.memo[self.chg(pop[0])]
        confs = list(set(ind.machs))
        ext_dat_df.loc[len(ext_dat_df.index)] = [proj_name, cate, len(confs), confs, ind.time_seq, ind.time_parallel,
                                                 ind.price, ind.min_fr, ind.max_fr]
        dis_folder = distributed_cond_path + proj_name
        if not os.path.exists(dis_folder):
            os.makedirs(dis_folder)
        with open(dis_folder + '/cate' + cate, 'w') as f:
            f.write(pformat(ind.mach_test_dict))
        f.close()
        unmatched_folder = unmatched_path + proj_name
        if not os.path.exists(unmatched_folder):
            os.makedirs(unmatched_folder)
        with open(unmatched_folder + '/cate' + cate, 'w') as f:
            f.write(pformat(ind.unmatched_fr_test))

    def maintain_memo(self, machs: list):
        conf_tup = self.chg(machs)
        if conf_tup not in self.memo.keys():
            time_seq, time_parallel, price, min_fr, max_fr, mach_test_dict, unmatched_fr_test = \
                machs_fr_lim(self.a, self.b, machs, self.fr, self.modu, self.avg_tm_dict,
                             self.tst_tm_arr, self.idx_tst_dict)
            new_ind = Individual(copy.deepcopy(machs), time_seq, time_parallel, price, min_fr, max_fr,
                                 mach_test_dict, unmatched_fr_test)
            self.cal_score(new_ind)
            self.memo[conf_tup] = new_ind

    def chg(self, machs: list):
        conf_list = [0 for _ in range(self.avail_configs_num)]
        for conf in machs:
            conf_list[config_idx_map[conf]] += 1
        return tuple(conf_list)


def get_alloc_z3(a: float, b: float, machs: list, fr: float,
                 avg_tm_dict: dict, tst_tm_arr: np.array, idx_tst_dict: dict):
    conf_num_list = [0 for _ in range(confs_num)]
    mach_test_dict = {}
    unmatched_fr_test = {}
    is_fir = True
    tst_times = np.array([])
    confs = []
    idx_idx = {}
    names = ['' for _ in range(len(machs))]
    for m, mach in enumerate(machs):
        mach_idx = config_idx_map[mach]
        conf_num_list[mach_idx] += 1
        confs.append(mach_idx)
        idx_idx[mach_idx + confs_num*(conf_num_list[mach_idx]-1)] = m
        if is_fir:
            tst_times = tst_tm_arr[:, mach_idx]
            is_fir = False
            continue
        tst_times = np.c_[tst_times, tst_tm_arr[:, mach_idx]]
    idx_config_map = {v: k for k, v in config_idx_map.items()}
    for ii in range(confs_num):
        n = conf_num_list[ii]
        if n == 1:
            var = idx_config_map[ii]
            mach_test_dict[var] = []
            names[idx_idx[ii]] = var
        elif n > 1:
            x = n - 1
            while x >= 0:
                jj = ii + confs_num * x
                var = idx_config_map[ii] + ':' + versions[x]
                mach_test_dict[var] = []
                names[idx_idx[jj]] = var
                x -= 1
    tst_num = len(tst_tm_arr)
    solver = OptTimeSolver(tst_times=tst_times, mach_num=len(machs), tst_num=tst_num)
    res_tsts, res_mach_tms = solver.figure_out()
    res_tms = [float(t.as_fraction()) for t in res_mach_tms]
    time_seq = sum(res_tms)
    time_parallel = max(res_tms)
    price = 0
    min_fr = 100
    max_fr = 0
    for i in range(tst_num):
        jj = np.where(np.array(res_tsts[i]))[0][0]
        conf_idx = confs[jj]
        tst = idx_tst_dict[i]
        node = tst[0] + '#' + tst[1]
        mach_test_dict[names[jj]].append(node)
        info = avg_tm_dict[tuple(tst)][mp_list[i][conf_idx]]
        price += info[price_idx]
        cur_fr = info[failure_rate_idx]
        if cur_fr > max_fr:
            max_fr = cur_fr
        if cur_fr < min_fr:
            min_fr = cur_fr
        if cur_fr > fr:
            unmatched_fr_test[node] = cur_fr
    return time_seq, time_parallel, price, min_fr, max_fr, mach_test_dict, unmatched_fr_test


def get_confidx_listidx_map_list(tst_num: int, avg_tm_dict: dict, idx_tst_dict: dict):
    map_list = [{} for _ in range(tst_num)]
    for i in range(tst_num):
        tst = idx_tst_dict[i]
        conf_info = avg_tm_dict[tuple(tst)]
        info_num = len(conf_info)
        for j in range(info_num):
            conf = conf_info[j][thrott_conf_idx]
            map_list[i][config_idx_map[conf]] = j
    return map_list


if __name__ == '__main__':
    global mp_list
    num_of_machines = [2, 4, 6, 8, 10, 12]
    pct_of_failure_rate = [0, 0.2, 0.4, 0.6, 0.8, 1]
    chp_or_fst = {'fast_para': [0, 1]}
    # chp_or_fst = {'cheap': [1, 0], 'fast_seq': [0, 1]}
    a = 1
    b = 0
    avail_configs = [
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
        '01noThrottling.sh'
    ]
    for i in range(confs_num):
        config_idx_map[avail_configs[i]] = i
    cnt = 1
    for proj_name in proj_names:
        res_file_name = res_path + proj_name
        map_file_name = idx_tst_path + proj_name
        np_name = np_path + proj_name + '.npy'
        ext_dat_df = pd.DataFrame(None,
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
        if os.path.exists(res_file_name):
            print(proj_name + ' has already been preprocessed.')
            with open(res_file_name, 'r') as f:
                tmp_dict = json.load(f)
            avg_tm_dict = {tuple(eval(k)): v for k, v in tmp_dict.items()}
            with open(map_file_name, 'r') as f:
                it_dict = json.load(f)
            tst_tm_arr = np.load(np_name)
            idx_tst_dict = {int(k): v for k, v in it_dict.items()}
        else:
            rec_dict = {}
            filenames = os.listdir(dir_path)
            for f in filenames:
                ext_dat(dir_path + f + '/', proj_name, rec_dict)
                print(proj_name + '#' + str(cnt) + '... ...')
                cnt += 1
            cnt = 1
            avg_tm_dict, idx_tst_dict, tst_tm_arr = cal_dat(proj_name=proj_name, rec_dict=rec_dict)
            np.save(np_name, np.array(tst_tm_arr))
            tmp_dict = {str(k): v for k, v in avg_tm_dict.items()}
            with open(res_file_name, 'w') as f:
                json.dump(tmp_dict, f)
            with open(map_file_name, 'w') as f:
                json.dump(idx_tst_dict, f)

        mp_list = get_confidx_listidx_map_list(len(idx_tst_dict.keys()), avg_tm_dict, idx_tst_dict)
        for mach in num_of_machines:
            for pct in pct_of_failure_rate:
                for modu, fac in chp_or_fst.items():
                    ga = GA(a=fac[0], b=fac[1], fr=pct, modu=modu, avail_configs=avail_configs,
                            avg_tm_dict=avg_tm_dict, pop_size=100, gene_length=mach, max_iter=100,
                            tst_tm_arr=tst_tm_arr, idx_tst_dict=idx_tst_dict)
                    ga.init_pop()
                    ga.run()
                    cate = str(mach) + '-' + str(pct) + '-' + modu
                    print('------------------------------------   ' + cate + '   ------------------------------------')
                    ga.print_best()
                    ga.rec_best(proj_name, cate)
        csv_name = 'ext_dat/' + proj_name + '.csv'
        # csv_name = 'ext_dat_z3/' + proj_name + '.csv'
        ext_dat_df.to_csv(csv_name, sep=',', index=False, header=True)
