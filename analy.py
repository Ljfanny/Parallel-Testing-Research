import os
import random
import copy
import pandas as pd
from pprint import pprint, pformat
from preproc import preproc, conf_prc_map
from proc_maven_log import read_setup_time_map

random.seed(0)
distributed_cond_path = 'test_distribution_cond/'
unmatched_path = 'failure_rate_unmatched_tests/'
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
    # 'fluent-logger-java_dot': setup time < 0?
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
versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12']
confs_num = 12
proj = ''
err_dat = 'error_tests.csv'
conf_idx_map = {}
thrott_conf_idx = 0
outer_round_idx = 1
avg_time_idx = 2
failure_rate_idx = 3
price_idx = 4


def analysis_machs(machs: list, setup_tm_dict: dict):
    price = 0
    mach_arr = copy.deepcopy(machs)
    mach_test_dict = {}
    mach_time_dict = {}
    multi_dict = {}
    conf_machs_map = {k: [] for k in list(set(machs))}
    bool_arr = [False for _ in range(confs_num)]
    num_arr = [0 for _ in range(confs_num)]
    idx_arr = []
    for m in machs:
        idx = conf_idx_map[m]
        num_arr[idx] += 1
        idx_arr.append(idx)
        if num_arr[idx] > 1: bool_arr[idx] = True
    for i, idx in enumerate(idx_arr):
        cur_mach = mach_arr[i]
        setup_tm = setup_tm_dict[cur_mach]
        prc = setup_tm * conf_prc_map[cur_mach]
        if bool_arr[idx]:
            num_arr[idx] -= 1
            if cur_mach not in multi_dict.keys(): multi_dict[cur_mach] = []
            mach_arr[i] = cur_mach + ':' + versions[num_arr[idx]]
            conf_machs_map[cur_mach].append(mach_arr[i])
            cur_mach = mach_arr[i]
        mach_test_dict[cur_mach] = []
        mach_time_dict[cur_mach] = setup_tm
        price += prc
    return price, mach_arr, mach_test_dict, mach_time_dict, multi_dict, conf_machs_map


def get_fst_alloc(machs: list, fr: float, avg_tm_dict: dict, setup_tm_dict: dict):
    price, mach_arr, mach_test_dict, mach_time_dict, _, _ = analysis_machs(machs, setup_tm_dict)
    unmatched_fr_test = {}
    confs = set(machs)
    min_fr = 100
    max_fr = 0
    for key, val in avg_tm_dict.items():
        tst = key[0] + '#' + key[1]
        min_para_time = float('inf')
        min_mach = ''
        min_conf = ''
        cur_fr = 0
        tmp_map = {}
        for item in val:
            thrott_conf = item[thrott_conf_idx]
            if thrott_conf not in confs:
                continue
            tmp_map[thrott_conf] = [item[avg_time_idx], item[price_idx], item[failure_rate_idx]]
        fr_arr = list(zip(*tmp_map.values()))[-1]
        for m in mach_arr:
            conf = m if m.find(':') == -1 else m[:m.index(':')]
            arr = tmp_map[conf]
            if mach_time_dict[m] + arr[0] < min_para_time and arr[2] <= fr:
                min_para_time = mach_time_dict[m] + arr[0]
                min_mach = m
                min_conf = conf
                cur_fr = arr[2]
        if min_mach == '':
            unmatched_fr_test[tst] = min(fr_arr)
            min_conf = [k for k, v in tmp_map.items() if v[2] == min(fr_arr)][0]
            cur_fr = tmp_map[min_conf][2]
            filtered_dict = {k: v for k, v in mach_time_dict.items() if k.find(min_conf) != -1}
            min_mach = [k for k, v in mach_time_dict.items() if v == min(filtered_dict.values())][0]
        mach_test_dict[min_mach].append(tst)
        mach_time_dict[min_mach] += tmp_map[min_conf][0]
        if cur_fr > max_fr:
            max_fr = cur_fr
        if cur_fr < min_fr:
            min_fr = cur_fr
        price += tmp_map[min_conf][1]
    time_parallel = max(mach_time_dict.values())
    time_seq = sum(mach_time_dict.values())
    return time_seq, time_parallel, price, min_fr, max_fr, mach_test_dict, unmatched_fr_test


def get_chp_alloc(machs: list, fr: float, avg_tm_dict: dict, setup_tm_dict: dict):
    price, mach_arr, mach_test_dict, mach_time_dict, multi_dict, conf_machs_map = analysis_machs(machs, setup_tm_dict)
    unmatched_fr_test = {}
    confs = set(machs)
    min_fr = 100
    max_fr = 0
    for key, val in avg_tm_dict.items():
        tst = key[0] + '#' + key[1]
        mini = float('inf')
        mini_conf = ''
        mini_time = 0
        mini_price = 0
        for item in val:
            thrott_conf = item[thrott_conf_idx]
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
        if mini_conf == '':
            tmp_list = sorted(val, key=lambda x: x[failure_rate_idx])
            i = 0
            while True:
                if tmp_list[i][thrott_conf_idx] in confs:
                    break
                else:
                    i += 1
            item = tmp_list[i]
            mini_conf = item[thrott_conf_idx]
            mini_price = item[price_idx]
            mini_time = item[avg_time_idx]
            cur_fr = item[failure_rate_idx]
            unmatched_fr_test[tst] = cur_fr
            if cur_fr > max_fr:
                max_fr = cur_fr
            if cur_fr < min_fr:
                min_fr = cur_fr
        if mini_conf in mach_test_dict.keys():
            mach_test_dict[mini_conf].append(tst)
            mach_time_dict[mini_conf] += mini_time
        else:
            multi_dict[mini_conf].append([tst, mini_time])
        price += mini_price
    for key, val in multi_dict.items():
        for tup in val:
            tst = tup[0]
            tm = tup[1]
            min_para_time = float('inf')
            min_mac = ''
            for mac in conf_machs_map[key]:
                if mach_time_dict[mac] + tm < min_para_time:
                    min_para_time = mach_time_dict[mac] + tm
                    min_mac = mac
            mach_time_dict[min_mac] += tm
            mach_test_dict[min_mac].append(tst)
    time_parallel = max(mach_time_dict.values())
    time_seq = sum(mach_time_dict.values())
    return time_seq, time_parallel, price, min_fr, max_fr, mach_test_dict, unmatched_fr_test


def machs_fr_lim(machs: list, fr: float, modu: str, avg_tm_dict: dict, setup_tm_dict: dict):
    if modu == 'cheap':
        return get_chp_alloc(machs, fr, avg_tm_dict, setup_tm_dict)
    else:
        return get_fst_alloc(machs, fr, avg_tm_dict, setup_tm_dict)


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

    def __init__(self, fr, modu, avail_confs,
                 avg_tm_dict, setup_tm_dict,
                 pop_size, gene_length, max_iter):
        self.fr = fr
        self.modu = modu
        self.avail_confs = avail_confs
        self.avg_tm_dict = avg_tm_dict
        self.setup_tm_dict = setup_tm_dict
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.max_iter = max_iter
        self.memo = {}
        self.population = []

    def init_pop(self):
        for _ in range(self.pop_size):
            machs = [random.choice(self.avail_confs) for _ in range(self.gene_length)]
            self.population.append(machs)
            self.maintain_memo(machs)

    def selection(self):
        pop = sorted(self.population, key=lambda ind: self.memo[self.chg(ind)].score)
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
            mut_chance = True if random.random() < float(1) / len(children[i])else False
            if mut_chance:
                pos = random.randint(0, self.gene_length - 1)
                children[i][pos] = random.choice(self.avail_confs)
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

    def print_best(self):
        pop = sorted(self.population, key=lambda ind: self.memo[self.chg(ind)].score)
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
        pop = sorted(self.population, key=lambda ind: self.memo[self.chg(ind)].score)
        ind = self.memo[self.chg(pop[0])]
        num_arr = list(self.chg(ind.machs))
        confs = list(set(ind.machs))
        conf_num_map = {}
        for cf in confs:
            conf_num_map[cf] = num_arr[conf_idx_map[cf]]
        ext_dat_df.loc[len(ext_dat_df.index)] = [proj_name, cate, len(confs), conf_num_map,
                                                 ind.time_seq, ind.time_parallel,
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
                machs_fr_lim(machs, self.fr, self.modu, self.avg_tm_dict, self.setup_tm_dict)
            new_ind = Individual(copy.deepcopy(machs), time_seq, time_parallel, price, min_fr, max_fr,
                                 mach_test_dict, unmatched_fr_test)
            if self.modu == 'cheap':
                new_ind.score = price
            else:
                new_ind.score = time_parallel
            self.memo[conf_tup] = new_ind

    def chg(self, machs: list):
        conf_list = [0 for _ in range(confs_num)]
        for conf in machs:
            conf_list[conf_idx_map[conf]] += 1
        return tuple(conf_list)


if __name__ == '__main__':
    num_of_machines = [1, 2, 4, 6, 8, 10, 12]
    pct_of_failure_rate = [0, 0.2, 0.4, 0.6, 0.8, 1]
    chp_or_fst = ['cheap', 'fast']
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
    for i in range(confs_num):
        conf_idx_map[avail_confs[i]] = i
    for proj_name in proj_names:
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
        avg_tm_dict = preproc(proj_name)
        setup_tm_dict = read_setup_time_map(proj_name)
        for mach_num in num_of_machines:
            for pct in pct_of_failure_rate:
                for modu in chp_or_fst:
                    ga = GA(fr=pct, modu=modu,
                            avail_confs=avail_confs,
                            avg_tm_dict=avg_tm_dict,
                            setup_tm_dict=setup_tm_dict,
                            pop_size=100, gene_length=mach_num, max_iter=100)
                    ga.init_pop()
                    ga.run()
                    cate = str(mach_num) + '-' + str(pct) + '-' + modu
                    print('------------------------------------   ' + cate + '   ------------------------------------')
                    ga.print_best()
                    ga.rec_best(proj_name, cate)
        csv_name = 'ext_dat/' + proj_name + '.csv'
        ext_dat_df.to_csv(csv_name, sep=',', index=False, header=True)
