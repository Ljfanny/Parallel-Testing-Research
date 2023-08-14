import os
import re
import json
from pprint import pprint

thrott_conf_idx = 0
avg_time_idx = 2
mvn_log_path = '../raft-maven-logs/'
rec_path = 'machine_setup_time/'
preproc_path = 'preprocessed_data_results/'
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
    'zxing_dot']


def analysis_maven_log(proj_name: str):
    conf_tot_tm_dict = {k: float(0) for k in avail_confs}
    proj_fld = mvn_log_path + proj_name
    conf_fld = os.listdir(proj_fld)
    for fld in conf_fld:
        log_fld = proj_fld + '/' + fld
        logs = os.listdir(log_fld)
        tms = 0
        cnt = 0
        for log in logs:
            log_dir = log_fld + '/' + log
            with open(log_dir) as f:
                for line in f:
                    tm = 0
                    rnd = 0
                    m = re.search(r'\[INFO] Total time: (\d*):(\d*) (\w+)', line)
                    if m:
                        rnd += 1
                        if m.group(3) == 'h':
                            tm = float(m.group(1)) * 3600 + float(m.group(2)) * 60
                        else:
                            tm = float(m.group(1)) * 60 + float(m.group(2))
                    else:
                        m = re.search(r'\[INFO] Total time: (\d*.\d*) s', line)
                        if m:
                            rnd += 1
                            tm = float(m.group(1))
                    tms += tm
                    cnt += rnd
        avg_setup_tm = tms / cnt
        conf_tot_tm_dict[fld] = avg_setup_tm
    return conf_tot_tm_dict


def load_tst_tm(proj_name):
    preproc_file = preproc_path + proj_name
    conf_tst_tm_dict = {k: float(0) for k in avail_confs}
    if os.path.exists(preproc_file):
        with open(preproc_file, 'r') as f:
            tmp_dict = json.load(f)
        avg_tm_dict = {tuple(eval(k)): v for k, v in tmp_dict.items()}
    else:
        print('[ERROR] {} does not exist.'.format(preproc_file))
        return conf_tst_tm_dict
    for key, val in avg_tm_dict.items():
        for itm in val:
            conf_tst_tm_dict[itm[thrott_conf_idx]] += itm[avg_time_idx]
    return conf_tst_tm_dict


def write_file(proj_name, conf_tot_tm_dict, conf_tst_tm_dict):
    setup_time_map = {k: t1 - conf_tst_tm_dict[k] for k, t1 in conf_tot_tm_dict.items()}
    file = rec_path + proj_name
    with open(file, 'w') as f:
        json.dump(setup_time_map, f)
    f.close()
    for i in setup_time_map.values():
        if i < 0:
            print(proj_name)
            pprint(setup_time_map)
            break
    # print('Done!')


def read_setup_time_map(proj_name: str):
    file = rec_path + proj_name
    with open(file, 'r') as f:
        setup_time_map = json.load(f)
    return setup_time_map


if __name__ == '__main__':
    for proj in proj_names:
        # print(proj)
        write_file(proj, analysis_maven_log(proj), load_tst_tm(proj))
