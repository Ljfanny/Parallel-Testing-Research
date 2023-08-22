import os
import re
import json
from analy import avail_confs, proj_names
from pprint import pprint

thrott_conf_idx = 0
avg_time_idx = 2
mvn_log_path = '../raft-maven-logs/'
preproc_path = 'preproc_csvs/'
ana_mvn_log_path = 'preproc_mvn_logs/'


def analysis_maven_log(proj_name: str):
    file = ana_mvn_log_path + proj_name
    if os.path.exists(file):
        with open(file, 'r') as f:
            conf_tot_tm_dict = json.load(f)
        return conf_tot_tm_dict
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
    with open(file, 'w') as f:
        json.dump(conf_tot_tm_dict, f)
    return conf_tot_tm_dict


def load_tst_tm(proj_name):
    preproc_file = preproc_path + proj_name
    conf_tst_tm_dict = {k: float(0) for k in avail_confs}
    if os.path.exists(preproc_file):
        with open(preproc_file, 'r') as f:
            tmp_dict = json.load(f)
        avg_tm_dict = {tuple(eval(k)): v for k, v in tmp_dict.items()}
    else:
        print('[ERROR] {} does not exist'.format(preproc_file))
        return conf_tst_tm_dict
    for key, val in avg_tm_dict.items():
        for itm in val:
            conf_tst_tm_dict[itm[thrott_conf_idx]] += itm[avg_time_idx]
    return conf_tst_tm_dict


def write_file(proj_name, conf_tot_tm_dict, conf_tst_tm_dict):
    setup_time_map = {k: t1 - conf_tst_tm_dict[k] if t1 >= conf_tst_tm_dict[k] else float(0)
                      for k, t1 in conf_tot_tm_dict.items()}
    pprint(setup_time_map)
    file = rec_path + proj_name
    with open(file, 'w') as f:
        json.dump(setup_time_map, f)
    f.close()


if __name__ == '__main__':
    for proj in proj_names:
        write_file(proj, analysis_maven_log(proj), load_tst_tm(proj))
