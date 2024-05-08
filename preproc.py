import os
import json
import pandas as pd

confs_num = 12
dir_path = '...raft-csvs'
err_dat = 'error_tests.csv'
tst_num_list = [
    2047,
    4459,
    55,
    713,
    14,
    66,
    390,
    560,
    502,
    202,
    18,
    79,
    412,
    6267,
    25,
    163,
    144,
    305,
    83,
    863,
    1689,
    297,
    80,
    308,
    345
]
conf_prc_map = {}


def get_conf_prc_map():
    global conf_prc_map
    conf_prc_df = pd.read_csv('config_price.csv')
    for idx, row in conf_prc_df.iterrows():
        conf_prc_map[row['config']] = row['price_hour']


# rec_dict: key=(classname, methodname, throttConf); value=[outerRound, numOfPassing, timeOfSum]
def ext_dat(csv_dir: str,
            proj_name: str,
            rec_dict: dict):
    dat = pd.read_csv(f'{csv_dir}/{proj_name}.csv', low_memory=False)
    conf_lst = conf_prc_map.keys()
    for _, row in dat.iterrows():
        throttConf = row['throttConf']
        if throttConf not in conf_lst or row['methodname'].__class__ != str:
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


# avg_tm_dict: key=(classname, methodname); value=[throttConf, outerRound, avgTime, failureRate, price]
def cal_dat(proj_name: str,
            rec_dict: dict):
    avg_tm_dict = {}
    idx = 0
    errs = []
    for key, val in rec_dict.items():
        tst = (key[0], key[1])
        if val[1] != 0:
            avg_time = val[2] / val[1]
        else:
            err = [proj_name, f'{key[0]}#{key[1]}', f'{key[2]}\n']
            errs.append(err)
            continue
        failure_rate = 1 - val[1] / val[0]
        prc = conf_prc_map[key[2]] * avg_time / 3600
        if tst not in avg_tm_dict.keys():
            avg_tm_dict[tst] = [[key[2], val[0], avg_time, failure_rate, prc]]
            idx += 1
        else:
            avg_tm_dict[tst].append([key[2], val[0], avg_time, failure_rate, prc])
    for err in errs:
        with open(err_dat, 'a') as f:
            f.write(','.join(err))
    return avg_tm_dict


def preproc(prep_dir,
            proj_name):
    preproc_file_name = f'{prep_dir}/{proj_name}'
    get_conf_prc_map()
    cnt = 1
    if os.path.exists(preproc_file_name):
        print(f'{proj_name} has already been preprocessed!')
        with open(preproc_file_name, 'r') as f:
            tmp_dict = json.load(f)
        avg_tm_dict = {tuple(eval(k)): v for k, v in tmp_dict.items()}
    else:
        rec_dict = {}
        filenames = os.listdir(dir_path)
        for f in filenames:
            ext_dat(csv_dir=f'{dir_path}/{f}', proj_name=proj_name, rec_dict=rec_dict)
            print(f'{proj_name}#{cnt}... ...')
            cnt += 1
        avg_tm_dict = cal_dat(proj_name=proj_name, rec_dict=rec_dict)
        if len(avg_tm_dict) == tst_num_list[proj_idx]:
            tmp_dict = {str(k): v for k, v in avg_tm_dict.items()}
            with open(preproc_file_name, 'w') as f:
                json.dump(tmp_dict, f)
    try:
        for t, info in avg_tm_dict.items():
            for i in range(confs_num):
                fr = info[i][3]
                if fr == 0: continue
                runtime = info[i][2]
                info[i][2] = runtime * sum([fr ** p for p in range(10)])
    except IndexError:
        return None
    return avg_tm_dict
