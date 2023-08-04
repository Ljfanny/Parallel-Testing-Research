import os
import pandas as pd

projName = ''
totalData = pd.DataFrame(None,
                         columns=['classname', 'methodname', 'throttConf', 'outerRound', 'avgTime', 'passRate', 'price'],
                         dtype=str)
configFastestData = pd.DataFrame(None,
                                 columns=['classname', 'methodname', 'throttConf', 'avgTime', 'passRate', 'price'])
configCheapestData = pd.DataFrame(None,
                                  columns=['classname', 'methodname', 'throttConf', 'avgTime', 'passRate', 'price'])
summarizationData = pd.DataFrame(None,
                                 columns=['projectname', 'price', 'throttConfNum', 'confs', 'time', 'cheapest|fastest'])
totalData['outerRound'] = totalData['outerRound'].astype('int')
totalData['avgTime'] = totalData['avgTime'].astype('float')
totalData['passRate'] = totalData['passRate'].astype('float')
totalData['price'] = totalData['price'].astype('float')


def to_mydict(df, keyName, valueName):
    mydict = {}
    for idx, row in df.iterrows():
        mydict[row[keyName]] = row[valueName]
    return mydict


# fastest
def ext_tradeoff_fast(configMap):
    global configFastestData, summarizationData
    configs = []
    prices = 0
    times = 0
    for ky, val in configMap.items():
        # passRate: large -> small
        tmp = sorted(val, key=lambda x: x[2], reverse=True)
        maxPassRate = tmp[0][2]
        tmpLen = len(tmp)
        fast = 100
        idx = -1
        for i in range(tmpLen):
            passRate = tmp[i][2]
            if passRate == maxPassRate:
                if fast > tmp[i][1]:
                    idx = i
                    fast = tmp[i][1]
            else:
                break
        config = tmp[idx]
        throttConf = config[0]
        price = config[3]
        configFastestData.loc[len(configFastestData.index)] = [ky[0], ky[1], throttConf, config[1], config[2], price]
        prices += price
        times += config[1]
        if throttConf not in configs:
            configs.append(throttConf)
    if prices > 0:
        summarizationData.loc[len(summarizationData.index)] = [projName, prices, len(configs), ','.join(configs), times, 'fastest']

# cheapest
def ext_tradeoff_cheap(configMap):
    global configCheapestData, summarizationData
    configs = []
    prices = 0
    times = 0
    for ky, val in configMap.items():
        # passRate: large -> small
        tmp = sorted(val, key=lambda x: x[2], reverse=True)
        maxPassRate = tmp[0][2]
        tmpLen = len(tmp)
        chp = 100
        idx = -1
        for i in range(tmpLen):
            passRate = tmp[i][2]
            if passRate == maxPassRate:
                if chp > tmp[i][3]:
                    idx = i
                    chp = tmp[i][3]
            else:
                break
        config = tmp[idx]
        throttConf = config[0]
        price = config[3]
        configFastestData.loc[len(configFastestData.index)] = [ky[0], ky[1], throttConf, config[1], config[2], price]
        prices += price
        times += config[1]
        if throttConf not in configs:
            configs.append(throttConf)
    if prices > 0:
        summarizationData.loc[len(summarizationData.index)] = [projName, prices, len(configs), ','.join(configs), times, 'cheapest']


def cal_avg_time(file):
    global totalData
    data = pd.read_csv(file)
    # key=(classname,methodname,throttConf); value=[outerRound,passNum,totalTime]
    dict = {}
    configPriceData = pd.read_csv('config_price.csv')
    configPriceMap = to_mydict(configPriceData, 'config', 'price_hour')
    for _, i in data.iterrows():
        throttConf = i['throttConf']
        if throttConf not in configPriceMap.keys():
            continue
        time = i['time']
        ky = (i['classname'], i['methodname'], throttConf)
        if ky in dict.keys():
            # outerRound
            dict[ky][0] += 1
            if i['result'] != 'pass':
                continue
            dict[ky][1] += 1
            dict[ky][2] += time
        else:
            if i['result'] != 'pass':
                dict[ky] = [1, 0, 0]
                continue
            dict[ky] = [1, 1, time]
    # key=(classname,methodname); value=[throttConf,priceHour]
    configMap = {}
    for ky, val in dict.items():
        priceHour = configPriceMap[ky[2]]
        if val[1] == 0:
            continue
        avgTime = val[2]/val[1]
        passRate = val[1]/val[0]
        price = priceHour * avgTime / 3600
        totalData.loc[len(totalData.index)] = [ky[0], ky[1], ky[2], val[0], avgTime, passRate, price]
        configKy = (ky[0], ky[1])
        if configKy in configMap.keys():
            configMap[configKy].append([ky[2], avgTime, passRate, price])
        else:
            configMap[configKy] = [[ky[2], avgTime, passRate, price]]
    ext_tradeoff_fast(configMap)
    ext_tradeoff_cheap(configMap)


if __name__ == "__main__":
    dirpath = 'results/'
    filenames = os.listdir(dirpath)
    for f in filenames:
        projName = f.split('_')[0]
        cal_avg_time(dirpath + f)
    totalData.to_csv('extractAvgTime.csv', sep=',', index=False, header=True)
    configFastestData.to_csv('extractMaxPassingRateFastest.csv', sep=',', index=False, header=True)
    configCheapestData.to_csv('extractMaxPassingRateCheapest.csv', sep=',', index=False, header=True)
    summarizationData.to_csv('summarization.csv', sep=',', index=False, header=True)