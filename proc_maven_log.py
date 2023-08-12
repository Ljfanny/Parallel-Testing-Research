import os
import json
import pandas as pd

rec_path = 'machine_setup_time/'
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


def analysis_maven_log():
    return


def write_file():
    return


def read_setup_time_map(proj_name: str):
    file_path = rec_path + proj_name
    with open(file_path, 'r') as f:
        setup_time_map = json.load(f)
    return setup_time_map


if __name__ == '__main__':
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
        '01noThrottling.sh'
    ]
