projs=(
    activiti_dot
    ambari_dot
    assertj-core_dot
    carbon-apimgt_analyzer-modules.org.wso2.carbon.apimgt.throttling.siddhi.extension
    commons-exec_dot
    db-scheduler_dot
    delight-nashorn-sandbox_dot
    elastic-job-lite_dot
    elastic-job-lite_elastic-job-lite-core
    esper_examples.rfidassetzone
    excelastic_dot
    fastjson_dot
    fluent-logger-java_dot
    handlebars.java_dot
    hbase_dot
    hector_dot
    http-request_dot
    httpcore_dot
    hutool_hutool-cron
    incubator-dubbo_dot
    incubator-dubbo_dubbo-remoting.dubbo-remoting-netty
    incubator-dubbo_dubbo-rpc.dubbo-rpc-dubbo
    java-websocket_dot
    logback_dot
    luwak_luwak
    ninja_dot
    noxy_noxy-discovery-zookeeper
    okhttp_dot
    orbit_dot
    orbit_actors.runtime
    oryx_framework.oryx-common
    retrofit_retrofit-adapters.rxjava
    retrofit_retrofit
    riptide_riptide-failsafe
    rxjava2-extras_dot
    spring-boot_dot
    timely_server
    undertow_dot
    wildfly_dot
    wro4j_dot
    wro4j_wro4j-extensions
    yawp_yawp-testing.yawp-testing-appengine
    zxing_dot
    )

export PARENT="2022-04-19-fix-10x50-30"
export SCRIPT=/workspace/data/raft/dockerThrottling/mvnTestResultParsing/extractRaftJunitResults_experiment.py

for i in "${projs[@]}"
do
    python3 $SCRIPT $PARENT/$i/
    mv $PARENT/$i/results.csv csvs-30/$i.csv
done