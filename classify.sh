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
commits=(
    activiti_activiti-b11f757a_dot*
    apache_ambari-675d6aeb_dot*
    joel-costigliola_assertj-core-22be3825_dot*
    wso2.carbon-apimgt-a82213e_analyzer-modules.org.wso2.carbon.apimgt.throttling.siddhi.extension*
    apache_commons-exec-ea4d1d42_dot*
    kagkarlsson.db-scheduler-4a8a28e_dot*
    javadelight.delight-nashorn-sandbox-da35edc_dot*
    elasticjob_elastic-job-lite-b022898e_dot*
    elasticjob.elastic-job-lite-b022898_elastic-job-lite-core*
    espertechinc.esper-590fa9c_examples.rfidassetzone*
    codingchili.excelastic-6bb7884_dot*
    alibaba.fastjson-e05e9c5_dot*
    fluent.fluent-logger-java-da14ec3_dot*
    jknack_handlebars.java-db77ef2b_dot*
    apache_hbase-e593f0ef_dot*
    hector-client_hector-29e88d0b_dot*
    kevinsawicki_http-request-2d62a3e9_dot*
    apache_httpcore-49247d20_dot*
    looly.hutool-91565d0_hutool-cron*
    apache_incubator-dubbo-737f7a7e_dot*
    apache.incubator-dubbo-737f7a7_dubbo-remoting.dubbo-remoting-netty*
    apache.incubator-dubbo-737f7a7_dubbo-rpc.dubbo-rpc-dubbo*
    tootallnate*
    qos-ch_logback-0f575319_dot*
    flaxsearch.luwak-c27ec08_luwak*
    ninjaframework_ninja-b1b58e87_dot*
    spinn3r.noxy-d53a494_noxy-discovery-zookeeper*
    square_okhttp-129c937f_dot*
    orbit_orbit-7bcd0940_dot*
    orbit.orbit-c4904af_actors.runtime*
    oryxproject.oryx-72ae4bb_framework.oryx-common*
    square.retrofit-ae28c3d_retrofit-adapters.rxjava*
    square.retrofit-ae28c3d_retrofit*
    zalando.riptide-8277e11_riptide-failsafe*
    davidmoten.rxjava2-extras-d0315b6_dot*
    spring-projects_spring-boot-cf24af0b_dot*
    nationalsecurityagency.timely-3a8cbd3_server*
    undertow-io_undertow-ac7204ab_dot*
    wildfly_wildfly-b19048b7_dot*
    wro4j_wro4j-7e3801e1_dot*
    wro4j.wro4j-185ab60_wro4j-extensions*
    feroult.yawp-b3bcf9c_yawp-testing.yawp-testing-appengine*
    zxing_zxing-59ea393a_dot*
    )

export GROUP="2022-04-19-fix-10x50-30"
cd $GROUP

for i in ${!projs[@]};
do
    export projName=${projs[i]}
    mkdir $projName
    mv ${commits[i]} $projName
    cd $projName
    for j in *.tgz;
    do
        tar -zxvf $j
    done
    rm *.tgz
    cd ..
done