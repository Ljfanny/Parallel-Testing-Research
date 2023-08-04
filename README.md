# Parallel-Testing-Research
The analysis includes generated data on Raft's performance, scalability and fault tolerance characteristics under different system configurations and workloads.

## DIR

###  ext_dat

> not including data about "fastest parallel execution"

The csv files in this folder contain the following information:

- project: including project name and module name

- machine_list_or_failure_rate_or_cheap_or_fast_category: 

  - Machine number: [2, 4, 6, 8, 10, 12]
  - Failure rate limit: [0, 0.2, 0.4, 0.6, 0.8, 1]
  - which one takes precedence: [cheapest, fastest sequential execution, fastest parallel execution]

  Each combination is called a category.

- num_confs: how many configurations are included?

- confs: specific configurations

- time_seq: time of sequential execution

- time_parallel: time of parallel execution

- price: total cost (only the test execution time is included without considering the startup time currently)

- min_failure_rate: minimum failure rate over all tests

- max_failure_rate: maximum failure rate over all tests

### failure_rate_unmatched_tests

This folder contains **dicts** recording the projects run under each category of conditions. Since some tests could not meet the failure rate constraint, they have been logged here.

The format: {test info: failure rate}

### index_test_map

Files recorded during the data analysis process, mapping increasing integers to tests for use in the z3-solver's Optimizer to find solutions.

### np_solver_arr

Binary files record the execution time of the tests under each configuration.

### preprocessed_data_results

Files record the results of the initial processing on the raw data.

The format: data type->dict

- key=(class name, method name)
- value=[configuration, number of executions, average time, failure rate, price]

### test_distribution_cond

Files record the allocation of tests (a test runs on a machine with what configuration) under various categories (relatively optimal).



## Code

### analy.py

This file is used for data analysis, in which:

- Given a known list of machines and a given error rate:

  - For **cheapest** or **fastest sequential execution**: This is equivalent to, given the conditions are met, each test choosing to run on the cheapest or fastest machine available.
  - For **fastest parallel execution**: This can be formulated as an integer programming problem, where an optimizer (use **z3-solver's Optimizer** currently) is used to find the optimal solution.

- Given the number of machines:

  Use a **genetic algorithm** to find the optimal configuration.

### solver_time.py

A class is used to implement the solver's solution process.
