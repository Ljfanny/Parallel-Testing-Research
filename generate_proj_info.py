import pandas as pd
import numpy as np
from analyze import proj_names
from preproc import preproc


if __name__ == '__main__':
    info_df = pd.DataFrame(None,
                           columns=['id',
                                    'project-module',
                                    'test_num',
                                    'min_max_failure_rate',
                                    'max_failure_rate',
                                    'avg_failure_rate',
                                    'avg_test_runtime'])
    for i, proj_name in enumerate(proj_names):
        preproc_proj_dict = preproc(proj_name)
        test_num = len(preproc_proj_dict.keys())
        min_fr = 100
        max_fr = 0
        sum_fr = 0
        sum_runtime = 0
        cand_num = 0
        for _, val in preproc_proj_dict.items():
            fr = max(np.array(val)[:, 3].astype('float'))
            sum_fr += sum(np.array(val)[:, 3].astype('float'))
            sum_runtime += sum(np.array(val)[:, 2].astype('float'))
            cand_num += len(val)
            if fr > max_fr: max_fr = fr
            if fr < min_fr: min_fr = fr
        info_df.loc[len(info_df.index)] = [
            f'P{i}',
            proj_name,
            test_num,
            min_fr,
            max_fr,
            sum_fr / cand_num,
            sum_runtime / cand_num
        ]
    info_df.to_csv('proj_info.csv', sep=',', header=True, index=False,
                   float_format='%.2f ')
