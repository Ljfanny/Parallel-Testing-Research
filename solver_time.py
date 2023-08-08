from z3 import *
import numpy as np


class OptTimeSolver:
    def __init__(self, tst_times: np.array, mach_num: int, tst_num: int):
        self.tst_times = tst_times
        self.mach_num = mach_num
        self.tst_num = tst_num
        self.timeout = 30000

    def figure_out(self):
        tsts = [[Bool('tst_%s_%s' % (i, j)) for j in range(self.mach_num)] for i in range(self.tst_num)]
        mach_tms = [Real('t_%s' % i) for i in range(self.mach_num)]
        maxi = Real('opt')

        opt = Optimize()
        cons = [maxi >= mach_tms[i] for i in range(self.mach_num)]
        for i in range(self.tst_num):
            cons.append(PbEq([(tst, 1) for tst in tsts[i]], 1))
        for j in range(self.mach_num):
            cons.append(mach_tms[j] == Sum([self.tst_times[i][j] * tsts[i][j] for i in range(self.tst_num)]))
        opt.add(cons)
        opt.minimize(maxi)

        res_tsts = [[False for _ in range(self.mach_num)] for _ in range(self.tst_num)]
        res_mach_tms = [0 for _ in range(self.mach_num)]
        opt.set(timeout=self.timeout)
        opt.check()
        # if opt.check() == sat:
        #     print('最优解！')
        # else:
        #     print('求解超时...')
        model = opt.model()
        for j in range(self.mach_num):
            res_mach_tms[j] = model[mach_tms[j]]
        for i in range(self.tst_num):
            for j in range(self.mach_num):
                res_tsts[i][j] = model[tsts[i][j]]
        return res_tsts, res_mach_tms
