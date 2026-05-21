import numpy as np

def filter_opt(opt, tag):
    ret = { }

    for k,v in opt.items():
        tokens = k.split('.')
        if tokens[0] == tag:
            ret['.'.join(tokens[1:])] = v

    return ret


class AverageValueMeter(object):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.var += value**2 * n
        self.n += n

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0

    def value(self):
        if self.n == 0:
            return 0.0, 0.0
        mean = self.sum / self.n
        std = np.sqrt(max(self.var / self.n - mean**2, 0))
        return mean, std



