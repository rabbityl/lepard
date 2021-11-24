from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import numpy as np
from collections import defaultdict

class Timer(object):

    def __init__(self):
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def tictoc(self, diff):
        self.diff = diff
        self.total_time += diff
        self.calls += 1

    def total(self):
        """ return the total amount of time """
        return self.total_time

    def avg(self):
        """ return the average amount of time """
        return self.total_time / float(self.calls)

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

class Timers(object):

    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def tictoc(self, key, diff):
        self.timers[key].tictoc( diff)

    def print(self, key=None):
        if key is None:
            # print all time
            for k, v in self.timers.items():
                print("{:}: \t  average {:.4f},  total {:.4f} ,\t calls {:}".format(k.ljust(30),  v.avg(), v.total_time, v.calls))
        else:
            print("Average time for {:}: {:}".format(key, self.timers[key].avg()))

    def get_avg(self, key):
        return self.timers[key].avg()