#!/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys

sns.set()

with open(sys.argv[1], newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    benchmarks = {}
    for row in reader:
        op = row[0]
        if op not in benchmarks:
            benchmarks[op] = ([], [])
        benchmarks[op][0].append(float(row[1]))
        if '/' in header[2]:
            benchmarks[op][1].append(float(row[2]) / float(row[1]))
        else:
            benchmarks[op][1].append(float(row[2]))

    for op, xy in benchmarks.items():
        plt.plot(xy[0], xy[1], '-o', label=op)
    plt.xscale('log', base=2)
    plt.legend()
    plt.ylim([0, None])
    plt.xlabel(header[1])
    plt.ylabel(header[2])
    plt.show()
