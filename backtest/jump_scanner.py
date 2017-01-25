#!/usr/bin/env python
import sys
import os

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [line.strip().split(',') for line in f]

data_list = read_file(sys.argv[1])
print '\n'.join(map(lambda x: str(x[0]+x[1]), filter(lambda x: ((x[0][1] == x[1][1]) and (abs(float(x[0][2])) > 0.001) and abs(float(x[1][2])/float(x[0][2])-1) > 0.2), zip(data_list[:-1],data_list[1:]))))
