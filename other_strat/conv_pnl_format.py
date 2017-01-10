#!/usr/bin/env python
###################################################
# convert from const cap cpnl to cumulative return
###################################################
import sys

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [line for line in f]

csv_list = read_file(sys.argv[1])

start = float(sys.argv[2])
last = 0.0
current = start
for csv in csv_list:
    fields = csv.split(',')
    current = current * (1 + (float(fields[1]) - last) / start)
    last = float(fields[1])
    print "%s,,,,,%s" % (fields[0], current)
