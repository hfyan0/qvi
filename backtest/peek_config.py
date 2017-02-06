#!/usr/bin/env python
import sys

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [map(lambda x: x.strip(), line.split(',')) for line in f]

def justify_str(s,totlen,left_right="right",padchar=' '):
    def extra(s,totlen):
        return ''.join(map(lambda x: padchar, range(totlen - len(s))))
    s = str(s)
    if left_right == "left":
        return s + extra(s,totlen)
    elif left_right == "right":
        return extra(s,totlen) + s
    else:
        return s

backtest_data=read_file(sys.argv[1])
backtest_data=filter(lambda x: "config" in x[0], backtest_data)

print '\n'.join(map(lambda x: x[:200], filter(lambda x: ":" in x, backtest_data[0])))
