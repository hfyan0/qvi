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

dt_sym_pos_dict = dict(map(lambda x: (x[0],dict(map(lambda s: (s.split(':')[0],float(s.split(':')[2])), filter(lambda x: "pos" not in x, x[7:])))), backtest_data))

for dt,sym_pos_dict in sorted(dt_sym_pos_dict.items()):
    print
    print dt
    print '\n'.join(map(lambda x: justify_str(round(x[1],3),8)+": "+x[0], sorted(filter(lambda x: x[1] > 0.001, sym_pos_dict.items()), key=lambda y: y[1], reverse=True)))
