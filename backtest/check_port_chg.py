#!/usr/bin/env python
import sys

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [map(lambda x: x.strip(), line.split(',')) for line in f]

backtest_data=read_file(sys.argv[1])
backtest_data=filter(lambda x: "config" not in x[0], backtest_data)

dt_sym_pos_dict = dict(map(lambda x: (x[0],dict(map(lambda s: (s.split(':')[0],float(s.split(':')[2])), filter(lambda x: "pos" not in x, x[7:])))), backtest_data))

last = None
for dt,sym_pos_dict in sorted(dt_sym_pos_dict.items(), key=lambda x: x[0]):
    if last is None:
        last = sym_pos_dict
        continue

    all_sym = set(sym_pos_dict.keys()).union(set(last.keys()))
    tot_chg = 0.0
    for s in all_sym:
        last_pos = last.get(s,0.0)
        cur_pos = sym_pos_dict.get(s,0.0)
        if cur_pos != last_pos:
            tot_chg += abs(cur_pos-last_pos)
            print "%s: %s: %s (%s -> %s)" % (dt,s,cur_pos-last_pos,last_pos,cur_pos)
   
    print "%s: total change: %s %%" % (dt,tot_chg*100)
    last = sym_pos_dict
    print
