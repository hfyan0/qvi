#!/usr/bin/env python
import sys
import math
from configobj import ConfigObj
from datetime import datetime, timedelta

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [map(lambda x: x.strip(), line.split(',',1)) for line in f]

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

def subtract_1sd_from_mean(v_list):
    m = sum(map(float, v_list)) / len(v_list)
    v = sum(map(lambda x: math.pow(x-m,2), v_list)) / (len(v_list)-1)
    s = math.sqrt(abs(v))
    return m-s

def avg_asset(a_list):
    def avg_asset_impl(remaining_list,out_list,last_a):
        if len(remaining_list) == 0:
            return out_list
        elif last_a == 0:
            return avg_asset_impl(remaining_list[1:],out_list,remaining_list[0])
        else:
            return avg_asset_impl(remaining_list[1:],out_list+[(remaining_list[0]+last_a)/2.0],remaining_list[0])
    return avg_asset_impl(a_list,[],0)

def get_data_as_dict(f):
    return dict(map(lambda x: (x[0],map(float,filter(lambda y: ('-' not in y) and ('%' not in y), map(lambda z: z.strip().replace(',',''), x[1].split())))), read_file(f)))

def correct_scale(f):
    if abs(f) < 0.00001:
        return 0.0
    elif f >= 1.0:
       return correct_scale(f/1000.0)
    elif f < 0.0001:
       return correct_scale(f*1000.0)
    else:
        return f

def correct_scale2(f):
    return f/100.0

def get_eps_from_roa(sym,sym_roa_dict,historical_noncurasset_src_dict,historical_curasset_src_dict,no_of_issued_shares):
    return map(lambda roa: correct_scale2(roa * (historical_noncurasset_src_dict[sym][0]+historical_curasset_src_dict[sym][0]) / no_of_issued_shares), sym_roa_dict[sym])

config = ConfigObj('config.ini')
historical_earnings_src_dict = get_data_as_dict(config["general"]["historical_earnings_src"])
historical_eps_src_dict = get_data_as_dict(config["general"]["historical_eps_src"])
historical_noncurasset_src_dict = get_data_as_dict(config["general"]["historical_noncurasset_src"])
historical_curasset_src_dict = get_data_as_dict(config["general"]["historical_curasset_src"])

symbol_list = sorted(list(set(historical_earnings_src_dict.keys()).intersection(set(historical_noncurasset_src_dict.keys()))))

sym_roa_list = map(lambda tup_list: map(lambda tup: correct_scale(tup[0]/(tup[1]+tup[2])), tup_list), map(lambda s: zip(historical_earnings_src_dict[s][0:-1],avg_asset(historical_noncurasset_src_dict[s]),avg_asset(historical_curasset_src_dict[s])), symbol_list))
print "ROA"
print '\n'.join(map(lambda x: justify_str(x[0],8)+": "+','.join(map(lambda a: justify_str(a,6), map(lambda y: round(y*100,2), x[1])))+" (%)", zip(symbol_list,sym_roa_list)))

sym_roa_list = zip(symbol_list,map(lambda x: sorted([x[0],subtract_1sd_from_mean(x),min(x)]), sym_roa_list))
sym_roa_dict = dict(sym_roa_list)

print "ROA"
print '\n'.join(map(lambda tup: justify_str(tup[0],8)+": "+','.join(map(lambda y: justify_str(y,6), map(lambda x: str(round(x*100,2)), tup[1])))+" (%)", sym_roa_list))

sym_est_eps_list = map(lambda s: (s,get_eps_from_roa(s,sym_roa_dict,historical_noncurasset_src_dict,historical_curasset_src_dict,float(historical_earnings_src_dict[s][0])/float(historical_eps_src_dict[s][0]))), symbol_list)
print "EPS"
print '\n'.join(map(lambda tup: justify_str(tup[0],8)+": "+','.join(map(lambda y: justify_str(y,6), map(lambda x: str(round(x,2)), tup[1]))), sym_est_eps_list))
