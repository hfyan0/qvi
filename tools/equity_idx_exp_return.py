#!/usr/bin/env python
import sys
import os
import math
import datetime
sys.path.append(os.path.dirname(sys.path[0]))
from qvi import get_hist_data_key_date,get_hist_data_key_sym

data_path="/home/qy/Dropbox/nirvana/mvo/data/"

file_eps      = data_path+"hist_idx_trail12m_eps.csv"
file_bps      = data_path+"hist_idx_bps.csv"
file_px       = data_path+"hist_idx_last_px.csv"
file_best_eps = data_path+"hist_idx_best_eps.csv"

eps_key_sym_dict       = get_hist_data_key_sym(file_eps)
bps_key_sym_dict       = get_hist_data_key_sym(file_bps)
px_key_sym_dict        = get_hist_data_key_sym(file_px)
best_eps_key_sym_dict  = get_hist_data_key_sym(file_best_eps)

symbol = sys.argv[1]
date_set_list = []
date_set_list.append(set(map(lambda x: x[0], eps_key_sym_dict[symbol]      )))
date_set_list.append(set(map(lambda x: x[0], bps_key_sym_dict[symbol]      )))
date_set_list.append(set(map(lambda x: x[0], px_key_sym_dict[symbol]       )))
date_set_list.append(set(map(lambda x: x[0], best_eps_key_sym_dict[symbol] )))

date_set  = reduce(lambda a, x: x.intersection(a), date_set_list)
date_list = sorted(list(date_set))

for dt in date_list:
    g = (filter(lambda x: x[0] <= dt, best_eps_key_sym_dict[symbol])[-1][1] / filter(lambda x: x[0] <= dt, eps_key_sym_dict[symbol])[-1][1]) - 1.0
    B = filter(lambda x: x[0] <= dt, bps_key_sym_dict[symbol])[-1][1]
    E = filter(lambda x: x[0] <= dt, eps_key_sym_dict[symbol])[-1][1]
    P = filter(lambda x: x[0] <= dt, px_key_sym_dict[symbol])[-1][1]

    sqrt_det = math.sqrt(math.pow(P - g * B - (1+g) * E, 2.0) + 4 * (1+g) * P * E)
    n_b = g * B + (1-g) * E - P
    exp_rtn_list = filter(lambda x: x > 0.0, map(lambda x: x / 2.0 / P, [n_b + sqrt_det, n_b - sqrt_det]))
    if len(exp_rtn_list) > 0:
        exp_rtn = min(exp_rtn_list)
        print ','.join(map(str, [dt,exp_rtn,g,B,E,P]))
