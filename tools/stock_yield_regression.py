#!/usr/bin/env python
import numpy as np
import statsmodels.api as sm
import sys
import os
import math
import datetime
sys.path.append(os.path.dirname(sys.path[0]))
from mvo import get_hist_data_key_date,get_hist_data_key_sym

def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results


MIN_DAY = 252*5 # trading day
MAX_DAY = 365*10 # calendar day
RTN_DAY = 365*1 # calendar day

data_path="/home/qy/Dropbox/nirvana/mvo/data/"

file_dps      = data_path+"hist_trail12m_dps.csv"
file_eps      = data_path+"hist_trail12m_eps.csv"
file_px       = data_path+"hist_last_px.csv"
file_best_eps = data_path+"hist_best_eps.csv"

dps_key_sym_dict       = get_hist_data_key_sym(file_dps)
eps_key_sym_dict       = get_hist_data_key_sym(file_eps)
px_key_sym_dict        = get_hist_data_key_sym(file_px)
best_eps_key_sym_dict  = get_hist_data_key_sym(file_best_eps)

symbol = sys.argv[1]
date_set_list = []
date_set_list.append(set(map(lambda x: x[0], dps_key_sym_dict[symbol]      )))
date_set_list.append(set(map(lambda x: x[0], eps_key_sym_dict[symbol]      )))
date_set_list.append(set(map(lambda x: x[0], best_eps_key_sym_dict[symbol] )))
date_set_list.append(set(map(lambda x: x[0], px_key_sym_dict[symbol]       )))

date_set  = reduce(lambda a, x: x.intersection(a), date_set_list)
date_list = sorted(list(date_set))

for dt in date_list[:-RTN_DAY]:
    px_ts       = filter(lambda x: x[0] >= (dt - datetime.timedelta(days=MAX_DAY)), filter(lambda x: x[0] <= dt, filter(lambda x: x[0] in date_set, px_key_sym_dict[symbol])))
    dps_ts      = filter(lambda x: x[0] >= (dt - datetime.timedelta(days=MAX_DAY)), filter(lambda x: x[0] <= dt, filter(lambda x: x[0] in date_set, dps_key_sym_dict[symbol])))
    eps_ts      = filter(lambda x: x[0] >= (dt - datetime.timedelta(days=MAX_DAY)), filter(lambda x: x[0] <= dt, filter(lambda x: x[0] in date_set, eps_key_sym_dict[symbol])))
    best_eps_ts = filter(lambda x: x[0] >= (dt - datetime.timedelta(days=MAX_DAY)), filter(lambda x: x[0] <= dt, filter(lambda x: x[0] in date_set, best_eps_key_sym_dict[symbol])))

    if any([len(px_ts) < MIN_DAY,len(dps_ts) < MIN_DAY,len(eps_ts) < MIN_DAY,len(best_eps_ts) < MIN_DAY]):
        continue

    fut_px_ts = filter(lambda x: x[0] > (px_ts[0][0] + datetime.timedelta(days=RTN_DAY)), px_key_sym_dict[symbol])[:len(px_ts)]
    ln_rtn = map(lambda x: math.log(math.pow(x[0][1]/x[1][1],365.0/RTN_DAY)), zip(fut_px_ts,px_ts))

    y = ln_rtn
    x = []
    # x.append(map(lambda x: math.log(x[0][1]/x[1][1]), zip(dps_ts,px_ts)))
    # x.append(map(lambda x: math.log(x[0][1]/x[1][1]), zip(best_eps_ts,eps_ts)))
    x.append(map(lambda x: 0.29+0.05*(math.log(x[0][1]/x[1][1])+1.4*math.log(x[2][1]/x[3][1])), zip(dps_ts,px_ts,best_eps_ts,eps_ts)))

    # print reg_m(y, x).summary()
    param_list = list(reg_m(y, x).params)

    ###################################################
    # forecast
    ###################################################
    pred_dt = filter(lambda x: (x[0] > fut_px_ts[-1][0]) and (x[0] in date_set), px_key_sym_dict[symbol])[0][0]
    pred_dt_px       = filter(lambda x: x[0] == pred_dt,       px_key_sym_dict[symbol])[0][1]
    pred_dt_dps      = filter(lambda x: x[0] == pred_dt,      dps_key_sym_dict[symbol])[0][1]
    pred_dt_eps      = filter(lambda x: x[0] == pred_dt,      eps_key_sym_dict[symbol])[0][1]
    pred_dt_best_eps = filter(lambda x: x[0] == pred_dt, best_eps_key_sym_dict[symbol])[0][1]
    # pred_rtn = math.exp(param_list[1]*math.log(pred_dt_dps/pred_dt_px) + param_list[0]*math.log(pred_dt_best_eps/pred_dt_eps) + param_list[2]) - 1
    ori_pred_ln_rtn = 0.29+0.05*(math.log(pred_dt_dps/pred_dt_px)+1.4*math.log(pred_dt_best_eps/pred_dt_eps))
    ori_pred_rtn = math.exp(ori_pred_ln_rtn) - 1
    pred_rtn = math.exp(param_list[0]*(ori_pred_ln_rtn) + param_list[1]) - 1

    print ','.join(map(str, [pred_dt,pred_rtn,ori_pred_rtn]+param_list))
    # print dt,math.exp(ln_rtn[-1])-1
    # print dt,px_ts[-1],fut_px_ts[-1],dps_ts[-1],eps_ts[-1],best_eps_ts[-1]
