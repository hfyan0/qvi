#!/usr/bin/env python
from configobj import ConfigObj
import sys
import math
from datetime import datetime, timedelta, date
import time
import numpy as np
from itertools import groupby

import os
sys.path.append(os.path.dirname(sys.path[0]))
from mvo import calc_cov_matrix_annualized,conv_to_hkd,intWithCommas,justify_str,markowitz,read_file

###################################################
def get_risk_aversion_factor(dt,hsi_expected_return_list):
    # exp_rtn = filter(lambda x: x[0]<=dt, hsi_expected_return_list)[-1][1]
    # risk_aversion_factor = max(min(10.0-exp_rtn,10.0),0.0)
    # print "risk_aversion_factor: %s %s" % (dt,risk_aversion_factor)
    # return risk_aversion_factor
    return 1.0

###################################################
config = ConfigObj('config.ini')
traded_symbol_set = set(config["general"]["traded_symbols"].split(','))
min_no_of_avb_sym = int(config["general"]["min_no_of_avb_sym"])
rebalance_interval = int(config["general"]["rebalance_interval"])
granularity = int(config["general"]["granularity"])
max_weight_dict = config["max_weight"]

her = config["general"]["hsi_expected_return"]
hsi_expected_return_list = map(lambda y: (datetime.strptime(y[1],"%Y-%m-%d").date(),float(y[2])), filter(lambda x: x[0]%2==0, zip(range(len(her)-1),her[:-1],her[1:])))

###################################################
hist_adj_px_list = sorted(map(lambda d: (datetime.strptime(d[0],"%Y-%m-%d").date(),d[1],float(d[2])), read_file(config["general"]["hist_adj_px"])), key=lambda x: x[0])
hist_adj_px_dict = {}
for d,it_lstup in groupby(hist_adj_px_list, lambda x: x[0]):
    hist_adj_px_dict[d] = dict(map(lambda x: (x[1],x[2]), list(it_lstup)))

hist_pb_ratio_dict = {}
for d,it_lstup in groupby(sorted(read_file(config["general"]["hist_pb_ratio"]), key=lambda x: x[0]), lambda x: x[0]):
    hist_pb_ratio_dict[datetime.strptime(d,"%Y-%m-%d").date()] = dict(map(lambda x: (x[1],float(x[2])), list(it_lstup)))

start_date = datetime.strptime(config["general"]["start_date"],"%Y-%m-%d").date()
date_list = sorted(filter(lambda x: x >= start_date, list(set(hist_adj_px_dict.keys()).intersection(set(hist_pb_ratio_dict.keys())))))

rebalance_date_list = map(lambda y: y[1], filter(lambda x: x[0]%rebalance_interval==0, [(i,d) for i,d in enumerate(date_list)]))

###################################################
pos_dict = {}
cash = float(config["general"]["init_capital"])

###################################################
for dt in rebalance_date_list:
    sym_pb_list = filter(lambda x: x[0] in traded_symbol_set, sorted(list(hist_pb_ratio_dict[dt].items()), key=lambda x: x[0]))
    if len(sym_pb_list) < min_no_of_avb_sym:
        continue
    # print dt
    symbol_list = map(lambda x: x[0], sym_pb_list)
    # print symbol_list
    expected_rtn_list = map(lambda x: x[1], sym_pb_list)
    max_weight_list = map(lambda x: float(max_weight_dict[x]), symbol_list)
    from_tgt_rtn = min(map(lambda x: x[1], sym_pb_list))
    to_tgt_rtn = max(map(lambda x: x[1], sym_pb_list))

    ###################################################
    specific_riskiness_list = map(lambda s: 0.0, symbol_list)
    hist_adj_px_list_fil = sorted(filter(lambda x: x[0] <= dt, hist_adj_px_list), key=lambda y: y[0])
    sym_time_series_list = map(lambda s: map(lambda ts: (ts[0],ts[2]), filter(lambda x: x[1] == s, hist_adj_px_list_fil)), symbol_list)
    cov_matrix,annualized_sd_list,annualized_adj_sd_list = calc_cov_matrix_annualized(sym_time_series_list, specific_riskiness_list)
    ###################################################

    optimal_soln_list = []
    for i in range(granularity):
        mu_p = from_tgt_rtn + (to_tgt_rtn - from_tgt_rtn) * float(i)/float(granularity)
        sol_list = markowitz(symbol_list, expected_rtn_list, cov_matrix, mu_p, max_weight_list)

        if sol_list is None:
            continue

        sol_list = list(sol_list["result"]['x'])

        sol_vec = np.asarray(sol_list)
        sol_vec_T = np.matrix(sol_vec).T

        market_port_exp_rtn = float(np.asarray(expected_rtn_list) * sol_vec_T)
        market_port_stdev = math.sqrt(float((sol_vec * cov_matrix) * sol_vec_T))
        market_port_sharpe_ratio = float(market_port_exp_rtn / market_port_stdev)
        market_port_kelly_f_true = float(market_port_exp_rtn / market_port_stdev / market_port_stdev)
        market_port_kelly_f_for_ranking = min(market_port_kelly_f_true, get_risk_aversion_factor(dt,hsi_expected_return_list))
        market_port_kelly_f = min(market_port_kelly_f_true, float(config["general"]["max_allowed_leverage"]))

        target_port_exp_rtn_aft_costs_for_ranking = (market_port_exp_rtn * market_port_kelly_f_for_ranking) - (max(market_port_kelly_f_for_ranking-1.0,0.0)*float(config["general"]["financing_cost"]))
        sol_list = map(lambda x: x * market_port_kelly_f, sol_list)

        if (len(optimal_soln_list) == 0) or (target_port_exp_rtn_aft_costs_for_ranking > optimal_soln_list[0]):
            optimal_soln_list = []
            optimal_soln_list.append(target_port_exp_rtn_aft_costs_for_ranking)
            optimal_soln_list.append(sol_list)

    if len(optimal_soln_list) > 0:
        sym_weight_dict = dict(zip(symbol_list,map(lambda w: str(round(w,5)), optimal_soln_list[1])))
        sym_px_weight_list = map(lambda s: (s,str(hist_adj_px_dict[dt].get(s,0.0)),str(sym_weight_dict.get(s,0.0))), sorted(list(traded_symbol_set)))

        ###################################################
        # sell all pos
        ###################################################
        cash += float(sum([hist_adj_px_dict[dt].get(s,filter(lambda x: x[1]==s, hist_adj_px_list)[-1])*pos for s,pos in pos_dict.items()]))
        pos_dict = {}
        ###################################################

        print str(dt)+","+str(cash)+","+str(len(sym_weight_dict))+","+','.join(map(lambda x: ':'.join(x), sym_px_weight_list))

        ###################################################
        # buy back
        ###################################################
        pos_dict = dict([(s,cash*float(w)/hist_adj_px_dict[dt][s]) for s,w in sym_weight_dict.items()])
        # print "mkt val of pos: %s" % sum([hist_adj_px_dict[dt][s]*pos for s,pos in pos_dict.items()])
        cash = 0
        ###################################################
