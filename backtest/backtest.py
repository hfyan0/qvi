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
from mvo import calc_cov_matrix_annualized,conv_to_hkd,intWithCommas,justify_str,markowitz,markowitz_robust,log_optimal_growth,read_file,extract_sd_from_cov_matrix

###################################################
config = ConfigObj('config.ini')
traded_symbol_set = set(config["general"]["traded_symbols"])
traded_symbol_list = sorted(config["general"]["traded_symbols"])
hedging_symbol_list = config["general"]["hedging_symbols"]
min_no_of_avb_sym = int(config["general"]["min_no_of_avb_sym"])
rebalance_interval = int(config["general"]["rebalance_interval"])
N = int(config["general"]["granularity"])
max_weight_dict = config["max_weight"]

her = config["general"]["hsi_expected_return"]
hsi_expected_return_list = sorted(map(lambda y: (datetime.strptime(y[1],"%Y-%m-%d").date(),float(y[2])), filter(lambda x: x[0]%2==0, zip(range(len(her)-1),her[:-1],her[1:]))), key=lambda x: x[0])
hsi_hhi_constituents_list = map(lambda x: (x[0],datetime.strptime(x[1],"%Y-%m-%d").date(),datetime.strptime(x[2],"%Y-%m-%d").date()), read_file(config["general"]["hsi_hhi_constituents"]))

###################################################
hist_adj_px_list = sorted(map(lambda d: (datetime.strptime(d[0],"%Y-%m-%d").date(),d[1],float(d[2])), read_file(config["general"]["hist_adj_px"])), key=lambda x: x[0])
hist_adj_px_dict = {}
for d,it_lstup in groupby(hist_adj_px_list, lambda x: x[0]):
    hist_adj_px_dict[d] = dict(map(lambda x: (x[1],x[2]), list(it_lstup)))

hist_bp_ratio_dict = {}
for d,it_lstup in groupby(sorted(read_file(config["general"]["hist_pb_ratio"]), key=lambda x: x[0]), lambda x: x[0]):
    if config["general"]["prefer_low_pb"].lower() == "true":
        hist_bp_ratio_dict[datetime.strptime(d,"%Y-%m-%d").date()] = dict(map(lambda x: (x[1],1.0/float(x[2])), list(it_lstup)))
    else:
        hist_bp_ratio_dict[datetime.strptime(d,"%Y-%m-%d").date()] = dict(map(lambda x: (x[1],float(x[2])), list(it_lstup)))

start_date = datetime.strptime(config["general"]["start_date"],"%Y-%m-%d").date()
date_list = sorted(filter(lambda x: x >= start_date, list(set(hist_adj_px_dict.keys()).intersection(set(hist_bp_ratio_dict.keys())))))

rebalance_date_list = map(lambda y: y[1], filter(lambda x: x[0]%rebalance_interval==0, [(i,d) for i,d in enumerate(date_list)]))

###################################################
pos_dict = {}
cash = float(config["general"]["init_capital"])

# ###################################################
# mkt_timing_buy_lock  = False
# mkt_timing_sell_lock = False
# ###################################################

###################################################
for dt in rebalance_date_list:
    avb_constituent_set = set(map(lambda x: x[0], filter(lambda z: dt<=z[2], filter(lambda y: y[1]<=dt, hsi_hhi_constituents_list))))
    # print "avb_constituent_set: %s" % avb_constituent_set
    sym_bp_list = sorted(filter(lambda x: (x[0] in traded_symbol_set) and (x[0] in avb_constituent_set), list(hist_bp_ratio_dict[dt].items())), key=lambda x: x[0])
    if len(sym_bp_list) < min_no_of_avb_sym:
        continue
    symbol_list = map(lambda x: x[0], sym_bp_list)
    # print "symbol_list: %s" % symbol_list
    expected_rtn_list = map(lambda x: x[1], sym_bp_list)
    max_weight_list = map(lambda x: float(max_weight_dict[x]), symbol_list)
    from_tgt_rtn = min(map(lambda x: x[1], sym_bp_list))
    to_tgt_rtn = max(map(lambda x: x[1], sym_bp_list))

    ###################################################
    specific_riskiness_list = len(hedging_symbol_list+symbol_list) * [0.0]
    hist_adj_px_list_fil = sorted(filter(lambda x: x[0] <= dt, hist_adj_px_list), key=lambda y: y[0])
    hedging_sym_time_series_list = map(lambda s: map(lambda ts: (ts[0],ts[2]), filter(lambda h: h[1] == s, hist_adj_px_list_fil)), hedging_symbol_list)
    sym_time_series_list = map(lambda s: map(lambda ts: (ts[0],ts[2]), filter(lambda x: x[1] == s, hist_adj_px_list_fil)), symbol_list)
    aug_cov_matrix,annualized_sd_list,annualized_adj_sd_list = calc_cov_matrix_annualized(hedging_sym_time_series_list+sym_time_series_list, specific_riskiness_list)
    cov_matrix = aug_cov_matrix
    for i in range(len(hedging_symbol_list)):
        cov_matrix = np.delete(cov_matrix, 0, 0)
        cov_matrix = np.delete(cov_matrix, 0, 1)
    ###################################################

    ###################################################
    if config["general"]["construction_method"] == "log_optimal_growth":
        log_optimal_sol_list = log_optimal_growth(symbol_list, expected_rtn_list, cov_matrix, max_weight_list)
        if log_optimal_sol_list is None:
            continue
        log_optimal_sol_list = list(log_optimal_sol_list["result"]['x'])
    ###################################################

    ###################################################
    if (config["general"]["construction_method"] == "markowitz_max_kelly_f") or (config["general"]["construction_method"] == "markowitz_max_sharpe"):
        markowitz_max_sharpe_sol_list = []
        markowitz_max_kelly_f_sol_list = []
        for i in range(N):
            mu_p = from_tgt_rtn + (to_tgt_rtn - from_tgt_rtn) * float(i)/float(N)
            if config["general"]["robust_optimization"].lower() == "true":
                tmp_sol_list = markowitz_robust(symbol_list, expected_rtn_list, cov_matrix, mu_p, max_weight_list, extract_sd_from_cov_matrix(cov_matrix))
            else:
                tmp_sol_list = markowitz(symbol_list, expected_rtn_list, cov_matrix, mu_p, max_weight_list)

            if tmp_sol_list is None:
                continue
            tmp_sol_list = list(tmp_sol_list["result"]['x'])

            sol_vec = np.asarray(tmp_sol_list)
            sol_vec_T = np.matrix(sol_vec).T

            frontier_port_exp_rtn = float(np.asarray(expected_rtn_list) * sol_vec_T)
            frontier_port_stdev = math.sqrt(float((sol_vec * cov_matrix) * sol_vec_T))
            frontier_port_sharpe_ratio = float(frontier_port_exp_rtn / frontier_port_stdev)

            frontier_port_kelly_f = float(frontier_port_exp_rtn / frontier_port_stdev / frontier_port_stdev)

            # print "frontier_port_kelly_f: %s" % (frontier_port_kelly_f)
            if (len(markowitz_max_sharpe_sol_list) == 0) or (frontier_port_sharpe_ratio > markowitz_max_sharpe_sol_list[0]):
                markowitz_max_sharpe_sol_list = [frontier_port_sharpe_ratio, tmp_sol_list]

            if len(markowitz_max_kelly_f_sol_list) == 0 or (frontier_port_kelly_f > markowitz_max_kelly_f_sol_list[0]):
                markowitz_max_kelly_f_sol_list = [frontier_port_kelly_f, tmp_sol_list]

        if (len(markowitz_max_sharpe_sol_list) == 0) or (len(markowitz_max_kelly_f_sol_list) == 0):
            continue
        markowitz_max_sharpe_sol_list = markowitz_max_sharpe_sol_list[1]
        markowitz_max_kelly_f_sol_list = markowitz_max_kelly_f_sol_list[1]
    ###################################################

    if config["general"]["construction_method"] == "log_optimal_growth":
        sol_list = log_optimal_sol_list
    elif config["general"]["construction_method"] == "markowitz_max_kelly_f":
        sol_list = markowitz_max_kelly_f_sol_list
    elif config["general"]["construction_method"] == "markowitz_max_sharpe":
        sol_list = markowitz_max_sharpe_sol_list
    ###################################################

    ###################################################
    # calculation of beta
    ###################################################
    aug_sol_list = len(hedging_symbol_list)*[0.0]+sol_list
    port_beta_list = map(lambda h: sum(map(lambda x: x[0]*x[1]/aug_cov_matrix.tolist()[h[0]][h[0]], zip(aug_cov_matrix.tolist()[h[0]],aug_sol_list))), enumerate(hedging_symbol_list))

    ###################################################
    sym_weight_dict = dict(zip(symbol_list,map(lambda w: str(round(w,5)), sol_list)))
    sym_px_weight_list = map(lambda s: (s,str(hist_adj_px_dict[dt].get(s,0.0)),str(sym_weight_dict.get(s,0.0))), traded_symbol_list)

    ###################################################
    # sell all existing pos
    ###################################################
    cash += float(sum([hist_adj_px_dict[dt].get(s,filter(lambda x: x[1]==s, hist_adj_px_list)[-1][2])*pos for s,pos in pos_dict.items()]))
    pos_dict = {}
    ###################################################

    ###################################################
    if config["general"]["constant_capital"].lower() == "true":
        capital_to_use = float(config["general"]["init_capital"])
    else:
        capital_to_use = cash
    ###################################################
    
    ###################################################
    # buy back
    ###################################################
    pos_dict = dict([(s,capital_to_use*float(w)/hist_adj_px_dict[dt][s]) for s,w in sym_weight_dict.items()])
    ###################################################
    # decide whether to hedge, and hedge with the most correlated index
    ###################################################
    hsi_expected_return = filter(lambda x: x[0] <= dt, hsi_expected_return_list)[-1][1]
    most_correlated_idx_idx = sorted(enumerate(port_beta_list), key=lambda x: x[1])[-1][0]
    most_correlated_idx_sym = hedging_symbol_list[most_correlated_idx_idx]

    if config["general"]["use_beta_hedge"].lower() == "true":
        pos_dict[most_correlated_idx_sym] = -port_beta_list[most_correlated_idx_idx] * capital_to_use / hist_adj_px_dict[dt][most_correlated_idx_sym]
    else:
        h = min(max(port_beta_list[most_correlated_idx_idx] - hsi_expected_return / 100.0 / 0.7, 0.0), 1.0)
        pos_dict[most_correlated_idx_sym] = -h * capital_to_use / hist_adj_px_dict[dt][most_correlated_idx_sym]
    ###################################################
    # print "mkt val of pos: %s" % sum([hist_adj_px_dict[dt][s]*pos for s,pos in pos_dict.items()])
    print str(dt)+","+str(round(cash,0))+","+','.join(map(str,map(lambda x: round(x,5), port_beta_list)))+","+most_correlated_idx_sym+","+str(round(h,5))+","+str(len(sym_weight_dict))+","+','.join(map(lambda x: ':'.join(x), sym_px_weight_list))+','+",".join(map(lambda x: "pos_"+x[0]+'_'+str(x[1]), pos_dict.items()))
    cash -= float(sum([hist_adj_px_dict[dt].get(s,filter(lambda x: x[1]==s, hist_adj_px_list)[-1][2])*pos for s,pos in pos_dict.items()]))
    ###################################################
