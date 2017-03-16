#!/usr/bin/env python

###################################################
# test if historical realized return falls within our confidence interval
###################################################

from configobj import ConfigObj
import sys
import math
from datetime import datetime, timedelta, date
import time
import numpy as np
from itertools import groupby

import os
sys.path.append(os.path.dirname(sys.path[0]))
from qvi import calc_cov_matrix_annualized,intWithCommas,justify_str,markowitz,markowitz_robust,markowitz_sharpe,log_optimal_growth,\
                read_file,extract_sd_from_cov_matrix,calc_return_list,get_hist_data_key_date,get_hist_data_key_sym,calc_irr_mean_ci_before_20170309,calc_irr_mean_cov_after_20170309,\
                get_industry_groups,preprocess_industry_groups,get_port_and_hdg_cov_matrix,log_optimal_hedge,sharpe_hedge,minvar_hedge

###################################################
AUDIT_DELAY = 3.0
###################################################
config = ConfigObj('config.ini')
config_common = ConfigObj(config["general"]["common_config"])

###################################################
traded_symbol_set = set(config["general"]["traded_symbols"])
traded_symbol_list = sorted(config["general"]["traded_symbols"])
rebalance_interval = int(config["general"]["rebalance_interval"])

###################################################
hist_adj_px_dict = get_hist_data_key_date(config_common["hist_data"]["hist_adj_px"])
hist_adj_px_list_sorted = sorted(list(set(map(lambda x: (datetime.strptime(x[0],"%Y-%m-%d").date(),x[1],float(x[2])), read_file(config_common["hist_data"]["hist_adj_px"])))),key=lambda y: y[0])
hist_unadj_px_dict = get_hist_data_key_date(config_common["hist_data"]["hist_unadj_px"])

hist_bps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_bps"])
hist_eps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_eps"])
hist_roa_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_roa"])
hist_totliabps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_totliabps"])

###################################################
industry_groups_list = get_industry_groups(preprocess_industry_groups(config_common["industry_group"]))

###################################################
start_date = datetime.strptime(config["general"]["start_date"],"%Y-%m-%d").date()
date_list = sorted(filter(lambda d: (d >= start_date), set(hist_unadj_px_dict.keys()).intersection(set(hist_adj_px_dict.keys()))))
rebalance_date_list = map(lambda y: y[1], filter(lambda x: x[0]%rebalance_interval==0, [(i,d) for i,d in enumerate(date_list)]))

###################################################
hit_or_miss_list = []
hit_or_miss_total_count = 0.0
for dt in rebalance_date_list:
    symbol_list = sorted(list(traded_symbol_set))
    symbol_list = filter(lambda x: x in hist_adj_px_dict[dt], symbol_list)
    symbol_list = filter(lambda x: x in hist_unadj_px_dict[dt], symbol_list)

    if len(symbol_list) == 0:
        continue
    specific_riskiness_list = len(symbol_list) * [0.0]
    hist_adj_px_list_fil_sorted = sorted(filter(lambda x: x[0] <= dt, hist_adj_px_list_sorted), key=lambda y: y[0])
    sym_time_series_list = map(lambda s: map(lambda ts: (ts[0],ts[2]), filter(lambda x: x[1] == s, hist_adj_px_list_fil_sorted)), symbol_list)
    cov_matrix,annualized_sd_list,annualized_adj_sd_list = calc_cov_matrix_annualized(sym_time_series_list, specific_riskiness_list)

    # irr_mean_ci_tuple_list = map(lambda s: calc_irr_mean_ci_before_20170309(config_common,dt,s[1],math.sqrt(cov_matrix[s[0]][s[0]]),hist_unadj_px_dict,hist_eps_dict,hist_bps_dict,int(config["general"]["confidence_level"]),AUDIT_DELAY,False), enumerate(symbol_list))
    calc_irr_mean_cov_after_20170309(config_common,dt,symbol_list,hist_bps_dict,hist_unadj_px_dict,hist_totliabps_dict,hist_eps_dict,hist_roa_dict,AUDIT_DELAY,True)

    # prep_tot_rtn_list = map(lambda s: map(lambda v: v[2], filter(lambda x: x[0]>=dt, filter(lambda x: x[1]==s, hist_adj_px_list_sorted))), symbol_list)
    #
    # ###################################################
    # num_of_days_actual_rtn = int(config["general"]["num_of_days_actual_rtn"])
    # annualized_tot_rtn_list = map(lambda x: round(math.pow(x[num_of_days_actual_rtn]/x[0],252.0/float(num_of_days_actual_rtn))-1.0,5) if len(x) > num_of_days_actual_rtn else None, prep_tot_rtn_list)
    # ###################################################
    #
    # hit_or_miss_list.append(map(lambda x: ( 1.0 if ((x[1]>x[0][1] and x[1]<x[0][2])) else -1.0 ) if all(map(lambda y: y is not None, x[0]+[x[1]])) else 0.0, zip(irr_mean_ci_tuple_list,annualized_tot_rtn_list)))
    # cum_hit_count = float(len(filter(lambda x: x > 0.001, map(lambda x: x[0], hit_or_miss_list))))
    # hit_or_miss_total_count = len(filter(lambda x: abs(x) > 0.001, map(lambda x: x[0], hit_or_miss_list)))
    # hit_prob_list = [round(cum_hit_count/hit_or_miss_total_count,5) if hit_or_miss_total_count > 0 else None]
    # print str(dt)+" "+'='.join(map(str, zip(symbol_list,irr_mean_ci_tuple_list,hit_prob_list,annualized_tot_rtn_list)))
