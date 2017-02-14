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
from mvo import calc_cov_matrix_annualized,intWithCommas,justify_str,markowitz,markowitz_robust,log_optimal_growth,\
                read_file,extract_sd_from_cov_matrix,calc_return_list,get_hist_data_key_date,get_hist_data_key_sym,calc_expected_return,\
                get_industry_groups,preprocess_industry_groups

###################################################
AUDIT_DELAY = 3.0
###################################################
config = ConfigObj('config.ini')
config_common = ConfigObj(config["general"]["common_config"])

print "config," + str(config["general"]["init_capital"]) + "," + ','.join(map(lambda x: ":".join(map(str, x)), (config["general"].items()+config_common["general"].items()+config["max_weight"].items())))

###################################################
traded_symbol_set = set(config["general"]["traded_symbols"])
traded_symbol_list = sorted(config["general"]["traded_symbols"])
hedging_symbol_list = config["general"]["hedging_symbols"]
min_no_of_avb_sym = int(config["general"]["min_no_of_avb_sym"])
rebalance_interval = int(config["general"]["rebalance_interval"])
N = int(config["general"]["granularity"])
max_weight_dict = config["max_weight"]

er_hdg_dict = {}
er_hdg_key_list = filter(lambda x: "expected_return_" in x, config_common["general"].keys())

for k,v in dict(map(lambda x: (config_common["general"][x][0], config_common["general"][x][1:]), er_hdg_key_list)).items():
    er_hdg_dict[k] = sorted(map(lambda y: (datetime.strptime(y[1],"%Y-%m-%d").date(),float(y[2])), filter(lambda x: x[0]%2==0, zip(range(len(v)-1),v[:-1],v[1:]))), key=lambda x: x[0])

print '\n'.join(map(str, er_hdg_dict.items()))


hsi_hhi_constituents_list = map(lambda x: (x[0],datetime.strptime(x[1],"%Y-%m-%d").date(),datetime.strptime(x[2],"%Y-%m-%d").date()), read_file(config_common["general"]["hsi_hhi_constituents"]))
dow_constituents_list = sorted(map(lambda x: (datetime.strptime(x[0],"%Y-%m-%d").date(),x[1:]), read_file(config_common["general"]["dow_constituents"])), key=lambda y: y[0])

###################################################
# hist_intexp_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_intexp"])
# hist_cogs_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_cogs"])
# hist_revenue_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_revenue"])
# hist_mktcap_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_mktcap"])
# hist_oper_roe_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_oper_roe"])
# hist_totequity_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_totequity"])
# hist_operatingexp_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_operatingexp"])
hist_adj_px_dict = get_hist_data_key_date(config_common["hist_data"]["hist_adj_px"])
hist_adj_px_list_sorted = sorted(list(set(map(lambda x: (datetime.strptime(x[0],"%Y-%m-%d").date(),x[1],float(x[2])), read_file(config_common["hist_data"]["hist_adj_px"])))),key=lambda y: y[0])
hist_unadj_px_dict = get_hist_data_key_date(config_common["hist_data"]["hist_unadj_px"])

hist_bps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_bps"])
hist_totasset_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_totasset"])
hist_oper_eps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_oper_eps"])
hist_eps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_eps"])
hist_roa_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_roa"])
hist_stattaxrate_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_stattaxrate"])
hist_operincm_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_operincm"])
hist_costofdebt_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_costofdebt"])
hist_totliabps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_totliabps"])

###################################################
industry_groups_list = get_industry_groups(preprocess_industry_groups(config_common["industry_group"]))

###################################################
start_date = datetime.strptime(config["general"]["start_date"],"%Y-%m-%d").date()
date_list = sorted(filter(lambda d: (d >= start_date) and all(map(lambda hs: hs in hist_adj_px_dict[d], hedging_symbol_list)), set(hist_unadj_px_dict.keys()).intersection(set(hist_adj_px_dict.keys()))))
rebalance_date_list = map(lambda y: y[1], filter(lambda x: x[0]%rebalance_interval==0, [(i,d) for i,d in enumerate(date_list)]))

###################################################
pos_dict = {}
cash = float(config["general"]["init_capital"])

###################################################
for dt in rebalance_date_list:
    avb_constituent_set = set(map(lambda x: x[0], filter(lambda z: dt<=z[2], filter(lambda y: y[1]<=dt, hsi_hhi_constituents_list))) + filter(lambda x: x[0] <= dt, dow_constituents_list)[-1][1])
    # print "avb_constituent_set: %s" % avb_constituent_set

    ###################################################
    symbol_list = filter(lambda x: x in avb_constituent_set, sorted(list(traded_symbol_set)))
    symbol_list = filter(lambda x: x in hist_adj_px_dict[dt], symbol_list)
    symbol_list = filter(lambda x: x in hist_unadj_px_dict[dt], symbol_list)

    ###################################################
    # check whether we have enough stocks to choose from
    ###################################################
    if len(symbol_list) < min_no_of_avb_sym:
        continue

    expected_rtn_list = calc_expected_return(config_common,dt,symbol_list,hist_bps_dict,hist_unadj_px_dict,hist_operincm_dict,hist_totasset_dict,hist_totliabps_dict,hist_costofdebt_dict,hist_stattaxrate_dict,hist_oper_eps_dict,hist_eps_dict,hist_roa_dict,AUDIT_DELAY,False)
    # print str(dt) + ": " + ', '.join(map(lambda x: x[0]+":["+str(round(x[1],3))+"]:a_"+str(round(x[2],3))+";e_"+str(round(x[3],3))+";b_"+str(round(x[4],3)), zip(symbol_list,expected_rtn_list,expected_rtn_asset_driver_list,expected_rtn_external_driver_list,expected_rtn_bv_list)))

    ###################################################
    hsi_expected_return = filter(lambda x: x[0] <= dt, hsi_expected_return_list)[-1][1]
    if config["general"]["hedging_type"].lower() == "toge":
        max_weight_list = map(lambda x: 1.0, hedging_symbol_list) + map(lambda x: float(max_weight_dict.get(x,max_weight_dict["single_name"])), symbol_list)
        symbol_list = hedging_symbol_list + symbol_list
        expected_rtn_list = (len(hedging_symbol_list) * [hsi_expected_return/100.0]) + expected_rtn_list
    else:
        max_weight_list = map(lambda x: float(max_weight_dict.get(x,max_weight_dict["single_name"])), symbol_list)
    from_tgt_rtn = min(expected_rtn_list)
    to_tgt_rtn = max(expected_rtn_list)

    ###################################################
    specific_riskiness_list = len(hedging_symbol_list+symbol_list) * [0.0]
    hist_adj_px_list_fil_sorted = sorted(filter(lambda x: x[0] <= dt, hist_adj_px_list_sorted), key=lambda y: y[0])
    sym_time_series_list = map(lambda s: map(lambda ts: (ts[0],ts[2]), filter(lambda x: x[1] == s, hist_adj_px_list_fil_sorted)), symbol_list)
    if config["general"]["hedging_type"].lower() == "toge":
        aug_cov_matrix,annualized_sd_list,annualized_adj_sd_list = calc_cov_matrix_annualized(sym_time_series_list, specific_riskiness_list)
    else:
        hedging_sym_time_series_list = map(lambda s: map(lambda ts: (ts[0],ts[2]), filter(lambda h: h[1] == s, hist_adj_px_list_fil_sorted)), hedging_symbol_list)
        aug_cov_matrix,annualized_sd_list,annualized_adj_sd_list = calc_cov_matrix_annualized(hedging_sym_time_series_list+sym_time_series_list, specific_riskiness_list)

    cov_matrix = aug_cov_matrix
    if config["general"]["hedging_type"].lower() != "toge":
        for i in range(len(hedging_symbol_list)):
            cov_matrix = np.delete(cov_matrix, 0, 0)
            cov_matrix = np.delete(cov_matrix, 0, 1)
        # print "aug_cov %s" % (aug_cov_matrix)
    ###################################################

    ###################################################
    if config["general"]["construction_method"] == "log_optimal_growth":
        log_optimal_sol_list = log_optimal_growth(symbol_list, expected_rtn_list, cov_matrix, max_weight_list, industry_groups_list, float(config["max_weight"]["industry"]))
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
                tmp_sol_list = markowitz(symbol_list, expected_rtn_list, cov_matrix, mu_p, max_weight_list, 0.0, industry_groups_list, float(config["max_weight"]["industry"]))

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
    sym_px_weight_list = map(lambda s: (s,str(hist_adj_px_dict[dt].get(s,0.0)),str(sym_weight_dict.get(s,0.0))), symbol_list)

    ###################################################
    # sell all existing pos
    ###################################################
    cash += float(sum([hist_adj_px_dict[dt].get(s,filter(lambda x: x[1]==s, hist_adj_px_list_sorted)[-1][2])*pos for s,pos in pos_dict.items()]))
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
    most_correlated_idx_idx = sorted(enumerate(port_beta_list), key=lambda x: x[1])[-1][0]
    most_correlated_idx_sym = hedging_symbol_list[most_correlated_idx_idx]

    h = -1.0
    if config["general"]["hedging_type"].lower() == "beta":
        h = port_beta_list[most_correlated_idx_idx]
        pos_dict[most_correlated_idx_sym] = -h * capital_to_use / hist_adj_px_dict[dt][most_correlated_idx_sym]
    elif config["general"]["hedging_type"].lower() == "riskadj":
        h = min(max(port_beta_list[most_correlated_idx_idx] - hsi_expected_return / 100.0 / 0.7, 0.0), 1.0)
        pos_dict[most_correlated_idx_sym] = -h * capital_to_use / hist_adj_px_dict[dt][most_correlated_idx_sym]
    elif config["general"]["hedging_type"].lower() == "logopt":
        sol_vec = np.asarray(sol_list)
        sol_vec_T = np.matrix(sol_vec).T
        port_var = float((sol_vec * cov_matrix) * sol_vec_T)
        hedge_var = aug_cov_matrix.tolist()[most_correlated_idx_idx][most_correlated_idx_idx]

        hv_list = []
        for i in range(100):
            h_tmp = -float(i)/100.0
            hv_list.append((h_tmp, h_tmp * hsi_expected_return - ((port_var + h_tmp*h_tmp*hedge_var + 2*h_tmp*port_beta_list[most_correlated_idx_idx]*hedge_var) / 2 / math.pow(1+h_tmp,2))))

        # print "hv_list: %s" % hv_list
        h = -sorted(hv_list, key=lambda x: x[1])[-1][0]
        pos_dict[most_correlated_idx_sym] = -h * capital_to_use / hist_adj_px_dict[dt][most_correlated_idx_sym]
    elif config["general"]["hedging_type"].lower() == "half":
        h = 0.5
        pos_dict[most_correlated_idx_sym] = -h * capital_to_use / hist_adj_px_dict[dt][most_correlated_idx_sym]
    ###################################################
    # print "mkt val of pos: %s" % sum([hist_adj_px_dict[dt][s]*pos for s,pos in pos_dict.items()])
    print str(dt)+","+str(round(cash,0))+","+','.join(map(str,map(lambda x: round(x,5), port_beta_list)))+","+most_correlated_idx_sym+","+str(round(h,5))+",["+str(len(sym_weight_dict))+"],"+','.join(map(lambda x: ':'.join(x), sym_px_weight_list))+','+",".join(map(lambda x: "pos_"+x[0]+'_'+str(x[1]), pos_dict.items()))
    cash -= float(sum([hist_adj_px_dict[dt].get(s,filter(lambda x: x[1]==s, hist_adj_px_list_sorted)[-1][2])*pos for s,pos in pos_dict.items()]))
    ###################################################
