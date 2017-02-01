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
from mvo import calc_cov_matrix_annualized,conv_to_hkd,intWithCommas,justify_str,markowitz,markowitz_robust,log_optimal_growth,read_file,extract_sd_from_cov_matrix,calc_return_list

###################################################
AUDIT_DELAY = 3.0
###################################################
def get_hist_data_key_date(filename):
    rtn_dict = {}
    for d,it_lstup in groupby(sorted(read_file(filename), key=lambda x: x[0]), lambda x: x[0]):
        rtn_dict[datetime.strptime(d,"%Y-%m-%d").date()] = dict(map(lambda y: (y[1],float(y[2])), filter(lambda x: abs(float(x[2])) > 0.0001, list(it_lstup))))
    return rtn_dict

def get_hist_data_key_sym(filename):
    rtn_dict = {}
    for s,it_lstup in groupby(sorted(read_file(filename), key=lambda x: x[1]), lambda x: x[1]):
        rtn_dict[s] = sorted(map(lambda y: (datetime.strptime(y[0],"%Y-%m-%d").date(),float(y[2])), filter(lambda x: abs(float(x[2])) > 0.0001, list(it_lstup))), key=lambda x: x[0])
    return rtn_dict

def shift_back_n_months(dt,n):
    return dt - timedelta(weeks = float(52.0/12.0*float(n)))

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
dow_constituents_list = sorted(map(lambda x: (datetime.strptime(x[0],"%Y-%m-%d").date(),x[1:]), read_file(config["general"]["dow_constituents"])), key=lambda y: y[0])

###################################################
hist_adj_px_dict = get_hist_data_key_date(config["hist_data"]["hist_adj_px"])
hist_adj_px_list_sorted = sorted(list(set(map(lambda x: (datetime.strptime(x[0],"%Y-%m-%d").date(),x[1],float(x[2])), read_file(config["hist_data"]["hist_adj_px"])))),key=lambda y: y[0])
hist_unadj_px_dict = get_hist_data_key_date(config["hist_data"]["hist_unadj_px"])
hist_bps_dict = get_hist_data_key_sym(config["hist_data"]["hist_bps"])
hist_totasset_dict = get_hist_data_key_sym(config["hist_data"]["hist_totasset"])
hist_totequity_dict = get_hist_data_key_sym(config["hist_data"]["hist_totequity"])
# hist_intexp_dict = get_hist_data_key_sym(config["hist_data"]["hist_intexp"])
hist_operatingexp_dict = get_hist_data_key_sym(config["hist_data"]["hist_operatingexp"])
# hist_cogs_dict = get_hist_data_key_sym(config["hist_data"]["hist_cogs"])
# hist_revenue_dict = get_hist_data_key_sym(config["hist_data"]["hist_revenue"])
# hist_mktcap_dict = get_hist_data_key_sym(config["hist_data"]["hist_mktcap"])
# hist_oper_roe_dict = get_hist_data_key_sym(config["hist_data"]["hist_oper_roe"])
hist_oper_eps_dict = get_hist_data_key_sym(config["hist_data"]["hist_oper_eps"])
hist_stattaxrate_dict = get_hist_data_key_sym(config["hist_data"]["hist_stattaxrate"])
hist_operincm_dict = get_hist_data_key_sym(config["hist_data"]["hist_operincm"])
hist_costofdebt_dict = get_hist_data_key_sym(config["hist_data"]["hist_costofdebt"])
hist_totliabps_dict = get_hist_data_key_sym(config["hist_data"]["hist_totliabps"])


###################################################
start_date = datetime.strptime(config["general"]["start_date"],"%Y-%m-%d").date()
date_list = sorted(filter(lambda x: x >= start_date, list(set(hist_unadj_px_dict.keys()).intersection(set(hist_adj_px_dict.keys())))))
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
    avb_constituent_set = set(map(lambda x: x[0], filter(lambda z: dt<=z[2], filter(lambda y: y[1]<=dt, hsi_hhi_constituents_list))) + filter(lambda x: x[0] <= dt, dow_constituents_list)[-1][1])
    # print "avb_constituent_set: %s" % avb_constituent_set

    ###################################################
    symbol_list = filter(lambda x: x in avb_constituent_set, sorted(list(traded_symbol_set)))
    symbol_list = filter(lambda x: x in hist_adj_px_dict[dt], symbol_list)
    symbol_list = filter(lambda x: x in hist_unadj_px_dict[dt], symbol_list)
    symbol_list = filter(lambda x: x in hist_bps_dict, symbol_list)
    symbol_list = filter(lambda x: x in hist_totasset_dict, symbol_list)

    ###################################################
    # book-to-price
    # asset-to-price
    ###################################################
    bp_list = []
    conser_bp_list = []
    conser_bp_dict = {}
    for sym in symbol_list:
        bps_list = map(lambda z: z[1], filter(lambda y: y[0] > shift_back_n_months(dt,5*12+AUDIT_DELAY), filter(lambda x: x[0] <= shift_back_n_months(dt,AUDIT_DELAY), hist_bps_dict[sym])))
        ###################################################
        if len(bps_list) >= 5:
            bps_r = calc_return_list(bps_list)
            bps_r = bps_r[1:][:-1]
            m = sum(bps_r)/len(bps_r)
            sd = np.std(np.asarray(bps_r))
            conser_bp = bps_list[-1] * (1 + m - float(config["general"]["bp_stdev"]) * sd) / hist_unadj_px_dict[dt][sym]
            conser_bp_list.append(conser_bp)
            conser_bp_dict[sym] = conser_bp
        else:
            conser_bp_list.append(0.0)
            conser_bp_dict[sym] = 0.0
        ###################################################
        if len(bps_list) > 0:
            bp_list.append(bps_list[-1] / hist_unadj_px_dict[dt][sym])
        else:
            bp_list.append(0.0)
        ###################################################

    ###################################################
    # check whether we have enough stocks to choose from
    ###################################################
    if len(symbol_list) < min_no_of_avb_sym:
        continue

    if config["general"]["expected_return_est_method"].lower() == "sunny":
        expected_rtn_list = []
        expected_rtn_asset_driver_list = []
        expected_rtn_external_driver_list = []
        expected_rtn_bproe_list = []
        for sym in symbol_list:
            ###################################################
            w_a_,w_e_,w_b_ = map(float, config["expected_rtn_ast_ext_bv"][sym])
            w_a = w_a_/sum([w_a_,w_e_,w_b_])
            w_e = w_e_/sum([w_a_,w_e_,w_b_])
            w_b = w_b_/sum([w_a_,w_e_,w_b_])

            ###################################################
            # asset driver
            ###################################################
            conser_oper_roa = None

            oper_roa_list = []
            oper_incm_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*5+AUDIT_DELAY), filter(lambda x: x[0] <= shift_back_n_months(dt,AUDIT_DELAY), hist_operincm_dict.get(sym,[])))
            for oi_dt,oper_incm in oper_incm_list:
                totasset_list = filter(lambda x: x[0] <= shift_back_n_months(oi_dt,6), hist_totasset_dict.get(sym,[]))
                if len(totasset_list) > 0:
                    oper_roa_list.append((oi_dt,oper_incm/totasset_list[-1][1]))

            oper_roa_list = map(lambda x: x[1], oper_roa_list)

            ###################################################
            oper_date_list = map(lambda x: x[0], oper_incm_list)
            if len(oper_date_list) >= 3:
                annualization_factor = round(365.0/min(map(lambda x: (x[0]-x[1]).days, zip(oper_date_list[1:],oper_date_list[:-1]))),0)
            else:
                annualization_factor = 0.0
            ###################################################

            if len(oper_roa_list) >= 5:
                oper_roa_list = sorted(oper_roa_list)[1:][:-1]
                m = sum(oper_roa_list)/len(oper_roa_list)
                sd = np.std(np.asarray(calc_return_list(oper_roa_list)))
                conser_oper_roa = annualization_factor * (m * (1 - float(config["general"]["roa_est_stdev"]) * sd))
            else:
                conser_oper_roa = 0.0

            bps_list = filter(lambda x: x[0] <= shift_back_n_months(dt,AUDIT_DELAY), hist_bps_dict.get(sym,[]))
            bps = bps_list[-1][1] if len(bps_list) > 0 else 0.0

            costofdebt_list = filter(lambda x: x[0] <= shift_back_n_months(dt,AUDIT_DELAY), hist_costofdebt_dict.get(sym,[]))
            totliabps_list = filter(lambda x: x[0] <= shift_back_n_months(dt,AUDIT_DELAY), hist_totliabps_dict.get(sym,[]))
            if len(costofdebt_list) > 0 and len(totliabps_list) > 0:
                iL = costofdebt_list[-1][1]/100.0 * totliabps_list[-1][1]
            else:
                iL = 0.0
            if len(totliabps_list) > 0:
                totliabps = totliabps_list[-1][1]
            else:
                totliabps = 0.0

            stattaxrate_list = filter(lambda x: x[0] <= shift_back_n_months(dt,AUDIT_DELAY), hist_stattaxrate_dict.get(sym,[]))
            if len(stattaxrate_list) > 0:
                taxrate = stattaxrate_list[-1][1]
            else:
                taxrate = 0.0

            expected_rtn_asset_driver_list.append(w_a * (1-taxrate)*(conser_oper_roa*(totliabps+bps)-iL)/hist_unadj_px_dict[dt][sym])

            ###################################################
            # external driver
            ###################################################
            oper_eps_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*5+AUDIT_DELAY), filter(lambda x: x[0] <= shift_back_n_months(dt,AUDIT_DELAY), hist_oper_eps_dict.get(sym,[])))
            oper_eps_list = map(lambda x: x[1], oper_eps_list)
            oper_eps_pchg_list = map(lambda x: float(x[0]-x[1])/x[1], zip(oper_eps_list[1:],oper_eps_list[:-1]))
            if len(oper_eps_pchg_list) >= 5:
                oper_eps_pchg_list = sorted(oper_eps_pchg_list)[1:][:-1]
                m = sum(oper_eps_pchg_list)/len(oper_eps_pchg_list)
                sd = np.std(np.asarray(oper_eps_pchg_list))
                oper_eps_list = map(lambda z: z[1], sorted(sorted(enumerate(oper_eps_list), key=lambda x: x[1])[1:][:-1], key=lambda y: y[0]))
                conser_oper_eps = oper_eps_list[-1] * (1 + annualization_factor * (m - float(config["general"]["roa_est_stdev"]) * sd))
            else:
                conser_oper_eps = 0.0

            expected_rtn_external_driver_list.append(w_e * (1-taxrate)*(conser_oper_eps-iL)/hist_unadj_px_dict[dt][sym])
            expected_rtn_bproe_list.append(w_b * float(config["general"]["typical_roe"])*conser_bp_dict[sym])

        expected_rtn_list = map(lambda x: sum(x), zip(expected_rtn_asset_driver_list,expected_rtn_external_driver_list,expected_rtn_bproe_list))

    elif config["general"]["expected_return_est_method"].lower() == "bp":
        expected_rtn_list = map(lambda x: x/100.0, conser_bp_list)

    # print str(dt) + ": " + ', '.join(map(lambda x: x[0]+":"+str(x[1]), zip(symbol_list,expected_rtn_list)))
    # print "expected_rtn_list: %s: %s" % (dt,'|'.join(map(lambda x: x[0]+":"+str(round(x[1],4)), sorted(zip(symbol_list,expected_rtn_list),key=lambda x: x[1]))))

    ###################################################
    max_weight_list = map(lambda x: float(max_weight_dict.get(x,max_weight_dict["default"])), symbol_list)
    from_tgt_rtn = min(expected_rtn_list)
    to_tgt_rtn = max(expected_rtn_list)

    ###################################################
    specific_riskiness_list = len(hedging_symbol_list+symbol_list) * [0.0]
    hist_adj_px_list_fil_sorted = sorted(filter(lambda x: x[0] <= dt, hist_adj_px_list_sorted), key=lambda y: y[0])
    hedging_sym_time_series_list = map(lambda s: map(lambda ts: (ts[0],ts[2]), filter(lambda h: h[1] == s, hist_adj_px_list_fil_sorted)), hedging_symbol_list)
    sym_time_series_list         = map(lambda s: map(lambda ts: (ts[0],ts[2]), filter(lambda x: x[1] == s, hist_adj_px_list_fil_sorted)), symbol_list)

    aug_cov_matrix,annualized_sd_list,annualized_adj_sd_list = calc_cov_matrix_annualized(hedging_sym_time_series_list+sym_time_series_list, specific_riskiness_list)
    cov_matrix = aug_cov_matrix
    for i in range(len(hedging_symbol_list)):
        cov_matrix = np.delete(cov_matrix, 0, 0)
        cov_matrix = np.delete(cov_matrix, 0, 1)
    # print "aug_cov %s" % (aug_cov_matrix)
    ###################################################

    ###################################################
    if config["general"]["construction_method"] == "log_optimal_growth":
        log_optimal_sol_list = log_optimal_growth(symbol_list, expected_rtn_list, cov_matrix, max_weight_list)
        if log_optimal_sol_list is None:
            print "shit"
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
                tmp_sol_list = markowitz(symbol_list, expected_rtn_list, cov_matrix, mu_p, max_weight_list, 0.0)

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
    hsi_expected_return = filter(lambda x: x[0] <= dt, hsi_expected_return_list)[-1][1]
    most_correlated_idx_idx = sorted(enumerate(port_beta_list), key=lambda x: x[1])[-1][0]
    most_correlated_idx_sym = hedging_symbol_list[most_correlated_idx_idx]

    h = -1.0
    if config["general"]["hedging_type"].lower() == "beta":
        h = port_beta_list[most_correlated_idx_idx]
        pos_dict[most_correlated_idx_sym] = -h * capital_to_use / hist_adj_px_dict[dt][most_correlated_idx_sym]
    elif config["general"]["hedging_type"].lower() == "smart":
        h = min(max(port_beta_list[most_correlated_idx_idx] - hsi_expected_return / 100.0 / 0.7, 0.0), 1.0)
        pos_dict[most_correlated_idx_sym] = -h * capital_to_use / hist_adj_px_dict[dt][most_correlated_idx_sym]
    ###################################################
    # print "mkt val of pos: %s" % sum([hist_adj_px_dict[dt][s]*pos for s,pos in pos_dict.items()])
    print str(dt)+","+str(round(cash,0))+","+','.join(map(str,map(lambda x: round(x,5), port_beta_list)))+","+most_correlated_idx_sym+","+str(round(h,5))+",["+str(len(sym_weight_dict))+"],"+','.join(map(lambda x: ':'.join(x), sym_px_weight_list))+','+",".join(map(lambda x: "pos_"+x[0]+'_'+str(x[1]), pos_dict.items()))
    cash -= float(sum([hist_adj_px_dict[dt].get(s,filter(lambda x: x[1]==s, hist_adj_px_list_sorted)[-1][2])*pos for s,pos in pos_dict.items()]))
    ###################################################
