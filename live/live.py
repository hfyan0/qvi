#!/usr/bin/env python
from configobj import ConfigObj
import sys
import math
from datetime import datetime, timedelta
import numpy as np

import os
sys.path.append(os.path.dirname(sys.path[0]))
from mvo import calc_cov_matrix_annualized,conv_to_hkd,intWithCommas,justify_str,markowitz,read_file

###################################################
config = ConfigObj('config.ini')

symbol_list = sorted([i for k in map(lambda x: config["general"][x].split(','), filter(lambda x: "traded_symbols" in x, config["general"].keys())) for i in k])
# print "Symbols: %s" % (','.join(symbol_list))

specific_riskiness_list = map(lambda s: float(config["specific_riskiness"].get(s,0)), symbol_list)
# print "Riskiness: %s" % (','.join(map(str, specific_riskiness_list)))

###################################################
# read time series of prices
###################################################
sym_data_list = map(lambda s: read_file(config["data_path"][s]), symbol_list)
sym_data_list = map(lambda x: filter(lambda y: len(y) > 5, x), sym_data_list)
sym_time_series_list = map(lambda data_list: map(lambda csv_fields: (datetime.strptime(csv_fields[0],"%Y-%m-%d"),float(csv_fields[5])), data_list), sym_data_list)

expected_rtn_dict = dict(map(lambda x: (x[0],float(x[1])), read_file(config["general"]["expected_return_file"])))
expected_rtn_list = map(lambda x: expected_rtn_dict.get(x,0), symbol_list)

cov_matrix,annualized_sd_list,annualized_adj_sd_list = calc_cov_matrix_annualized(sym_time_series_list, specific_riskiness_list)

###################################################
# inclusion of cash
###################################################
# symbol_list.append("Cash")    
# expected_rtn_list.append(0.0)    
# cov_matrix = np.column_stack([cov_matrix, np.zeros(cov_matrix.shape[0])])
# newrow = np.transpose(np.asarray(map(lambda x: 0.0, range(len(symbol_list)))))
# cov_matrix = np.vstack([cov_matrix, newrow])

if (config["general"]["printCovMatrix"].lower() == "true"):
    print
    print "cov_matrix"
    print cov_matrix

str_expected_rtn_list = map(lambda x: str(round(x,4)*100)+'%', expected_rtn_list)
str_annualized_sd_list = map(lambda x: str(round(x,4)*100)+'%', annualized_sd_list)
str_annualized_adj_sd_list = map(lambda x: str(round(x,4)*100)+'%', annualized_adj_sd_list)
str_sharpe_list = map(lambda x: str(round(x[0]/x[1],3)), zip(expected_rtn_list,annualized_adj_sd_list))
print
print "%s%s%s%s%s" % (justify_str("Symbol",8),justify_str("E[r]",10),justify_str("SD[r]",10),justify_str("SD_adj[r]",10),justify_str("Sharpe",10))
print '\n'.join(map(lambda x: justify_str(x[0],8)+justify_str(x[1],10)+justify_str(x[2],10)+justify_str(x[3],10)+justify_str(x[4],10), zip(symbol_list,str_expected_rtn_list,str_annualized_sd_list,str_annualized_adj_sd_list,str_sharpe_list)))

sorted_expected_rtn_list = sorted(expected_rtn_list)
from_tgt_rtn = sorted_expected_rtn_list[0]
to_tgt_rtn = sorted_expected_rtn_list[-1]
mu_sd_sharpe_soln_list = []
N = 1000

max_weight_dict = config["max_weight"]
max_weight_list = map(lambda x: float(max_weight_dict.get(x,1.0)), symbol_list)

# print "expected_rtn_list: %s" % (expected_rtn_list)
# print "cov_matrix: %s" % (cov_matrix)
# print "max_weight_list: %s" % (max_weight_list)

for i in range(N):

    mu_p = from_tgt_rtn + (to_tgt_rtn - from_tgt_rtn) * float(i)/float(N)
    # mu_p = to_tgt_rtn * float(i)/float(N)
    sol_list = markowitz(symbol_list, expected_rtn_list, cov_matrix, mu_p, max_weight_list)

    if sol_list is None:
        continue

    sol_list = list(sol_list["result"]['x'])

    sol_vec = np.asarray(sol_list)
    sol_vec_T = np.matrix(sol_vec).T

    market_port_exp_dollar_rtn_list = map(lambda x: x[0]*x[1]*float(config["general"]["capital"]), zip(sol_list,expected_rtn_list))
    market_port_exp_rtn = float(np.asarray(expected_rtn_list) * sol_vec_T)
    market_port_stdev = math.sqrt(float((sol_vec * cov_matrix) * sol_vec_T))
    market_port_sharpe_ratio = float(market_port_exp_rtn / market_port_stdev)
    market_port_kelly_f_true = float(market_port_exp_rtn / market_port_stdev / market_port_stdev)
    market_port_kelly_f_for_ranking = min(market_port_kelly_f_true, float(config["general"]["risk_aversion"]))
    market_port_kelly_f = min(market_port_kelly_f_true, float(config["general"]["max_allowed_leverage"]))

    target_port_exp_rtn_aft_costs_for_ranking = (market_port_exp_rtn * market_port_kelly_f_for_ranking) - (max(market_port_kelly_f_for_ranking-1.0,0.0)*float(config["general"]["financing_cost"]))
    target_port_exp_rtn_aft_costs = (market_port_exp_rtn * market_port_kelly_f) - (max(market_port_kelly_f-1.0,0.0)*float(config["general"]["financing_cost"]))
    target_port_stdev = (market_port_stdev * market_port_kelly_f)
    target_port_sharpe_ratio = (target_port_exp_rtn_aft_costs / target_port_stdev)
    target_port_exp_dollar_rtn_list = map(lambda x: x * market_port_kelly_f, market_port_exp_dollar_rtn_list)
    sol_list = map(lambda x: x * market_port_kelly_f, sol_list)

    if (len(mu_sd_sharpe_soln_list) == 0) or (target_port_exp_rtn_aft_costs_for_ranking > mu_sd_sharpe_soln_list[0]):
        mu_sd_sharpe_soln_list = []
        mu_sd_sharpe_soln_list.append(float(target_port_exp_rtn_aft_costs_for_ranking))
        mu_sd_sharpe_soln_list.append(float(target_port_exp_rtn_aft_costs))
        mu_sd_sharpe_soln_list.append(float(target_port_stdev))
        mu_sd_sharpe_soln_list.append(float(target_port_sharpe_ratio))
        mu_sd_sharpe_soln_list.append(float(market_port_exp_rtn))
        mu_sd_sharpe_soln_list.append(float(market_port_stdev))
        mu_sd_sharpe_soln_list.append(float(market_port_sharpe_ratio))
        mu_sd_sharpe_soln_list.append(float(market_port_kelly_f))
        mu_sd_sharpe_soln_list.append(float(market_port_kelly_f_true))
        mu_sd_sharpe_soln_list.append(float(market_port_kelly_f_for_ranking))
        mu_sd_sharpe_soln_list.append(target_port_exp_dollar_rtn_list)
        mu_sd_sharpe_soln_list.append(sol_list)

if len(mu_sd_sharpe_soln_list) == 0:
    print "No solution found"
    sys.exit(0)

target_port_exp_rtn_aft_costs_for_ranking, target_port_exp_rtn_aft_costs, target_port_stdev, target_port_sharpe_ratio, market_port_exp_rtn, market_port_stdev, market_port_sharpe_ratio, market_port_kelly_f, market_port_kelly_f_true, market_port_kelly_f_for_ranking, target_port_exp_dollar_rtn_list, sol_list = tuple(mu_sd_sharpe_soln_list)

print
print "Market portfolio:  E[r] = %s stdev = %s Sharpe ratio = %s Risk aversion: %s Kelly f* = %s (for_ranking: %s, used: %s)" % (str(round(market_port_exp_rtn*100, 3)) + " %", str(round(market_port_stdev*100,3)) + " %", round(market_port_sharpe_ratio,3), config["general"]["risk_aversion"], str(round(market_port_kelly_f_true,3)), round(market_port_kelly_f_for_ranking,3), round(market_port_kelly_f,3))
print "Target portfolio:  E[r] = %s stdev = %s Sharpe ratio = %s" % (str(round(target_port_exp_rtn_aft_costs*100, 3)) + " %", str(round(target_port_stdev*100,3)) + " %", round(target_port_sharpe_ratio,3))

###################################################
# target return
###################################################
financing_dollar_cost = max(market_port_kelly_f-1.0,0.0)*float(config["general"]["capital"])*float(config["general"]["financing_cost"])
print "Target portfolio:  Expected return for 1 year: HKD %s" % (intWithCommas(int(sum(target_port_exp_dollar_rtn_list))))
print "Target portfolio:  Expected return for 1 year: HKD %s (after financing costs)" % (intWithCommas(int(sum(target_port_exp_dollar_rtn_list)-financing_dollar_cost)))
# print '\n'.join(map(lambda x: justify_str(x[0],7) + ":  HKD " + justify_str(intWithCommas(int(x[1])),8), filter(lambda y: abs(y[1]) > 1 , sorted(zip(symbol_list,target_port_exp_dollar_rtn_list), key=lambda tup: tup[1], reverse=True))))
print "Financing dollar cost: HKD %s" % (intWithCommas(int(financing_dollar_cost)))

###################################################
# current positions
###################################################
current_pos_list = read_file(config["general"]["current_positions"])
cur_px_dict = dict(map(lambda x: (x[0],float(x[1])), read_file(config["general"]["current_prices"])))
current_mkt_val_dict = {}
if len(current_pos_list) > 0:
    current_mkt_val_dict = dict(map(lambda x: (x[0],conv_to_hkd(x[1],cur_px_dict[x[0]]*float(x[2]))), current_pos_list))
    current_weight_dict = dict(map(lambda s: (s,current_mkt_val_dict[s]/sum(current_mkt_val_dict.values())), current_mkt_val_dict.keys()))
    current_weight_list = map(lambda s: current_weight_dict.get(s,0.0), symbol_list)

    cur_pos_vec = np.asarray(current_weight_list)
    cur_pos_vec_T = np.matrix(cur_pos_vec).T
    cur_port_exp_rtn = float(np.asarray(expected_rtn_list) * cur_pos_vec_T)
    cur_port_stdev = math.sqrt(float((cur_pos_vec * cov_matrix) * cur_pos_vec_T))
    cur_port_sharpe_ratio = float(cur_port_exp_rtn / cur_port_stdev) if abs(cur_port_stdev) > 0.0001 else 0.0
###################################################

###################################################
sym_sol_list = filter(lambda x: abs(x[1]) > 0.001, zip(symbol_list,sol_list))
sym_sol_list.extend(map(lambda x: (x,0.0), filter(lambda k: k not in map(lambda y: y[0], sym_sol_list), current_mkt_val_dict.keys())))
sym_sol_list = sorted(list(set(sym_sol_list)), reverse=True, key=lambda tup: tup[1])
###################################################

###################################################
# stat about current portfolio
###################################################
if len(current_pos_list) > 0:
    current_port_exp_dollar_rtn_list = map(lambda s: (s,int(expected_rtn_dict[s] * current_mkt_val_dict[s])), current_mkt_val_dict.keys())
    print "Current portfolio: E[r] = %s stdev = %s Sharpe ratio = %s Kelly f* = %s" % (str(round(cur_port_exp_rtn*100, 3)) + " %", str(round(cur_port_stdev*100,3)) + " %", round(cur_port_sharpe_ratio,3), round(cur_port_exp_rtn/cur_port_stdev/cur_port_stdev,3))

print "Target portfolio:  Market value: HKD %s" % (justify_str(intWithCommas(int(float(config["general"]["capital"])*market_port_kelly_f)),11))
if len(current_pos_list) > 0:
    print "Current portfolio: Market value: HKD %s" % (justify_str(intWithCommas(int(sum(current_mkt_val_dict.values()))),11))
    print "Current portfolio: Expected return for 1 year: HKD %s" % (intWithCommas(int(sum(map(lambda x: x[1], current_port_exp_dollar_rtn_list)))))
    # print '\n'.join(map(lambda x: justify_str(x[0],7) + ":  HKD " + justify_str(intWithCommas(x[1]),8), sorted(current_port_exp_dollar_rtn_list, key=lambda tup: tup[1], reverse=True)))

###################################################
# solution
###################################################
header = "   Symbol:      Price         E[r] %           %     Amount (HKD)  |      Current  |         Diff"
columns = []
columns.append(map(lambda x: justify_str(x[0],9), sym_sol_list))
columns.append(map(lambda x: ": ", sym_sol_list))
columns.append(map(lambda x: justify_str(cur_px_dict.get(x[0],"---"),10), sym_sol_list))
columns.append(map(lambda x: "   ", sym_sol_list))
columns.append(map(lambda x: justify_str(round(expected_rtn_dict[x[0]]*100,2),10), sym_sol_list))
columns.append(map(lambda x: " %   ", sym_sol_list))
columns.append(map(lambda x: justify_str(round(x[1]*100,1),7), sym_sol_list))
columns.append(map(lambda x: " %     $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(x[1] * float(config["general"]["capital"]))),10), sym_sol_list))
columns.append(map(lambda x: "  | $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(current_mkt_val_dict.get(x[0],0))),10), sym_sol_list))
columns.append(map(lambda x: "  | $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(x[1] * float(config["general"]["capital"]) - current_mkt_val_dict.get(x[0],0))),10), sym_sol_list))
print
print "Target portfolio:"

targetportdetails_list=[header]+map(lambda x: ''.join(x), zip(columns[0],columns[1],columns[2],columns[3],columns[4],columns[5],columns[6],columns[7],columns[8],columns[9],columns[10],columns[11],columns[12]))
print '\n'.join(map(lambda x: justify_str(x[0],5)+")"+x[1], enumerate(targetportdetails_list)))

