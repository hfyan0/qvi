#!/usr/bin/env python
from configobj import ConfigObj
import sys
import math
from datetime import datetime, timedelta
import numpy as np
import cPickle

import os
sys.path.append(os.path.dirname(sys.path[0]))
from qvi import CurrencyConverter,intWithCommas,justify_str,\
                markowitz_sharpe,log_optimal_growth,read_file,\
                calc_expected_return_before_201703,\
                calc_irr_mean_cov_after_20170309_live,\
                get_hist_data_key_sym,get_industry_groups,preprocess_industry_groups

###################################################
time_check = datetime.now()
time_check_printout = ["Time taken:"]

###################################################
config = ConfigObj('config.ini')
config_common = ConfigObj(config["general"]["common_config"])

###################################################
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("gmail.com",80))
local_ip = (s.getsockname()[0])
s.close()
prep_data_folder = dict(map(lambda x: x.split(':'), config["general"]["prep_data_folder"]))[local_ip]
###################################################

print "Start reading data..."
symbol_list = sorted([ i for k in map(lambda x: config["general"][x] if isinstance(config["general"][x], (list, tuple)) else [config["general"][x]], filter(lambda x: "traded_symbols" in x, config["general"].keys())) for i in k ])
hedging_symbol_list = config["general"]["hedging_symbols"]
specific_riskiness_list = len(hedging_symbol_list) * [0.0] + map(lambda s: float(config["specific_riskiness"].get(s,0.0)), symbol_list)

###################################################
with open(prep_data_folder+"/symbol_with_enough_fundl.pkl", "rb") as symbol_with_enough_fundl_file:
    symbol_with_enough_fundl_list = cPickle.load(symbol_with_enough_fundl_file)

cur_px_dict = dict(map(lambda x: (x[0],float(x[1])), read_file(config["general"]["current_prices"])))
hist_unadj_px_dict = {}
hist_unadj_px_dict[datetime.now().date()] = cur_px_dict

###################################################
ind_grp_list_1 = preprocess_industry_groups(config_common["industry_group"])
industry_groups_dict = dict(ind_grp_list_1)
industry_groups_list = get_industry_groups(ind_grp_list_1)
# print '\n'.join(map(lambda x: ':'.join(map(str, x)), industry_groups_dict.items()))



irr_combined_mean_list,irr_combined_cov_list,irr_combined_ci_list = calc_irr_mean_cov_after_20170309_live(config_common,prep_data_folder,datetime.now().date(),symbol_with_enough_fundl_list,hist_unadj_px_dict,True)

if any(map(lambda x: x is None, [symbol_with_enough_fundl_list,irr_combined_mean_list,irr_combined_cov_list,irr_combined_ci_list])):
    print "Problem calculating IRR. Exiting..."
    sys.exit(0)

irr_combined_mean_dict = dict(zip(symbol_with_enough_fundl_list,irr_combined_mean_list))
# print irr_combined_mean_dict
print
print "irr_combined_cov_list:"
print len(irr_combined_cov_list)
print len(irr_combined_cov_list[0])
irr_combined_cov_matrix = np.asmatrix(irr_combined_cov_list)


with open(prep_data_folder+"/aug_cov_matrix.pkl", "r") as aug_cov_matrix_file:
    aug_cov_matrix = cPickle.load(aug_cov_matrix_file)
with open(prep_data_folder+"/cov_matrix.pkl", "r") as cov_matrix_file:
    cov_matrix = cPickle.load(cov_matrix_file)



###################################################
with open(prep_data_folder+"/hist_bps_dict.pkl", "r") as hist_bps_dict_file:
    hist_bps_dict = cPickle.load(hist_bps_dict_file)




###################################################
# EPS override
###################################################
sym_with_missed_next_earnings = filter(lambda x: x in symbol_list, config["eps_override"]["miss_next_earnings"])
for i,sym in enumerate(symbol_list):
    if sym in config["eps_override"]:
        r = float(config["eps_override"][sym]) / cur_px_dict[sym]
        irr_combined_mean_list[i] = r
        irr_combined_mean_dict[sym] = r
    if sym in sym_with_missed_next_earnings:
        b = hist_bps_dict[sym][-1][1]
        p = cur_px_dict[sym]
        e = float(config["eps_override"][sym])
        r1 = None
        r2 = None
        try:
            r1 = (2*b - p + math.sqrt(math.pow(2*b-p,2)-4*p*e))/2/p
            r2 = (2*b - p - math.sqrt(math.pow(2*b-p,2)-4*p*e))/2/p
        except:
            pass
        r = min(filter(lambda x: x is not None, [r1,r2]))
        irr_combined_mean_list[i] = r
        irr_combined_mean_dict[sym] = r

time_check_printout.append("Calculating expected return: %s" % (datetime.now()-time_check))
print "Expected return:"
time_check = datetime.now()
print '\n'.join(map(lambda x: justify_str(x[0],5)+": "+justify_str(round(x[1]*100,2),8)+" %", sorted(irr_combined_mean_dict.items(),key=lambda x: x[1],reverse=True)))


curcy_converter = CurrencyConverter(config_common["currency_rate"])
###################################################
# current positions
###################################################
current_pos_list = read_file(config["general"]["current_positions"])
current_mkt_val_dict = {}
if len(current_pos_list) > 0:
    current_mkt_val_dict = dict(map(lambda x: (x[0],curcy_converter.conv_to_hkd(x[1],datetime.now().date(),cur_px_dict[x[0]]*float(x[2]))), current_pos_list))
    current_weight_dict = dict(map(lambda s: (s,current_mkt_val_dict[s]/sum(current_mkt_val_dict.values())), current_mkt_val_dict.keys()))
    current_weight_list = map(lambda s: current_weight_dict.get(s,0.0), symbol_list)

    cur_pos_vec = np.asarray(current_weight_list)
    cur_pos_vec_T = np.matrix(cur_pos_vec).T
###################################################

max_weight_dict = config["max_weight"]
max_weight_list = map(lambda x: float(max_weight_dict.get(x,max_weight_dict["single_name"])), symbol_list)

N = int(config["general"]["granularity"])

###################################################
print "Starting portfolio optimization..."
time_check = datetime.now()
###################################################
markowitz_max_sharpe_sol_list = []
if float(config["general"]["markowitz_max_sharpe_weight"] > 0.0):
    tmp_sol_list = markowitz_sharpe(symbol_with_enough_fundl_list, irr_combined_mean_list, irr_combined_cov_matrix, max_weight_list, float(config["general"]["min_expected_return"]), industry_groups_list, float(config["max_weight"]["industry"]), float(config["general"]["portfolio_change_inertia"]), float(config["general"]["hatred_for_small_size"]), current_weight_list)
    if tmp_sol_list is None:
        print "Failed to find solution."
        sys.exit(0)
    markowitz_max_sharpe_sol_list = list(tmp_sol_list["result"]['x'])
###################################################

###################################################
log_optimal_sol_list = []
if float(config["general"]["log_optimal_growth_weight"]) > 0.0:
    tmp_sol_list = log_optimal_growth(symbol_with_enough_fundl_list, irr_combined_mean_list, irr_combined_cov_matrix, max_weight_list, industry_groups_list, float(config["max_weight"]["industry"]), float(config["general"]["portfolio_change_inertia"]), float(config["general"]["hatred_for_small_size"]), current_weight_list)
    if tmp_sol_list is None:
        print "Failed to find solution."
        sys.exit(0)
    log_optimal_sol_list = list(tmp_sol_list["result"]['x'])
###################################################

###################################################
sol_list = len(symbol_list) * [0.0]
if float(config["general"]["log_optimal_growth_weight"]) > 0.0:
    sol_list = map(sum, zip(sol_list,map(lambda x: x * float(config["general"]["log_optimal_growth_weight"]), log_optimal_sol_list)))
if float(config["general"]["markowitz_max_sharpe_weight"]) > 0.0:
    sol_list = map(sum, zip(sol_list,map(lambda x: x * float(config["general"]["markowitz_max_sharpe_weight"]), markowitz_max_sharpe_sol_list)))
###################################################

sol_vec = np.asarray(sol_list)
sol_vec_T = np.matrix(sol_vec).T
target_port_exp_rtn = float(np.asarray(irr_combined_mean_list) * sol_vec_T)
target_port_stdev = math.sqrt(float((sol_vec * irr_combined_cov_matrix) * sol_vec_T))
target_port_sharpe_ratio = float(target_port_exp_rtn / target_port_stdev)
target_port_kelly_f = float(target_port_exp_rtn / target_port_stdev / target_port_stdev)
target_port_exp_dollar_rtn_list = map(lambda x: x[0]*x[1]*float(config["general"]["capital"]), zip(sol_list,irr_combined_mean_list))


###################################################
print
print 100 * "-"
print "Target  portfolio: E[r] = %s stdev = %s Sharpe ratio = %s Kelly f* = %s" % (justify_str(round(target_port_exp_rtn*100, 3),7) + " %", justify_str(round(target_port_stdev*100,3),7) + " %", justify_str(round(target_port_sharpe_ratio,3),5), justify_str(round(target_port_kelly_f,3),7))

###################################################
sym_sol_list = filter(lambda x: abs(x[1]) > 0.001, zip(symbol_list,sol_list))
sym_sol_list.extend(map(lambda x: (x,0.0), filter(lambda k: k not in map(lambda y: y[0], sym_sol_list), current_mkt_val_dict.keys())))
###################################################

###################################################
# target return
###################################################
print "Target  portfolio: Expected return for 1 year: HKD %s" % (intWithCommas(int(sum(target_port_exp_dollar_rtn_list))))
# print '\n'.join(map(lambda x: justify_str(x[0],7) + ":  HKD " + justify_str(intWithCommas(int(x[1])),8), filter(lambda y: abs(y[1]) > 1 , sorted(zip(symbol_list,target_port_exp_dollar_rtn_list), key=lambda tup: tup[1], reverse=True))))

print "Target  portfolio: Market value: HKD %s" % (justify_str(intWithCommas(int(float(config["general"]["capital"]))),11))
if len(current_pos_list) > 0:
    current_port_mkt_val = sum(current_mkt_val_dict.values())
    print "Current portfolio: Market value: HKD %s" % (justify_str(intWithCommas(int(current_port_mkt_val)),11))

###################################################
# beta
###################################################
beta_dict_list = map(lambda h: dict(map(lambda x: (x[1],x[0]/aug_cov_matrix.tolist()[h[0]][h[0]]), zip(aug_cov_matrix.tolist()[h[0]],hedging_symbol_list+symbol_list))), enumerate(hedging_symbol_list))
# print "Beta:"
# print '\n'.join(map(lambda x: x[0]+": "+str(round(x[1],3)), sorted(beta_dict_list[0].items())))

aug_sol_list = len(hedging_symbol_list)*[0.0]+sol_list
sol_port_beta_list = map(lambda h: sum(map(lambda x: x[0]*x[1]/aug_cov_matrix.tolist()[h[0]][h[0]], zip(aug_cov_matrix.tolist()[h[0]],aug_sol_list))), enumerate(hedging_symbol_list))
print "Target  portfolio: Beta: " + '  '.join(map(lambda x: hedging_symbol_list[x[0]]+": "+justify_str(str(round(x[1],3)),5)+" (Beta hedge notional: "+justify_str(intWithCommas(int(current_port_mkt_val*x[1])),9)+" )", enumerate(sol_port_beta_list)))

aug_current_weight_list = len(hedging_symbol_list)*[0.0]+current_weight_list
current_port_beta_list = map(lambda h: sum(map(lambda x: x[0]*x[1]/aug_cov_matrix.tolist()[h[0]][h[0]], zip(aug_cov_matrix.tolist()[h[0]],aug_current_weight_list))), enumerate(hedging_symbol_list))
print "Current portfolio: Beta: " + '  '.join(map(lambda x: hedging_symbol_list[x[0]]+": "+justify_str(str(round(x[1],3)),5)+" (Beta hedge notional: "+justify_str(intWithCommas(int(current_port_mkt_val*x[1])),9)+" )", enumerate(current_port_beta_list)))


hsi_expected_return = float(config["general"]["hsi_expected_return"])
optimal_h_list = map(lambda x: (x[0],x[1]-(hsi_expected_return/0.7)), zip(hedging_symbol_list,current_port_beta_list))
print "Current portfolio: HSI expected return: " + str(round(hsi_expected_return*100.0,2)) + " %   Optimal hedge:  " + '  '.join(map(lambda x: x[0] + ": " + str(round(x[1],5)) + " ( " + justify_str(intWithCommas(int(current_port_mkt_val*x[1])),9) + " ) ", optimal_h_list))


###################################################
# sorting of output list
###################################################
sym_sol_list = list(set(sym_sol_list))
sym_sol_list_with_diff = map(lambda x: (x, int(x[1] * float(config["general"]["capital"]) - current_mkt_val_dict.get(x[0],0)), irr_combined_mean_dict.get(x[0]),0), sym_sol_list)
sym_sol_list = map(lambda x: x[0], sorted(sym_sol_list_with_diff, reverse=True, key=lambda tup: (tup[1],tup[2])))

###################################################
# solution
###################################################
header = "   Symbol:  Indus     Price         E[r] %         Beta     Beta      w %     Amount (HKD)  |      Current  |         Diff  |  Symbol"
columns = []
columns.append(map(lambda x: justify_str(x[0],9), sym_sol_list))
columns.append(map(lambda x: ": ", sym_sol_list))
columns.append(map(lambda x: justify_str(industry_groups_dict.get(x[0],"---"),6), sym_sol_list))
columns.append(map(lambda x: justify_str(cur_px_dict.get(x[0],"---"),10), sym_sol_list))
columns.append(map(lambda x: "   ", sym_sol_list))
columns.append(map(lambda x: justify_str(round(irr_combined_mean_dict.get(x[0],0)*100,2),10), sym_sol_list))
columns.append(map(lambda x: " %    ", sym_sol_list))
columns.append(map(lambda x: justify_str(round(beta_dict_list[0][x[0]],3),9), sym_sol_list))
columns.append(map(lambda x: "", sym_sol_list))
columns.append(map(lambda x: justify_str(round(beta_dict_list[1][x[0]],3),9), sym_sol_list))
columns.append(map(lambda x: "", sym_sol_list))
columns.append(map(lambda x: justify_str(round(x[1]*100,1),7), sym_sol_list))
columns.append(map(lambda x: " %     $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(x[1] * float(config["general"]["capital"]))),10), sym_sol_list))
columns.append(map(lambda x: "  | $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(current_mkt_val_dict.get(x[0],0))),10), sym_sol_list))
columns.append(map(lambda x: "  | $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(x[1] * float(config["general"]["capital"]) - current_mkt_val_dict.get(x[0],0))),10), sym_sol_list))
columns.append(map(lambda x: "  | ", sym_sol_list))
columns.append(map(lambda x: justify_str(x[0],5), sym_sol_list))
print
print "Target portfolio:"

targetportdetails_list=[header]+map(lambda x: ''.join(x), zip(columns[0],columns[1],columns[2],columns[3],columns[4],columns[5],columns[6],columns[7],columns[8],columns[9],columns[10],columns[11],columns[12],columns[13],columns[14],columns[15],columns[16],columns[17],columns[18],columns[19]))
print '\n'.join(map(lambda x: justify_str(x[0],5)+")"+x[1], enumerate(targetportdetails_list)))

###################################################
time_check_printout.append("Portfolio optimization: %s" % (datetime.now()-time_check))
print '\n'.join(time_check_printout)
