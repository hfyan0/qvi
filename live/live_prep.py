#!/usr/bin/env python
from configobj import ConfigObj
import sys
import math
from datetime import datetime, timedelta
import numpy as np
import cPickle
import multiprocessing as mp
# from pathos.multiprocessing import ProcessingPool as Pool

import os
sys.path.append(os.path.dirname(sys.path[0]))
from qvi import CurrencyConverter,calc_cov_matrix_annualized,intWithCommas,justify_str,\
                markowitz_sharpe,log_optimal_growth,read_file,read_file_mul,\
                calc_expected_return_before_201703,\
                calc_irr_mean_cov_after_20170309_prep,\
                get_hist_data_key_sym,get_industry_groups,preprocess_industry_groups

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
num_of_jobs = int(mp.cpu_count()*float(dict(map(lambda x: x.split(':'), config_common["general"]["percentage_of_cpu_cores_to_use"]))[local_ip]))
###################################################

print "Start reading data... %s" % (datetime.now())
symbol_list = sorted([ i for k in map(lambda x: config["general"][x] if isinstance(config["general"][x], (list, tuple)) else [config["general"][x]], filter(lambda x: "traded_symbols" in x, config["general"].keys())) for i in k ])
hedging_symbol_list = config["general"]["hedging_symbols"]


###################################################
# read time series of prices
###################################################
print "Start reading symbol day bar... %s" % (datetime.now())
sym_data_list = map(lambda s: read_file(config["data_path"].get(s,config["data_path"]["default"]+s+".csv")), hedging_symbol_list+symbol_list)
# sym_data_dict = dict(Pool(4).map(read_file_mul, map(lambda s: (str(config["data_path"].get(s,config["data_path"]["default"]+s+".csv")),s), hedging_symbol_list+symbol_list)))
# sym_data_list = map(lambda s: sym_data_dict[s], hedging_symbol_list+symbol_list)
print "Finished reading symbol day bar... %s" % (datetime.now())

print "Start processing symbol day bar... %s" % (datetime.now())
sym_data_list = map(lambda x: filter(lambda y: len(y) > 5, x), sym_data_list)
sym_time_series_list = map(lambda data_list: map(lambda csv_fields: (datetime.strptime(csv_fields[0],"%Y-%m-%d"),float(csv_fields[5])), data_list), sym_data_list)
insufficient_data_list = map(lambda ss: ss[0]+": "+str(len(ss[1])), filter(lambda x: len(x[1])<50, zip(hedging_symbol_list+symbol_list,sym_time_series_list)))
if len(insufficient_data_list) > 0:
    print "Insufficient data:"
    print '\n'.join(insufficient_data_list)
print "Finished processing symbol day bar... %s" % (datetime.now())

print "Start calculating cov matrix... %s" % (datetime.now())
aug_cov_matrix,annualized_sd_list,annualized_adj_sd_list = calc_cov_matrix_annualized(sym_time_series_list, num_of_jobs)
print "Abnormal stdev to check:"
print '\n'.join(map(lambda ss: ": ".join(map(str, ss)) + " %", filter(lambda x: (x[1] > 50.0) or (x[1] < 10.0), zip(hedging_symbol_list+symbol_list,map(lambda x: round(x*100.0,2), annualized_sd_list)))))
cov_matrix = aug_cov_matrix
for i in range(len(hedging_symbol_list)):
    cov_matrix = np.delete(cov_matrix, 0, 0)
    cov_matrix = np.delete(cov_matrix, 0, 1)
print "Finished calculating cov matrix... %s" % (datetime.now())

with open(prep_data_folder+"/aug_cov_matrix.pkl", "w+") as aug_cov_matrix_file:
    cPickle.dump(aug_cov_matrix,aug_cov_matrix_file)
with open(prep_data_folder+"/cov_matrix.pkl", "w+") as cov_matrix_file:
    cPickle.dump(cov_matrix,cov_matrix_file)


###################################################
print "Start reading fundamental data... %s" % (datetime.now())
hist_bps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_bps"])
hist_eps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_eps"])
hist_roa_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_roa"])
hist_totliabps_dict = get_hist_data_key_sym(config_common["hist_data"]["hist_totliabps"])

with open(prep_data_folder+"/hist_bps_dict.pkl", "w+") as hist_bps_dict_file:
    cPickle.dump(hist_bps_dict,hist_bps_dict_file)
with open(prep_data_folder+"/hist_eps_dict.pkl", "w+") as hist_eps_dict_file:
    cPickle.dump(hist_eps_dict,hist_eps_dict_file)
with open(prep_data_folder+"/hist_roa_dict.pkl", "w+") as hist_roa_dict_file:
    cPickle.dump(hist_roa_dict,hist_roa_dict_file)
with open(prep_data_folder+"/hist_totliabps_dict.pkl", "w+") as hist_totliabps_dict_file:
    cPickle.dump(hist_totliabps_dict,hist_totliabps_dict_file)
print "Finished reading fundamental data... %s" % (datetime.now())

print "Start calc_irr_mean_cov_after_20170309_prep... %s" % (datetime.now())
calc_irr_mean_cov_after_20170309_prep(config_common,prep_data_folder,datetime.now().date(),symbol_list,hist_bps_dict,hist_totliabps_dict,hist_eps_dict,hist_roa_dict,int(config["general"]["monte_carlo_num_of_times"]),int(config["general"]["num_of_fut_divd_periods"]),0,True)
print "Finished calc_irr_mean_cov_after_20170309_prep... %s" % (datetime.now())


