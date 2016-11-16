#!/usr/bin/env python

import sys
import math
from configobj import ConfigObj
from datetime import datetime, timedelta

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [map(lambda x: x.strip(), line.split(',')) for line in f]

def calc_req_rate_of_return(g,pe):
    g=float(g)
    pe=float(pe)
    return (2.0+g-pe+math.sqrt(math.pow(pe-2.0-g,2.0)+4.0*pe))/2.0/pe

def est_expected_rtn(req_rate,divd_yield,stamp_duty_rate,fund_expense_ratio):
    return min((0.8*req_rate+0.2*divd_yield),req_rate)-stamp_duty_rate-fund_expense_ratio

config = ConfigObj('config.ini')
cur_px_dict = dict(map(lambda x: (x[0],float(x[1])), read_file(config["general"]["current_prices"])))
eps_dict = dict([(k,float(v)) for k,v in config["eps"].items()])
growth_rate_dict = dict([(k,float(v)) for k,v in config["growth_rate"].items()])
divd_per_share_dict = dict([(k,float(v)) for k,v in config["annual_divd_per_share"].items()])
divd_withholdg_tax_dict = dict([(k,float(v)) for k,v in config["divd_withholding_tax_rate"].items()])
stamp_duty_rate_dict = dict([(k,float(v)) for k,v in config["stamp_duty_rate"].items()])
fund_expense_ratio_dict = dict([(k,float(v)) for k,v in config["fund_expense_ratio"].items()])

sym_with_divd_yield = divd_per_share_dict.keys()
sym_divd_yield_dict = dict(map(lambda s: (s,divd_per_share_dict[s] * (1-divd_withholdg_tax_dict[s])/cur_px_dict[s]), sym_with_divd_yield))

print '\n'.join(["%s_divd_yield,%s" % (k,v) for k,v in sym_divd_yield_dict.items()])

sym_with_req_rate_rtn = eps_dict.keys()
sym_req_rate_rtn_dict = dict(map(lambda s: (s,est_expected_rtn(calc_req_rate_of_return(growth_rate_dict.get(s,0),cur_px_dict[s]/eps_dict[s]),sym_divd_yield_dict[s],stamp_duty_rate_dict[s],fund_expense_ratio_dict[s])), sym_with_req_rate_rtn))

symbol_list = list(set(sym_req_rate_rtn_dict.keys() + sym_divd_yield_dict.keys()))

sym_exp_rtn_list = []
for s in symbol_list:
    if s in sym_req_rate_rtn_dict:
        sym_exp_rtn_list.append((s,sym_req_rate_rtn_dict[s]))
    elif s in sym_divd_yield_dict:
        sym_exp_rtn_list.append((s,sym_divd_yield_dict[s]))

print '\n'.join(map(lambda x: x[0]+','+str(x[1]), sorted(sym_exp_rtn_list,key=lambda tup: tup[1],reverse=True)))
