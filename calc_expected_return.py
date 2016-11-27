#!/usr/bin/env python
import sys
import math
from configobj import ConfigObj
from datetime import datetime, timedelta

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [map(lambda x: x.strip(), line.split(',')) for line in f]

def justify_str(s,totlen,left_right,padchar):
    def extra(s,totlen):
        return ''.join(map(lambda x: padchar, range(totlen - len(s))))
    s = str(s)
    if left_right == "left":
        return s + extra(s,totlen)
    elif left_right == "right":
        return extra(s,totlen) + s
    else:
        return s

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
divd_withholdg_tax_rate_dict = dict([(k,float(v)) for k,v in config["divd_withholding_tax_rate"].items()])
stamp_duty_rate_dict = dict([(k,float(v)) for k,v in config["stamp_duty_rate"].items()])
fund_expense_ratio_dict = dict([(k,float(v)) for k,v in config["fund_expense_ratio"].items()])

traded_symbol_list = sorted([i for k in map(lambda x: config["general"][x].split(','), filter(lambda x: "traded_symbols" in x, config["general"].keys())) for i in k])
sym_with_divd_yield = filter(lambda x: x in traded_symbol_list, divd_per_share_dict.keys())
sym_divd_yield_dict = dict(map(lambda s: (s,((divd_per_share_dict[s] * (1-divd_withholdg_tax_rate_dict.get(s,0.0))/cur_px_dict[s]) if divd_per_share_dict[s] > 0.0 else -fund_expense_ratio_dict.get(s,0.0))), sym_with_divd_yield))

sym_with_req_rate_rtn = filter(lambda x: x in traded_symbol_list, eps_dict.keys())
sym_req_rate_rtn_dict = dict(map(lambda s: (s,est_expected_rtn(calc_req_rate_of_return(growth_rate_dict.get(s,0),cur_px_dict[s]/eps_dict[s]),sym_divd_yield_dict[s],stamp_duty_rate_dict.get(s,0.0),fund_expense_ratio_dict.get(s,0.0))), sym_with_req_rate_rtn))

symbol_list = sorted(list(set(sym_req_rate_rtn_dict.keys() + sym_divd_yield_dict.keys())))

sys.stdout.write("    symbol")
sys.stdout.write("      divd")
sys.stdout.write("  req rate")
sys.stdout.write("\n")
print '\n'.join(map(lambda s: justify_str(s,10,"right",' ')+justify_str(round(sym_divd_yield_dict.get(s,0)*100,2),10,"right",' ')+justify_str(round(sym_req_rate_rtn_dict.get(s,0)*100,2),10,"right",' '), sorted(symbol_list)))

sym_exp_rtn_list = []
for s in symbol_list:
    if s in sym_req_rate_rtn_dict:
        sym_exp_rtn_list.append((s,sym_req_rate_rtn_dict[s]))
    elif s in sym_divd_yield_dict:
        sym_exp_rtn_list.append((s,sym_divd_yield_dict[s]))

outfile = open(config["general"]["expected_return_file"], "w")
outfile.write('\n'.join(map(lambda x: x[0]+','+str(x[1]), sorted(sym_exp_rtn_list,key=lambda tup: tup[1],reverse=True))))
outfile.close()
