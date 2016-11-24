#!/usr/bin/env python
import sys
import math
import numpy as np
import cvxopt
from cvxopt import blas, solvers
from configobj import ConfigObj
from datetime import datetime, timedelta

def conv_to_hkd(currency,amt):
    if currency == "USD":
        return 7.8 * float(amt)
    return float(amt)

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [map(lambda x: x.strip(), line.split(',')) for line in f]

def justify_str(s,totlen,left_right="right",padchar=' '):
    def extra(s,totlen):
        return ''.join(map(lambda x: padchar, range(totlen - len(s))))
    s = str(s)
    if left_right == "left":
        return s + extra(s,totlen)
    elif left_right == "right":
        return extra(s,totlen) + s
    else:
        return s

def intWithCommas(x):
    if type(x) not in [type(0), type(0L)]:
        raise TypeError("Parameter must be an integer.")
    if x < 0:
        return '-' + intWithCommas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)

def calc_return_list(price_list):
    return map(lambda x: (x[0]/x[1])-1.0, zip(price_list[1:], price_list[:-1]))

def calc_correl(ts_a,ts_b):
    common_date_set = set(map(lambda x: x[0], ts_a)).intersection(set(map(lambda x: x[0], ts_b)))
    a_ext = map(lambda x: x[1], filter(lambda x: x[0] in common_date_set, ts_a))
    b_ext = map(lambda x: x[1], filter(lambda x: x[0] in common_date_set, ts_b))
    return round(np.corrcoef(calc_return_list(a_ext),calc_return_list(b_ext))[0][1],5)

def calc_var(ts):
    r_ls = calc_return_list(ts)
    n = len(r_ls)
    m = float(sum(r_ls))/n
    return sum(map(lambda x: math.pow(x-m,2), r_ls)) / float(n)

def calc_sd(ts):
    return math.sqrt(calc_var(ts))

def get_annualization_factor(date_list):
    min_diff = min(map(lambda x: (x[0]-x[1]).days, zip(date_list[1:],date_list[:-1])))
    if min_diff == 1:
        return 252
    elif min_diff == 7:
        return 52
    else:
        return 12

def calc_cov_matrix_annualized(sym_time_series_list, specific_riskiness_list):
    ###################################################
    # correlation matrix
    ###################################################
    correl_matrix = map(lambda ts_x: map(lambda ts_y: calc_correl(ts_x,ts_y), sym_time_series_list), sym_time_series_list)
    correl_matrix = np.asmatrix(correl_matrix)

    np.set_printoptions(precision=5,suppress=True)
    if len(correl_matrix.tolist()) < 6:
        print
        print "correl_matrix"
        for row in correl_matrix.tolist():
            for element in row:
                str_element = str(round(element,3))
                sys.stdout.write(justify_str(str_element,8))
            sys.stdout.write('\n')

    ###################################################
    # covariance matrix
    ###################################################
    sd_list = np.asarray(map(lambda ts: calc_sd(map(lambda x: x[1], ts)), sym_time_series_list))
    annualization_factor_list = (map(lambda ts: get_annualization_factor(map(lambda x: x[0], ts)), sym_time_series_list))
    annualized_sd_list = map(lambda x: x[0]*math.sqrt(x[1]), zip(sd_list,annualization_factor_list))
    annualized_adj_sd_list = map(lambda x: x[0]+x[1], zip(annualized_sd_list,specific_riskiness_list))
    D = np.diag(annualized_adj_sd_list)
    return ((D * correl_matrix * D),annualized_sd_list,annualized_adj_sd_list)

def calc_mean_vec(sym_time_series_list):
    return map(lambda ts: sum(map(lambda x: x[1], ts))/len(ts), sym_time_series_list)

def markowitz(symbol_list,expected_rtn_list,cov_matrix,mu_p,max_weight_list):
    def iif(cond, iftrue=1.0, iffalse=0.0):
        if cond:
            return iftrue
        else:
            return iffalse

    n = len(symbol_list)

    P = cvxopt.matrix(cov_matrix)
    q = cvxopt.matrix([0.0 for i in range(n)])

    # G and h determine the inequality constraints in the
    # form $G x \leq h$. We write $w_i \geq 0$ as $-1 \times x_i \leq 0$
    # and also add a (superfluous) $x_i \leq 1$ constraint
    G = cvxopt.matrix([[ (-1.0)**(1+j%2) * iif(i == j/2) for i in range(n) ]
                for j in range(2*n)
                ]).trans()
    h = cvxopt.matrix([ max_weight_list[j/2] * iif(j % 2) for j in range(2*n) ])
    # A and b determine the equality constraints defined as A x = b
    A = cvxopt.matrix([[ 1.0 for i in range(n) ],
                expected_rtn_list]).trans()
    b = cvxopt.matrix([ 1.0, float(mu_p) ])

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        return None

    w = list(sol['x'])
    f = 2.0*sol['primal objective']

    return {'w': w, 'f': f, 'args': (P, q, G, h, A, b), 'result': sol }



###################################################
config = ConfigObj('config.ini')

symbol_list = sorted([i for k in map(lambda x: config["general"][x].split(','), filter(lambda x: "traded_symbols" in x, config["general"].keys())) for i in k])
print "Symbols: %s" % (','.join(symbol_list))

specific_riskiness_list = map(lambda s: float(config["specific_riskiness"].get(s,0)), symbol_list)
print "Riskiness: %s" % (','.join(map(str, specific_riskiness_list)))

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
print "symbol,E[r],SD[r],SD_adj[r],Sharpe"
print '\n'.join(map(lambda x: justify_str(x[0],8)+justify_str(x[1],10)+justify_str(x[2],10)+justify_str(x[3],10)+justify_str(x[4],10), zip(symbol_list,str_expected_rtn_list,str_annualized_sd_list,str_annualized_adj_sd_list,str_sharpe_list)))

sorted_expected_rtn_list = sorted(expected_rtn_list)
from_tgt_rtn = sorted_expected_rtn_list[0]
to_tgt_rtn = sorted_expected_rtn_list[-1]
mu_sd_sharpe_soln_list = []
N = 1000

max_weight_dict = config["max_weight"]
max_weight_list = map(lambda x: float(max_weight_dict[x]), symbol_list)

for i in range(N):

    mu_p = from_tgt_rtn + (to_tgt_rtn - from_tgt_rtn) * float(i)/float(N)
    # mu_p = to_tgt_rtn * float(i)/float(N)
    sol_list = markowitz(symbol_list, expected_rtn_list, cov_matrix, mu_p, max_weight_list)

    if sol_list is None:
        continue

    sol_list = list(sol_list["result"]['x'])

    sol_vec = np.asarray(sol_list)
    sol_vec_T = np.matrix(sol_vec).T

    target_port_exp_rtn = float(np.asarray(expected_rtn_list) * sol_vec_T)
    target_port_stdev = math.sqrt(float((sol_vec * cov_matrix) * sol_vec_T))
    target_port_sharpe_ratio = float(target_port_exp_rtn / target_port_stdev)

    if (len(mu_sd_sharpe_soln_list) == 0) or (target_port_sharpe_ratio > mu_sd_sharpe_soln_list[2]):
        mu_sd_sharpe_soln_list = []
        mu_sd_sharpe_soln_list.append(float(target_port_exp_rtn))
        mu_sd_sharpe_soln_list.append(float(target_port_stdev))
        mu_sd_sharpe_soln_list.append(float(target_port_sharpe_ratio))
        mu_sd_sharpe_soln_list.append(sol_list)

target_port_exp_rtn, target_port_stdev, target_port_sharpe_ratio, sol_list = tuple(mu_sd_sharpe_soln_list)

print

###################################################
# Kelly's criterion
###################################################
kelly_f = float(target_port_exp_rtn / target_port_stdev / target_port_stdev)
print "Kelly f* = %s" % (kelly_f)
kelly_f = min(kelly_f,1.0)
print "Kelly f* (unleveraged) = %s" % (kelly_f)

print
print "Market portfolio:  E[r] = %s stdev = %s Sharpe ratio = %s" % (str(round(target_port_exp_rtn*100, 3)) + " %", str(round(target_port_stdev*100,3)) + " %", round(target_port_sharpe_ratio,3))
print "Target portfolio:  E[r] = %s stdev = %s" % (str(round(target_port_exp_rtn*kelly_f*100, 3)) + " %", str(round(target_port_stdev*kelly_f*100,3)) + " %")

###################################################
# current positions
###################################################
current_pos_list = read_file(config["general"]["current_positions"])
cur_px_dict = dict(map(lambda x: (x[0],float(x[1])), read_file(config["general"]["current_prices"])))
current_mkt_val_dict = {}
if len(current_pos_list) > 0:
    current_mkt_val_dict = dict(map(lambda x: (x[0],conv_to_hkd(x[1],cur_px_dict[x[0]]*float(x[2]))), current_pos_list))
    current_weight_dict = dict(map(lambda s: (s,current_mkt_val_dict[s]/float(config["general"]["capital"])), current_mkt_val_dict.keys()))
    current_weight_list = map(lambda s: current_weight_dict.get(s,0.0), symbol_list)

    cur_pos_vec = np.asarray(current_weight_list)
    cur_pos_vec_T = np.matrix(cur_pos_vec).T
    cur_port_exp_rtn = float(np.asarray(expected_rtn_list) * cur_pos_vec_T)
    cur_port_stdev = math.sqrt(float((cur_pos_vec * cov_matrix) * cur_pos_vec_T))
    cur_port_sharpe_ratio = float(cur_port_exp_rtn / cur_port_stdev)
###################################################

###################################################
sol_list = map(lambda x: x * kelly_f, sol_list)
sym_sol_list = filter(lambda x: abs(x[1]) > 0.0001, zip(symbol_list,sol_list))
sym_sol_list.extend(map(lambda x: (x,0.0), filter(lambda k: k not in map(lambda y: y[0], sym_sol_list), current_mkt_val_dict.keys())))
sym_sol_list = sorted(list(set(sym_sol_list)), reverse=True, key=lambda tup: tup[1])
###################################################

###################################################
# stat about current portfolio
###################################################
if len(current_pos_list) > 0:
    current_port_exp_rtn_list = map(lambda s: (s,int(expected_rtn_dict[s] * current_mkt_val_dict[s])), current_mkt_val_dict.keys())
    print "Current portfolio: E[r] = %s stdev = %s Sharpe ratio = %s" % (str(round(cur_port_exp_rtn*100, 3)) + " %", str(round(cur_port_stdev*100,3)) + " %", round(cur_port_sharpe_ratio,3))

print "Target portfolio:  Market value: HKD %s" % (justify_str(intWithCommas(int(float(config["general"]["capital"])*target_port_sharpe_ratio)),11))
if len(current_pos_list) > 0:
    print "Current portfolio: Market value: HKD %s" % (justify_str(intWithCommas(int(sum(current_mkt_val_dict.values()))),11))
    print "Current portfolio: Expected return for 1 year: HKD %s" % (intWithCommas(int(sum(map(lambda x: x[1], current_port_exp_rtn_list)))))
    print '\n'.join(map(lambda x: justify_str(x[0],7) + ":  HKD " + justify_str(intWithCommas(x[1]),8), sorted(current_port_exp_rtn_list, key=lambda tup: tup[1], reverse=True)))

###################################################
# solution
###################################################
header = "   Symbol:                      %     Amount (HKD)  |      Current  |         Diff"
columns = []
columns.append(map(lambda x: justify_str(x[0],9), sym_sol_list))
columns.append(map(lambda x: ": ", sym_sol_list))
columns.append(map(lambda x: justify_str(cur_px_dict.get(x[0],"---"),10), sym_sol_list))
columns.append(map(lambda x: "   ", sym_sol_list))
columns.append(map(lambda x: justify_str(round(x[1]*100,1),7), sym_sol_list))
columns.append(map(lambda x: " %     $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(x[1] * float(config["general"]["capital"]))),10), sym_sol_list))
columns.append(map(lambda x: "  | $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(current_mkt_val_dict.get(x[0],0))),10), sym_sol_list))
columns.append(map(lambda x: "  | $ ", sym_sol_list))
columns.append(map(lambda x: justify_str(intWithCommas(int(x[1] * float(config["general"]["capital"]) - current_mkt_val_dict.get(x[0],0))),10), sym_sol_list))
print
print "Target portfolio:"
print '\n'.join([header]+map(lambda x: ''.join(x), zip(columns[0],columns[1],columns[2],columns[3],columns[4],columns[5],columns[6],columns[7],columns[8],columns[9],columns[10])))
