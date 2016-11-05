#!/usr/bin/env python

import math
import numpy as np
import cvxopt
from cvxopt import blas, solvers
from configobj import ConfigObj
from datetime import datetime

def read_file(file_loc):
    with open(file_loc,'r') as f:
        return [line.split(',') for line in f]

def calc_return_list(price_list):
    return map(lambda x: (x[0]/x[1])-1, zip(price_list[1:], price_list[:-1]))

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

def calc_cov_matrix_annualized(sym_time_series_list):
    ###################################################
    # correlation matrix
    ###################################################
    correl_matrix = map(lambda ts_x: map(lambda ts_y: calc_correl(ts_x,ts_y), sym_time_series_list), sym_time_series_list)
    correl_matrix = np.asmatrix(correl_matrix)

    print
    print "correl_matrix"
    np.set_printoptions(precision=3)
    print np.asmatrix(correl_matrix)
    ###################################################

    # covariance matrix
    ###################################################
    sd_list = np.asarray(map(lambda ts: calc_sd(map(lambda x: x[1], ts)), sym_time_series_list))
    annualization_factor_list = (map(lambda ts: get_annualization_factor(map(lambda x: x[0], ts)), sym_time_series_list))
    annualized_sd_list = map(lambda x: x[0]*math.sqrt(x[1]), zip(sd_list,annualization_factor_list))
    print
    print "annualized_sd_list"
    print map(lambda x: round(x,3), annualized_sd_list)
    D = np.diag(annualized_sd_list)
    return (D * correl_matrix * D)

def calc_mean_vec(sym_time_series_list):
    return map(lambda ts: sum(map(lambda x: x[1], ts))/len(ts), sym_time_series_list)

def markowitz(px_return_list,cov_matrix,mu_p):
    def iif(cond, iftrue=1.0, iffalse=0.0):
        if cond:
            return iftrue
        else:
            return iffalse

    solvers.options['show_progress'] = False

    n = len(sym_time_series_list)

    P = cvxopt.matrix(cov_matrix)
    q = cvxopt.matrix([0.0 for i in range(n)])

    # G and h determine the inequality constraints in the
    # form $G x \leq h$. We write $w_i \geq 0$ as $-1 \times x_i \leq 0$
    # and also add a (superfluous) $x_i \leq 1$ constraint
    G = cvxopt.matrix([[ (-1.0)**(1+j%2) * iif(i == j/2) for i in range(n) ]
                for j in range(2*n)
                ]).trans()
    h = cvxopt.matrix([ iif(j % 2) for j in range(2*n) ])
    # A and b determine the equality constraints defined as A x = b
    A = cvxopt.matrix([[ 1.0 for i in range(n) ],
                px_return_list
                ]).trans()
    b = cvxopt.matrix([ 1.0, float(mu_p) ])

    sol = solvers.qp(P, q, G, h, A, b)

    # if sol['status'] != 'optimal':
    #     raise Exception("Could not solve problem.")

    w = list(sol['x'])
    f = 2.0*sol['primal objective']

    return {'w': w, 'f': f, 'args': (P, q, G, h, A, b), 'result': sol }



# ###################################################
# np.random.seed(123)
# n_assets = 4
# n_obs = 1000
# return_vec = np.random.randn(n_assets, n_obs)

###################################################
config = ConfigObj('config.ini')

symbol_list = sorted(config["data_path"].keys())
print "Symbols: %s" % (','.join(symbol_list))

###################################################
# read time series of prices
###################################################
sym_data_list = map(lambda s: read_file(config["data_path"][s]), symbol_list)
sym_data_list = map(lambda x: filter(lambda y: len(y) > 5, x), sym_data_list)
sym_time_series_list = map(lambda data_list: map(lambda csv_fields: (datetime.strptime(csv_fields[0],"%Y-%m-%d"),float(csv_fields[5])), data_list), sym_data_list)

px_return_list = map(lambda s: float(config["expect_returns"][s]), symbol_list)
cov_matrix = calc_cov_matrix_annualized(sym_time_series_list)

mu_p = config["expect_returns"]["target"]

print
print "solution"
sol_list = map(lambda x: x, list(markowitz(px_return_list, cov_matrix, mu_p)["result"]['x']))
print "\n".join(map(lambda x: str(x[0]) + ": " + str(round(x[1]*100,1)) + " %", sorted(zip(symbol_list,sol_list), reverse=True, key=lambda tup: tup[1])))
