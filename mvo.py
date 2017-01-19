#!/usr/bin/env python
import sys
import math
import numpy as np
import cvxopt
from cvxopt import blas, solvers
from datetime import datetime, timedelta
import scipy.optimize

###################################################
# Most recent (for correl)
###################################################
LOOKBACK_DAYS = 378
###################################################

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
    a_ext = map(lambda x: x[1], filter(lambda x: x[0] in common_date_set, ts_a))[-LOOKBACK_DAYS:]
    b_ext = map(lambda x: x[1], filter(lambda x: x[0] in common_date_set, ts_b))[-LOOKBACK_DAYS:]
    if len(a_ext) < 30 or len(b_ext) < 30:
        return -1
    else:
        return round(np.corrcoef(calc_return_list(a_ext),calc_return_list(b_ext))[0][1],5)

def calc_var(ts):
    r_ls = calc_return_list(ts)[-LOOKBACK_DAYS:]
    n = len(r_ls)
    m = float(sum(r_ls))/n
    return sum(map(lambda x: math.pow(x-m,2), r_ls)) / float(n)

def calc_sd(ts):
    if len(ts) < 30:
        return 9999.9
    else:
        return math.sqrt(calc_var(ts))

def get_annualization_factor(date_list):
    if len(date_list) < 30:
        return 9999.9
    else:
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
    ij_correl_dict = dict((tuple(sorted([idx_i,idx_j])), calc_correl(sym_time_series_list[idx_i],sym_time_series_list[idx_j])) for idx_i in range(len(sym_time_series_list)) for idx_j in range(idx_i,len(sym_time_series_list)) )

    correl_matrix = []
    for idx_i in range(len(sym_time_series_list)):
        row = []
        for idx_j in range(len(sym_time_series_list)):
            row.append(ij_correl_dict[ tuple(sorted([idx_i,idx_j])) ])
        correl_matrix.append(row)

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
    annualized_sd_list = map(lambda x: x[0]*math.sqrt(x[1]), zip(sd_list.tolist(),annualization_factor_list))
    annualized_adj_sd_list = map(lambda x: x[0]+x[1], zip(annualized_sd_list,specific_riskiness_list))
    D = np.diag(annualized_adj_sd_list)
    # print "D: %s" % str(D)
    # print "correl_matrix: %s" % str(correl_matrix)
    # print "cov_matrix: %s" % str(D*correl_matrix*D)
    return ((D * correl_matrix * D),annualized_sd_list,annualized_adj_sd_list)

def calc_mean_vec(sym_time_series_list):
    return map(lambda ts: sum(map(lambda x: x[1], ts))/len(ts), sym_time_series_list)

def extract_sd_from_cov_matrix(cov_matrix):
    return map(lambda x: math.sqrt(x), np.diag(cov_matrix).tolist())

def markowitz(symbol_list,expected_rtn_list,cov_matrix,mu_p,max_weight_list,portfolio_change_inertia=None,hatred_for_small_size=None,current_weight_list=None):
    def iif(cond, iftrue=1.0, iffalse=0.0):
        if cond:
            return iftrue
        else:
            return iffalse

    n = len(symbol_list)

    ###################################################
    # P and q determine the objective function to minimize
    # which in cvxopt is defined as $.5 x^T P x + q^T x$
    if portfolio_change_inertia is None or current_weight_list is None or hatred_for_small_size is None:
        P = cvxopt.matrix(cov_matrix)
        q = cvxopt.matrix([0.0 for i in range(n)])
    else:
        P = cvxopt.matrix(0.5 * cov_matrix + (portfolio_change_inertia - hatred_for_small_size) * np.identity(n))
        q = cvxopt.matrix( map(lambda cw: (-2 * portfolio_change_inertia * cw) + (2 * hatred_for_small_size), current_weight_list) )

    ###################################################
    # G and h determine the inequality constraints in the
    # form $G x \leq h$. We write $w_i \geq 0$ as $-1 \times x_i \leq 0$
    # and also add a (superfluous) $x_i \leq 1$ constraint
    ###################################################
    # G x <= h
    # -1  0  0  0 ... 0          0
    #  1  0  0  0 ... 0          w_0
    #  0 -1  0  0 ... 0          0
    #  0  1  0  0 ... 0          w_1
    #  0  0 -1  0 ... 0
    #  0  0  1  0 ... 0
    ###################################################
    G = cvxopt.matrix([[ (-1.0)**(1+j%2) * iif(i == j/2) for i in range(n) ]
                for j in range(2*n)
                ]).trans()
    h = cvxopt.matrix([ max_weight_list[j/2] * iif(j % 2) for j in range(2*n) ])
    ###################################################

    # A and b determine the equality constraints defined as A x = b
    A = cvxopt.matrix([[ 1.0 for i in range(n) ],
                expected_rtn_list]).trans()
    b = cvxopt.matrix([ 1.0, float(mu_p) ])

    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except:
        return None

    if sol['status'] != 'optimal':
        return None

    w = list(sol['x'])
    f = 2.0*sol['primal objective']

    return {'w': w, 'f': f, 'args': (P, q, G, h, A, b), 'result': sol }

def log_optimal_growth(symbol_list,expected_rtn_list,cov_matrix,max_weight_list,portfolio_change_inertia=None,hatred_for_small_size=None,current_weight_list=None):
    def iif(cond, iftrue=1.0, iffalse=0.0):
        if cond:
            return iftrue
        else:
            return iffalse

    n = len(symbol_list)

    ###################################################
    # P and q determine the objective function to minimize
    # which in cvxopt is defined as $.5 x^T P x + q^T x$
    if portfolio_change_inertia is None or current_weight_list is None or hatred_for_small_size is None:
        P = cvxopt.matrix(0.5 * cov_matrix)
        q = cvxopt.matrix( map(lambda x: -x, expected_rtn_list))
    else:
        P = cvxopt.matrix(0.5 * cov_matrix + (portfolio_change_inertia - hatred_for_small_size) * np.identity(n))
        negated_expected_rtn_list = map(lambda x: -x, expected_rtn_list)
        penalty_list = map(lambda cw: (-2 * portfolio_change_inertia * cw) + (2 * hatred_for_small_size), current_weight_list)
        q = cvxopt.matrix(map(lambda x: x[0]+x[1], zip(negated_expected_rtn_list,penalty_list)))

    ###################################################
    # G and h determine the inequality constraints in the
    # form $G x \leq h$. We write $w_i \geq 0$ as $-1 \times x_i \leq 0$
    # and also add a (superfluous) $x_i \leq 1$ constraint
    ###################################################
    # G x <= h
    # -1  0  0  0 ... 0          0
    #  1  0  0  0 ... 0          w_0
    #  0 -1  0  0 ... 0          0
    #  0  1  0  0 ... 0          w_1
    #  0  0 -1  0 ... 0
    #  0  0  1  0 ... 0
    ###################################################
    G = cvxopt.matrix([[ (-1.0)**(1+j%2) * iif(i == j/2) for i in range(n) ]
                for j in range(2*n)
                ]).trans()
    h = cvxopt.matrix([ max_weight_list[j/2] * iif(j % 2) for j in range(2*n) ])
    ###################################################

    # A and b determine the equality constraints defined as A x = b
    A = cvxopt.matrix([[ 1.0 for i in range(n) ]]).trans()
    b = cvxopt.matrix([ 1.0 ])

    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except:
        return None

    if sol['status'] != 'optimal':
        return None

    w = list(sol['x'])
    f = 2.0*sol['primal objective']

    return {'w': w, 'f': f, 'args': (P, q, G, h, A, b), 'result': sol }

def markowitz_robust(symbol_list,expected_rtn_list,cov_matrix,mu_p,max_weight_list,expected_rtn_uncertainty_list,portfolio_change_inertia=None,hatred_for_small_size=None,current_weight_list=None):
    n = len(symbol_list)
    c_ = [{'type':'eq', 'fun': lambda w: sum(w)-1.0 }, {'type':'eq', 'fun': lambda w: sum([w_*r_ for w_,r_ in zip(w,expected_rtn_list)])-mu_p }]   # sum of weights = 100% and expected_rtn = mu
    b_ = map(lambda x: (0.0, x), max_weight_list)  # upper and lower bounds of weights

    def objective_func(w, cov_matrix, portfolio_change_inertia=None,hatred_for_small_size=None,current_weight_list=None):
        w = np.asarray(w)
        mu_uncertainty_matrix = np.diag(map(lambda x: math.pow(x,2), expected_rtn_uncertainty_list))
        mu_robustness_penalty = 0.1 * math.sqrt((w * np.asmatrix(mu_uncertainty_matrix)).dot(w))

        if portfolio_change_inertia is None or current_weight_list is None or hatred_for_small_size is None:
            covar = (w * cov_matrix).dot(w)
            return covar + mu_robustness_penalty
        else:
            mod_cov_matrix = 0.5 + cov_matrix + (portfolio_change_inertia - hatred_for_small_size) * np.identity(n)
            quad_term = (w * mod_cov_matrix).dot(w)
            print "quad: %s" % quad
            lin_term = sum(map(lambda cur_w: ((-2 * portfolio_change_inertia * cw) + (2 * hatred_for_small_size)) * cur_w, current_weight_list))
            print "lin_term: %s" % lin_term
            return quad_term + lin_term + mu_robustness_penalty

    ###################################################
    w = np.ones([n])/n
    try:
        sol = scipy.optimize.minimize(objective_func, w, cov_matrix, method='SLSQP', constraints=c_, bounds=b_)  
        # print "sol: %s" % sol
    except:
        return None

    if not sol.success: 
       return None
    return {'result': {'x': list(sol.x)} }


