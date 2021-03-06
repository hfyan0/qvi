#!/usr/bin/env python
import sys
import os
import fnmatch
import math
import numpy as np
import cvxopt
from cvxopt import blas, solvers
from datetime import datetime, timedelta
import scipy.optimize
import itertools
import random
import time
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import cPickle
import glob

random.seed(time.time())
###################################################
LOOKBACK_DAYS = 378 # Most recent (for correl)
###################################################

class CurrencyConverter(object):
    def __init__(self, currency_rate_dict):
        self.preread_data_list = {}
        self.preread_data_dict = {}
        for curcy,file_loc in currency_rate_dict.items():
            self.preread_data_list[curcy] = sorted(map(lambda x: (datetime.strptime(x[0],"%Y-%m-%d").date(),x[2]), read_file(file_loc)), key=lambda x: x[0])
            self.preread_data_dict[curcy] = dict(self.preread_data_list[curcy])
    def get_conv_rate_to_hkd(self,currency,dt):
        if currency in self.preread_data_dict:
            return float(self.preread_data_dict[currency].get(dt, filter(lambda x: x[0] <= dt, self.preread_data_list[currency])[-1][1]))
        return 1.0
    def conv_to_hkd(self,currency,dt,amt):
        return float(amt) * self.get_conv_rate_to_hkd(currency,dt)

def read_file_mul(tup):
    if len(tup) == 2:
        file_loc,k = tup
        with open(file_loc,'r') as f:
            return (k,[map(lambda x: x.strip(), line.split(',')) for line in f])
    else:
        return None


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

###################################################
def get_hist_data_key_date(filename):
    rtn_dict = {}
    for d,it_lstup in itertools.groupby(sorted(read_file(filename), key=lambda x: x[0]), lambda x: x[0]):
        rtn_dict[datetime.strptime(d,"%Y-%m-%d").date()] = dict(map(lambda y: (y[1],float(y[2])), filter(lambda x: abs(float(x[2])) > 0.0001, list(it_lstup))))
    return rtn_dict

def get_hist_data_key_sym(filename):
    rtn_dict = {}
    for s,it_lstup in itertools.groupby(sorted(read_file(filename), key=lambda x: x[1]), lambda x: x[1]):
        rtn_dict[s] = sorted(map(lambda y: (datetime.strptime(y[0],"%Y-%m-%d").date(),float(y[2])), filter(lambda x: abs(float(x[2])) > 0.0001, list(it_lstup))), key=lambda x: x[0])
    return rtn_dict

def shift_back_n_months(dt,n):
    return dt - timedelta(weeks = float(52.0/12.0*float(n)))
###################################################
def adj_val_per_share_with_no_of_shares(ps_list,no_of_shr_list):
    sorted_no_of_shr_list = sorted(no_of_shr_list, key=lambda x: x[0])
    sorted_ps_list        = sorted(ps_list, key=lambda x: x[0])

    if len(sorted_ps_list) == 0 or len(sorted_no_of_shr_list) == 0:
        return sorted_ps_list

    agg_list = []
    for dt,val in sorted_ps_list:
        tmp_list=map(lambda x: x[1], filter(lambda x: x[0]<=dt, sorted_no_of_shr_list))
        no_of_shr = tmp_list[-1] if len(tmp_list) > 0 else sorted_no_of_shr_list[0][1]
        agg_list.append((dt,val,no_of_shr))

    # print agg_list
    return map(lambda x: (x[0],x[1]*x[2]/agg_list[-1][2]), agg_list)
###################################################

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

###################################################
# From https://github.com/better/irr
###################################################
MAX_LOG_RATE = 1e3
BASE_TOL = 1e-12

def irr_binary_search(stream, tol=BASE_TOL):
    rate_lo, rate_hi = -MAX_LOG_RATE, +MAX_LOG_RATE
    sgn = np.sign(stream[0]) # f(x) is decreasing
    for steps in range(100):
        rate = (rate_lo + rate_hi)/2
        r = np.arange(len(stream))
        # Factor exp(m) out because it doesn't affect the sign
        m = max(-rate * r)
        f = np.exp(-rate * r - m)
        t = np.dot(f, stream)
        if abs(t) < tol * math.exp(-m):
            break
        if t * sgn > 0:
            rate_hi = rate
        else:
            rate_lo = rate
    rate = (rate_lo + rate_hi) / 2
    return math.exp(rate) - 1


def irr_newton(stream, tol=BASE_TOL):
    rate = 0.0
    for steps in range(50):
        r = np.arange(len(stream))
        # Factor exp(m) out of the numerator & denominator for numerical stability
        m = max(-rate * r)
        f = np.exp(-rate * r - m)
        t = np.dot(f, stream)
        if abs(t) < tol * math.exp(-m):
            break
        u = np.dot(f * r, stream)
        # Clip the update to avoid jumping into some numerically unstable place
        rate = rate + np.clip(t / u, -1.0, 1.0)

    return math.exp(rate) - 1


def irr_vec(cfs):
    # Create companion matrix for every row in `cfs`
    M, N = cfs.shape
    A = np.zeros((M, (N-1)**2))
    A[:,N-1::N] = 1
    A = A.reshape((M,N-1,N-1))
    A[:,0,:] = cfs[:,-2::-1] / -cfs[:,-1:]  # slice [-1:] to keep dims

    # Calculate roots; `eigvals` is a gufunc
    res = np.linalg.eigvals(A)

    # Find the solution that makes the most sense...
    mask = (res.imag == 0) & (res.real > 0)
    res = np.ma.array(res.real, mask=~mask, fill_value=np.nan)
    rate = 1.0/res - 1
    idx = np.argmin(np.abs(rate), axis=1)
    irr = rate[np.arange(M), idx].filled()
    return irr

def calc_return_list(price_list):
    return np.divide(np.array(price_list[1:]),np.array(price_list[:-1]))-np.ones(len(price_list)-1)

def calc_correl(ts_a,ts_b):
    common_date_set = set(map(lambda x: x[0], ts_a)).intersection(set(map(lambda x: x[0], ts_b)))
    if len(common_date_set) < 30:
        return 1.0 # conservative

    a_ext = map(lambda x: x[1], sorted(filter(lambda x: x[0] in common_date_set, ts_a), key=lambda x: x[0]))
    b_ext = map(lambda x: x[1], sorted(filter(lambda x: x[0] in common_date_set, ts_b), key=lambda x: x[0]))
    common_len = min(min(len(a_ext),len(b_ext)),LOOKBACK_DAYS)
    a_ext = a_ext[-common_len:]
    b_ext = b_ext[-common_len:]

    return round(np.corrcoef(calc_return_list(a_ext),calc_return_list(b_ext))[0][1],5)

def calc_sd(ts):
    if len(ts) < 30:
        return 9999.9
    else:
        return np.std(calc_return_list(ts[-LOOKBACK_DAYS:]))

def get_annualization_factor(date_list):
    if len(date_list) < 3:
        return 1
    else:
        date_list = sorted(date_list)
        dl = sorted(map(lambda x: (x[0]-x[1]).days, zip(date_list[1:],date_list[:-1])))
        median_diff = (dl[(len(dl)/2)-1] + dl[(len(dl)/2)])/2.0
        if median_diff <= 4:
            return 252
        elif median_diff <= 7:
            return 52
        elif median_diff <= 40:
            return 12
        elif median_diff <= 35*3:
            return 4
        elif median_diff <= 35*6:
            return 2
        else:
            return 1

def calc_cov_matrix_annualized(sym_time_series_list,num_of_jobs=3,debug_mode=False):
    ###################################################
    # correlation matrix
    ###################################################
    N = len(sym_time_series_list)
    # ij_correl_dict = dict((tuple(sorted([idx_i,idx_j])), calc_correl(sym_time_series_list[idx_i],sym_time_series_list[idx_j])) for idx_i in range(len(sym_time_series_list)) for idx_j in range(idx_i,len(sym_time_series_list)) )
    # ts_list = [j for i in map(lambda idx_j: map(lambda idx_i: (sym_time_series_list[idx_i],sym_time_series_list[idx_j],idx_i,idx_j), range(N)), range(N)) for j in i]
    ij_correl_dict = {}

    ###################################################
    def calc_correl_exclusive(idx_i,idx_j):
        common_date_set = set(map(lambda x: x[0], sym_time_series_list[idx_i])).intersection(set(map(lambda x: x[0], sym_time_series_list[idx_j])))
        if len(common_date_set) < 30:
            return 1.0 # conservative
        a_ext = map(lambda x: x[1], sorted(filter(lambda x: x[0] in common_date_set, sym_time_series_list[idx_i][-(LOOKBACK_DAYS*2):]), key=lambda x: x[0]))
        b_ext = map(lambda x: x[1], sorted(filter(lambda x: x[0] in common_date_set, sym_time_series_list[idx_j][-(LOOKBACK_DAYS*2):]), key=lambda x: x[0]))
        common_len = min(min(len(a_ext),len(b_ext)),LOOKBACK_DAYS)
        a_ext = a_ext[-common_len:]
        b_ext = b_ext[-common_len:]
        res = np.corrcoef(calc_return_list(a_ext),calc_return_list(b_ext))[0][1]
        return round(1.0 if np.isnan(res) or res is None else res,5)
    ###################################################

    for idx_i in range(N):
        for idx_j in range(idx_i,N):
            ij_correl_dict[tuple(sorted([idx_i,idx_j]))] = calc_correl_exclusive(idx_i,idx_j)

    correl_matrix = []
    for idx_i in range(N):
        row = []
        for idx_j in range(N):
            row.append(ij_correl_dict[ tuple(sorted([idx_i,idx_j])) ])
        correl_matrix.append(row)

    if debug_mode:
        print "len(correl_matrix) %s" % len(correl_matrix)
        print ','.join(map(str, map(lambda x: len(x), correl_matrix)))

    correl_matrix = np.asmatrix(correl_matrix)

    np.set_printoptions(precision=5,suppress=True)
    # if len(correl_matrix.tolist()) < 6:
    #     print
    #     print "correl_matrix"
    #     for row in correl_matrix.tolist():
    #         for element in row:
    #             str_element = str(round(element,3))
    #             sys.stdout.write(justify_str(str_element,8))
    #         sys.stdout.write('\n')

    ###################################################
    # covariance matrix
    ###################################################
    sd_list = np.asarray(map(lambda ts: calc_sd(map(lambda x: x[1], ts)), sym_time_series_list))
    annualization_factor_list = (map(lambda ts: get_annualization_factor(map(lambda x: x[0], ts)), sym_time_series_list))
    annualized_sd_list = map(lambda x: x[0]*math.sqrt(x[1]), zip(sd_list.tolist(),annualization_factor_list))
    D = np.diag(annualized_sd_list)
    # print "D: %s" % str(D)
    # print "correl_matrix: %s" % str(correl_matrix)
    # print "cov_matrix: %s" % str(D*correl_matrix*D)
    return ((D * correl_matrix * D),annualized_sd_list,annualized_sd_list)

def extract_sd_from_cov_matrix(cov_matrix):
    return map(lambda x: math.sqrt(x), np.diag(cov_matrix).tolist())

def markowitz(symbol_list,expected_rtn_list,cov_matrix,mu_p,max_weight_list,min_exp_rtn,industry_groups_list,max_weight_industry,portfolio_change_inertia=None,hatred_for_small_size=None,current_weight_list=None):
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
    # G = cvxopt.matrix([
    #             [ (-1.0)**(1+j%2) * iif(i == j/2) for i in range(n) ]
    #             for j in range(2*n)
    #             ]
    #             ).trans()
    # h = cvxopt.matrix([ max_weight_list[j/2] * iif(j % 2) for j in range(2*n) ])
    G = cvxopt.matrix(
                [ [ (-1.0 if (i==j) else 0.0) for i in range(n) ] for j in range(n) ] +
                [ [ ( 1.0 if (i==j) else 0.0) for i in range(n) ] for j in range(n) ] +
                map(lambda ig_set: map(lambda s: 1.0 if (s in ig_set) else 0.0, symbol_list), industry_groups_list)
                ).trans()
    h = cvxopt.matrix( [ 0.0 for i in range(n) ] + [ 0.0 if expected_rtn_list[i] < min_exp_rtn else max_weight_list[i] for i in range(n) ] + map(lambda x: max_weight_industry, industry_groups_list) )
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

def markowitz_sharpe(symbol_list,expected_rtn_list,cov_matrix,max_weight_list,min_exp_rtn,industry_groups_list,max_weight_industry,portfolio_change_inertia=None,hatred_for_small_size=None,current_weight_list=None):
    def iif(cond, iftrue=1.0, iffalse=0.0):
        if cond:
            return iftrue
        else:
            return iffalse

    n = len(symbol_list)

    ###################################################
    # P and q determine the objective function to minimize
    # which in cvxopt is defined as $.5 x^T P x + q^T x$

    ###################################################
    # FIXME
    ###################################################
    P = cvxopt.matrix(2.0 * cov_matrix)
    q = cvxopt.matrix([0.0 for i in range(n)])
    # if portfolio_change_inertia is None or current_weight_list is None or hatred_for_small_size is None:
    #     P = cvxopt.matrix(2.0 * cov_matrix)
    #     q = cvxopt.matrix([0.0 for i in range(n)])
    # else:
    #     P = cvxopt.matrix(2.0 * (cov_matrix + (portfolio_change_inertia - hatred_for_small_size) * np.identity(n)))
    #     l1 = map(lambda cw: (-2 * portfolio_change_inertia * cw) + (2 * hatred_for_small_size), current_weight_list)
    #     l2 = map(lambda x: -x, expected_rtn_list)
    #     q = cvxopt.matrix( map(lambda x: x[0]+x[1], zip(l1,l2)) )

    ###################################################
    # G x <= h
    ###################################################
    min_exp_rtn_and_max_weight_constraint_list = [ 0.0 if expected_rtn_list[i] < min_exp_rtn else max_weight_list[i] for i in range(n) ]

    Gm = [ [ ((-1.0 if (i==j) else 0.0) - 0.0)                                           for i in range(n) ] for j in range(n) ] +\
         [ [ (( 1.0 if (i==j) else 0.0) - min_exp_rtn_and_max_weight_constraint_list[j]) for i in range(n) ] for j in range(n) ] +\
         map(lambda ig_set: map(lambda s: (1.0 if (s in ig_set) else 0.0) - max_weight_industry, symbol_list), industry_groups_list)

    G = cvxopt.matrix(Gm).trans()
    h = cvxopt.matrix([ 0.0 ] * len(Gm))

    ###################################################
    # A and b determine the equality constraints defined as A x = b
    ###################################################
    A = cvxopt.matrix( [expected_rtn_list] ).trans()
    b = cvxopt.matrix([ 1.0 ])

    ###################################################
    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except:
        return None

    if sol['status'] != 'optimal':
        return None

    ###################################################
    # transform back to the original space
    ###################################################
    y_list = list(sol['x'])
    sy = sum(y_list)
    sol['x'] = map(lambda y: y / sy, y_list)
    return {'result': sol }

def log_optimal_growth(symbol_list,expected_rtn_list,cov_matrix,max_weight_list,min_exp_rtn,industry_groups_list,max_weight_industry,portfolio_change_inertia=None,hatred_for_small_size=None,current_weight_list=None):
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
    # G = cvxopt.matrix([[ (-1.0)**(1+j%2) * iif(i == j/2) for i in range(n) ]
    #             for j in range(2*n)
    #             ] +
    #             map(lambda ig_set: map(lambda s: 1.0 if (s in ig_set) else 0.0, symbol_list), industry_groups_list)
    #             ).trans()
    # h = cvxopt.matrix([ max_weight_list[j/2] * iif(j % 2) for j in range(2*n) ] + map(lambda x: max_weight_industry, industry_groups_list))

    min_exp_rtn_and_max_weight_constraint_list = [ 0.0 if expected_rtn_list[i] < min_exp_rtn else max_weight_list[i] for i in range(n) ]

    Gm = [ [ ((-1.0 if (i==j) else 0.0) - 0.0)                                           for i in range(n) ] for j in range(n) ] +\
         [ [ (( 1.0 if (i==j) else 0.0) - min_exp_rtn_and_max_weight_constraint_list[j]) for i in range(n) ] for j in range(n) ] +\
         map(lambda ig_set: map(lambda s: (1.0 if (s in ig_set) else 0.0) - max_weight_industry, symbol_list), industry_groups_list)

    G = cvxopt.matrix(Gm).trans()
    h = cvxopt.matrix([ 0.0 ] * len(Gm))

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


###################################################

def calc_expected_return_before_201703(config,dt,symbol_list,hist_bps_dict,hist_unadj_px_dict,hist_operincm_dict,hist_totasset_dict,hist_totliabps_dict,hist_costofdebt_dict,hist_stattaxrate_dict,hist_oper_eps_dict,hist_eps_dict,hist_roa_dict,hist_outshr_dict,delay_months,debug_mode):

    def cal_weighted_avg(a_list):
        n = len(a_list)
        return sum(map(lambda x: x[0]*x[1], enumerate(a_list))) / (n*(n-1)/2.0)

    curcy_converter = CurrencyConverter(config["currency_rate"])
    ###################################################
    # book-to-price
    ###################################################
    bp_list = []
    conser_bp_ratio_list = []
    conser_bp_ratio_dict = {}
    for sym in symbol_list:
        if debug_mode:
            print "sym: %s" % sym

        reporting_curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["reporting_currency"].get(sym,config["reporting_currency"]["default"]),dt)
        price_curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["price_currency"].get(sym,config["price_currency"]["default"]),dt)
        # print sym,dt,reporting_curcy_conv_rate

        bps_list = filter(lambda y: y[0] > shift_back_n_months(dt,5*12+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_bps_dict.get(sym,[])))
        bps_list = adj_val_per_share_with_no_of_shares(bps_list,hist_outshr_dict.get(sym,[]))
        bps_list = map(lambda x: x[1], bps_list)
        ###################################################
        if len(bps_list) >= 3 and sym in hist_unadj_px_dict[dt]:
            bps_r = calc_return_list(bps_list)
            # m = np.sum(bps_r)/(bps_r.size)
            # sd = np.std(bps_r)
            # conser_bp_ratio = max(reporting_curcy_conv_rate * bps_list[-1] * (1 + min(m,0.0) - float(config["general"]["bp_stdev"]) * sd) / price_curcy_conv_rate / hist_unadj_px_dict[dt][sym], 0.0)
            conser_bp_ratio = max(reporting_curcy_conv_rate * cal_weighted_avg(bps_list) / price_curcy_conv_rate / hist_unadj_px_dict[dt][sym], 0.0)
            conser_bp_ratio_list.append(conser_bp_ratio)
            conser_bp_ratio_dict[sym] = conser_bp_ratio
        else:
            conser_bp_ratio_list.append(0.0)
            conser_bp_ratio_dict[sym] = 0.0
        ###################################################
        if len(bps_list) > 0 and sym in hist_unadj_px_dict[dt]:
            bp_list.append(bps_list[-1] / price_curcy_conv_rate / hist_unadj_px_dict[dt][sym])
        else:
            bp_list.append(0.0)
        ###################################################

    expected_rtn_list = []
    expected_rtn_asset_driver_list = []
    expected_rtn_external_driver_list = []
    expected_rtn_bv_list = []
    for sym in symbol_list:
        if debug_mode:
            print "--------------------------------------------------"
            print "sym: %s" % sym
            print "--------------------------------------------------"

        if sym not in hist_unadj_px_dict[dt]:
            expected_rtn_asset_driver_list.append(0)
            expected_rtn_external_driver_list.append(0)
            expected_rtn_bv_list.append(0)
            print "--------------------------------------------------"
            print "No unadj price: %s" % sym
            print "--------------------------------------------------"
            continue

        reporting_curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["reporting_currency"].get(sym,config["reporting_currency"]["default"]),dt)
        price_curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["price_currency"].get(sym,config["price_currency"]["default"]),dt)
        ###################################################
        indus_grp = config["industry_group"].get(sym,config["industry_group"]["default"])
        w_ig_list = []
        try:
            w_ig_list.append((1.0,str(int(indus_grp))))
        except Exception, e:
            w_ig_list.extend(map(lambda x: (float(x.split(':')[1]),str(x.split(':')[0])), indus_grp))

        w_a_w_e_bv_list = []
        for w,ig in w_ig_list:
            w_a_w_e_bv_list.append(tuple(map(lambda x: w*x, map(float, config["expected_rtn_ast_ext_bvrlzn_bvrcvy"].get(ig, config["expected_rtn_ast_ext_bvrlzn_bvrcvy"]["0"])))))

        w_a    = sum(map(lambda x: x[0], w_a_w_e_bv_list))
        w_e    = sum(map(lambda x: x[1], w_a_w_e_bv_list))
        bvrlzn = sum(map(lambda x: x[2], w_a_w_e_bv_list))
        bvrcvy = sum(map(lambda x: x[3], w_a_w_e_bv_list))
        # print "sym: %s %s %s" % (sym, w_a, w_e)

        ###################################################
        # asset driver: calculating my own oper ROA
        ###################################################
        conser_oper_roa = None

        oper_roa_list = []
        oper_incm_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*4+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_operincm_dict.get(sym,[])))
        for oi_dt,oper_incm in oper_incm_list:
            totasset_list = filter(lambda x: x[0] <= shift_back_n_months(oi_dt,6), hist_totasset_dict.get(sym,[]))
            if len(totasset_list) > 0:
                oper_roa_list.append((oi_dt,oper_incm/totasset_list[-1][1]))

        ###################################################
        # annualization factor
        ###################################################
        oper_date_list = sorted(list(set(map(lambda x: x[0], oper_incm_list))))
        if debug_mode:
            print "oper_date_list: %s" % oper_date_list
        annualization_factor = get_annualization_factor(oper_date_list)
        ###################################################

        ###################################################
        # asset driver: calculating conservative oper ROA
        ###################################################
        if len(oper_roa_list) >= 5:
            oper_roa_list = sorted(sorted(oper_roa_list,key=lambda x: x[1])[1:][:-1],key=lambda x: x[0])
            oper_roa_list = map(lambda x: x[1], oper_roa_list)

            # m = sum(oper_roa_list)/len(oper_roa_list)
            # sd = np.std(np.asarray(oper_roa_list))
            # conser_oper_roa = annualization_factor * (min(m,oper_roa_list[-1]) - float(config["general"]["roa_drvr_stdev"]) * sd)
            conser_oper_roa = annualization_factor * cal_weighted_avg(oper_roa_list)
            if debug_mode:
                print "oper_roa_list: %s" % oper_roa_list
                print "annualization_factor: %s" % annualization_factor
                # print "m: %s" % m
                # print "sd: %s" % sd
                print "conser_oper_roa (annualized): %s" % conser_oper_roa
        else:
            conser_oper_roa = 0.0

        ###################################################
        # asset driver: just using ROA
        ###################################################
        roa_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*4+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_roa_dict.get(sym,[])))
        if len(roa_list) >= 5:
            roa_list = sorted(sorted(roa_list,key=lambda x: x[1])[1:][:-1],key=lambda x: x[0])
            roa_list = map(lambda x: x[1], roa_list)
            # m = sum(roa_list)/len(roa_list)
            # sd = np.std(np.asarray(roa_list))
            ###################################################
            # Bloomberg's ROA is already annualized
            ###################################################
            # conser_roa = (min(m,roa_list[-1]) - float(config["general"]["roa_drvr_stdev"]) * sd) / 100.0
            conser_roa = cal_weighted_avg(roa_list) / 100.0
            if debug_mode:
                print "roa_list (annualized %%): %s" % roa_list
                # print "m: %s" % m
                # print "sd: %s" % sd
                print "conser_roa (annualized): %s" % conser_roa
        else:
            conser_roa = 0.0

        bps_list = filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_bps_dict.get(sym,[]))
        bps = bps_list[-1][1] if len(bps_list) > 0 else 0.0

        costofdebt_list = filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_costofdebt_dict.get(sym,[]))
        totliabps_list = filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_totliabps_dict.get(sym,[]))
        totliabps_list = adj_val_per_share_with_no_of_shares(totliabps_list,hist_outshr_dict.get(sym,[]))
        if len(costofdebt_list) > 0 and len(totliabps_list) > 0:
            iL = costofdebt_list[-1][1]/100.0 * totliabps_list[-1][1]
        else:
            iL = 0.0
        ###################################################
        # FIXME
        ###################################################
        iL = 0.0
        ###################################################

        if len(totliabps_list) > 0:
            totliabps = totliabps_list[-1][1]
        else:
            totliabps = 0.0

        stattaxrate_list = filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_stattaxrate_dict.get(sym,[]))
        if len(stattaxrate_list) > 0:
            taxrate = stattaxrate_list[-1][1]
        else:
            taxrate = 0.0

        meth_1 = (1-taxrate)*(conser_oper_roa*(totliabps+bps)-iL)*reporting_curcy_conv_rate/price_curcy_conv_rate/hist_unadj_px_dict[dt][sym]
        meth_2 = (conser_roa*(totliabps+bps))*reporting_curcy_conv_rate/price_curcy_conv_rate/hist_unadj_px_dict[dt][sym]

        if debug_mode:
            print "tax rate: %s" % taxrate
            # print "stattaxrate_list: %s" % stattaxrate_list
            print "totliabps: %s" % totliabps
            print "bps: %s" % bps
            print "--------------------------------------------------"
            print "asset driver: meth_1: %s" % meth_1
            print "asset driver: meth_2: %s" % meth_2
            print "--------------------------------------------------"

        meth_1 = w_a*meth_1
        meth_2 = w_a*meth_2

        expected_rtn_asset_driver_list.append(min(meth_1,meth_2))

        ###################################################
        # external driver: oper_eps
        ###################################################
        oper_eps_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*4+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_oper_eps_dict.get(sym,[])))
        oper_eps_list = adj_val_per_share_with_no_of_shares(oper_eps_list,hist_outshr_dict.get(sym,[]))
        oper_eps_chg_list = map(lambda x: x[0][1]-x[1][1], zip(oper_eps_list[1:],oper_eps_list[:-1]))
        if len(oper_eps_chg_list) >= 5:
            oper_eps_list = map(lambda z: z[1][1], sorted(sorted(enumerate(oper_eps_list), key=lambda x: x[1][1])[1:][:-1], key=lambda y: y[0]))
            oper_eps_chg_list = sorted(oper_eps_chg_list)[1:][:-1]
            # m = sum(oper_eps_chg_list)/len(oper_eps_chg_list)
            # sd = np.std(np.asarray(oper_eps_chg_list))
            # conser_oper_eps = (oper_eps_list[-1] + min(m,0.0) - float(config["general"]["ext_drvr_stdev"]) * sd) * annualization_factor
            conser_oper_eps = cal_weighted_avg(oper_eps_list) * annualization_factor
            if debug_mode:
                print "oper_eps_list: %s" % oper_eps_list
                print "oper_eps_chg_list: %s" % oper_eps_chg_list
                print "annualization_factor: %s" % annualization_factor
                # print "m: %s" % m
                # print "sd: %s" % sd
                print "conser_oper_eps (annualized): %s" % conser_oper_eps
        else:
            conser_oper_eps = 0.0

        ###################################################
        # external driver: eps
        ###################################################
        eps_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*4+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_eps_dict.get(sym,[])))
        eps_list = adj_val_per_share_with_no_of_shares(eps_list,hist_outshr_dict.get(sym,[]))
        eps_chg_list = map(lambda x: x[0][1]-x[1][1], zip(eps_list[1:],eps_list[:-1]))
        if len(eps_chg_list) >= 5:
            eps_list = map(lambda z: z[1][1], sorted(sorted(enumerate(eps_list), key=lambda x: x[1][1])[1:][:-1], key=lambda y: y[0]))
            eps_chg_list = sorted(eps_chg_list)[1:][:-1]
            # m = sum(eps_chg_list)/len(eps_chg_list)
            # sd = np.std(np.asarray(eps_chg_list))
            # conser_eps = (eps_list[-1] + min(m,0.0) - float(config["general"]["ext_drvr_stdev"]) * sd) * annualization_factor
            conser_eps = cal_weighted_avg(eps_list) * annualization_factor
            if debug_mode:
                print "eps_list: %s" % eps_list
                print "eps_chg_list: %s" % eps_chg_list
                print "annualization_factor: %s" % annualization_factor
                # print "m: %s" % m
                # print "sd: %s" % sd
                print "conser_eps: %s" % conser_eps
        else:
            conser_eps = 0.0

        meth_1 = (1-taxrate)*(conser_oper_eps-iL)*reporting_curcy_conv_rate/price_curcy_conv_rate/hist_unadj_px_dict[dt][sym]
        meth_2 = conser_eps*reporting_curcy_conv_rate/price_curcy_conv_rate/hist_unadj_px_dict[dt][sym]

        if debug_mode:
            print "tax rate: %s" % taxrate
            # print "stattaxrate_list: %s" % stattaxrate_list
            print "reporting_curcy_conv_rate: %s" % reporting_curcy_conv_rate
            print "price_curcy_conv_rate: %s" % price_curcy_conv_rate
            print "--------------------------------------------------"
            print "external driver: meth_1: %s" % meth_1
            print "external driver: meth_2: %s" % meth_2
            print "--------------------------------------------------"

        meth_1 = w_e*meth_1
        meth_2 = w_e*meth_2

        expected_rtn_external_driver_list.append(min(meth_1,meth_2))

        ###################################################
        # liquidation to realize book value
        ###################################################
        meth_1 = math.pow(bvrcvy*conser_bp_ratio_dict[sym],1.0/bvrlzn)-1
        expected_rtn_bv_list.append(meth_1)

        if debug_mode:
            print "conser_bp_ratio: %s" % conser_bp_ratio_dict[sym]
            print "bvrlzn: %s" % bvrlzn
            print "bvrcvy: %s" % bvrcvy
            print "--------------------------------------------------"
            print "book value realization: %s" % meth_1
            print "--------------------------------------------------"


    expected_rtn_list = map(lambda x: max(x[0]+x[1],x[2]), zip(expected_rtn_asset_driver_list,expected_rtn_external_driver_list,expected_rtn_bv_list))

    return expected_rtn_list


def cal_irr_mean_var_Reinschmidt(mean_list,sd_list,corr_matrix,cov_matrix):
    N = len(mean_list)-1
    G = 1.0+np.irr(mean_list)
    D = sum([mean_list[i] * (N-i) * math.pow(G,N-i-1) for i in range(N+1)])
    G_prime_list = [-math.pow(G,N-i)/D for i in range(N+1)]
    partial_D_partial_G = sum([mean_list[i] * (N-i) * (N-i-1) * math.pow(G,N-i-2) for i in range(N+1)])
    partial_D_partial_x_list = [math.pow(G,N-i-1) * (N-i) + partial_D_partial_G * G_prime_list[i] for i in range(N+1)]
    sec_derivative_G = [[-(N-k)*math.pow(G,N-k-1)/D*G_prime_list[j] + math.pow(G,N-k)/D/D*partial_D_partial_x_list[j] for j in range(N+1)] for k in range(N+1)]
    if corr_matrix is not None:
        EG = G + sum([sec_derivative_G[j][k] * corr_matrix[j][k]*sd_list[j]*sd_list[k] for j in range(N+1) for k in range(N+1)])/2.0
        VG = sum([G_prime_list[j] * G_prime_list[k] * corr_matrix[j][k]*sd_list[j]*sd_list[k] for j in range(N+1) for k in range(N+1)])
    elif cov_matrix is not None:
        EG = G + sum([sec_derivative_G[j][k] * cov_matrix[j][k] for j in range(N+1) for k in range(N+1)])/2.0
        VG = sum([G_prime_list[j] * G_prime_list[k] * cov_matrix[j][k] for j in range(N+1) for k in range(N+1)])
    EIRR = EG - 1.0
    VIRR = VG
    SIRR = math.sqrt(VIRR)
    return EIRR,SIRR


def find_all_roots_brentq(f, a, b, pars=(), min_window=0.0001):
    try:
        one_root = scipy.optimize.brentq(f, a, b, pars)
        # print "Root at %g in [%g,%g] interval" % (one_root, a, b)
    except ValueError:
        # print "No root in [%g,%g] interval" % (a, b)
        return [] # No root in the interval

    if one_root-min_window>a:
        lesser_roots_list = [one_root] + find_all_roots_brentq(f, a, one_root-min_window, pars)
    else:
        lesser_roots_list = []

    if one_root+min_window<b:
        greater_roots_list = [one_root] + find_all_roots_brentq(f, one_root+min_window, b, pars)
    else:
        greater_roots_list = []

    return lesser_roots_list + [one_root] + greater_roots_list


def adj_irr_by_cf_time(irr_with_1st_cf_in_1yr, time_b4_1st_cf):
    if irr_with_1st_cf_in_1yr is None or time_b4_1st_cf is None:
        return None
    def f(irr_with_1st_cf_in_lessthan1yr, irr_with_1st_cf_in_1yr, time_b4_1st_cf):
        return math.pow(1.0+irr_with_1st_cf_in_lessthan1yr,1.0-time_b4_1st_cf)/irr_with_1st_cf_in_lessthan1yr-(1.0/irr_with_1st_cf_in_1yr)
    if irr_with_1st_cf_in_1yr > 0:
        sol_set = set(find_all_roots_brentq(f, 0.001, 1, pars=(irr_with_1st_cf_in_1yr, time_b4_1st_cf)))
        return None if len(sol_set) == 0 else list(sol_set)[0]
    else:
        sol_set = set(find_all_roots_brentq(f, -1, -0.001, pars=(irr_with_1st_cf_in_1yr, time_b4_1st_cf)))
        return None if len(sol_set) == 0 else list(sol_set)[0]

def get_divd_rlzn_external_driver(cur_eps,rand_eps_chg_list,bps):
    # print "cur_eps: %s" % cur_eps
    # print "bps: %s" % bps 
    # print "rand_eps_chg_list: %s" % rand_eps_chg_list
    eps_rlzn_path_list = (np.array([cur_eps]*len(rand_eps_chg_list)) + np.cumsum(rand_eps_chg_list)).tolist()
    # print "eps_rlzn_path_list: %s" % eps_rlzn_path_list

    dvd_rlzn_path_list = []
    for i in range(len(eps_rlzn_path_list)):
        if eps_rlzn_path_list[i] > 0.0:
            dvd = max(np.sum(np.array(eps_rlzn_path_list[:(i+1)])) - np.sum(np.array(dvd_rlzn_path_list)), 0.0)
            dvd_rlzn_path_list.append(dvd)
        else:
            dvd_rlzn_path_list.append(0.0)
            # if (sum(eps_rlzn_path_list) - sum(dvd_rlzn_path_list) < -bps):
            #     break
    return dvd_rlzn_path_list

def get_divd_rlzn_asset_driver(cur_roa,rand_roa_chg_list,bps,liabps):
    # print "cur_roa: %s" % cur_roa
    # print "bps: %s" % bps 
    # print "liabps: %s" % liabps
    # print "rand_roa_chg_list: %s" % rand_roa_chg_list
    roa_rlzn_path_list = (np.array([cur_roa]*len(rand_roa_chg_list)) + np.cumsum(rand_roa_chg_list)).tolist()
    # print "roa_rlzn_path_list: %s" % roa_rlzn_path_list

    init_aps = bps+liabps
    eps_rlzn_path_list = []
    aps_rlzn_path_list = []
    dvd_rlzn_path_list = []
    for i in range(len(roa_rlzn_path_list)):
        if i == 0:
            prev_aps = init_aps
        else:
            prev_aps = aps_rlzn_path_list[-1]

        prev_aps = max(prev_aps,0)

        eps_rlzn_path_list.append(prev_aps * roa_rlzn_path_list[i])

        if eps_rlzn_path_list[i] > 0.0:
            dvd_rlzn_path_list.append( max(np.sum(np.array(eps_rlzn_path_list[:(i+1)])) - np.sum(np.array(dvd_rlzn_path_list)), 0.0) )
        else:
            dvd_rlzn_path_list.append(0.0)
            # if (sum(eps_rlzn_path_list) - sum(dvd_rlzn_path_list) < -bps):
            #     break

        aps_rlzn_path_list.append(prev_aps + eps_rlzn_path_list[-1] - dvd_rlzn_path_list[-1])

    # print "eps_rlzn_path_list: %s" % eps_rlzn_path_list
    # print "aps_rlzn_path_list: %s" % aps_rlzn_path_list
    # print "dvd_rlzn_path_list: %s" % dvd_rlzn_path_list
    return dvd_rlzn_path_list


def cal_irr_mean_ci_MonteCarlo_1D(cur_px,cur_eps,sigma,bps,num_terms,num_times_montecarlo,confidence_level):
    irr_result_list = []
    for _ in itertools.repeat(None, num_times_montecarlo):
        dvd_rlzn_path_list = get_divd_rlzn_external_driver(cur_eps,np.random.normal(0, sigma, num_terms).tolist(),bps)
        irr = np.irr([-cur_px] + dvd_rlzn_path_list)
        if not math.isnan(irr):
            irr_result_list.append(round(irr,5))
    sorted_irr_list = sorted(irr_result_list)
    if len(sorted_irr_list) < 30:
        return None,None,None
    ci_leftside_pctg = (100.0-confidence_level)/2.0/100.0
    ci_rightside_pctg = 1.0 - ci_leftside_pctg

    irr_mean = round(sum(sorted_irr_list)/len(sorted_irr_list),5)
    # irr_median = sorted_irr_list[len(sorted_irr_list)/2]
    irr_lower_pctl = sorted_irr_list[int(len(sorted_irr_list)*ci_leftside_pctg)]
    irr_upper_pctl = sorted_irr_list[int(len(sorted_irr_list)*ci_rightside_pctg)]
    irr_true_mean = cur_eps/cur_px
    irr_true_lower_pctl = irr_true_mean - (irr_mean - irr_lower_pctl)
    irr_true_upper_pctl = irr_true_mean - (irr_mean - irr_upper_pctl)
    return [irr_true_mean,irr_true_lower_pctl,irr_true_upper_pctl]

def calc_irr_mean_ci_before_20170309(config,dt,symbol,ann_sd_sym_rtn,hist_unadj_px_dict,hist_eps_dict,hist_bps_dict,confidence_level,delay_months,debug_mode):
    irr_mean_ci_tuple = [None,None,None]

    curcy_converter = CurrencyConverter(config["currency_rate"])

    if debug_mode:
        print "symbol: %s" % symbol

    reporting_curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["reporting_currency"].get(symbol,config["reporting_currency"]["default"]),dt)
    price_curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["price_currency"].get(symbol,config["price_currency"]["default"]),dt)
    # print symbol,dt,reporting_curcy_conv_rate

    bps_list = map(lambda z: z[1], filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_bps_dict.get(symbol,[])))
    if len(bps_list) == 0:
        return irr_mean_ci_tuple
    else:
        bps = bps_list[-1]

    next_cf_dt_list = map(lambda x: x[0], filter(lambda y: y[0] > shift_back_n_months(dt,5*12+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_eps_dict.get(symbol,[]))))
    time_to_next_cf = 1.0 if (len(next_cf_dt_list) == 0) else (((next_cf_dt_list[-1] + timedelta(weeks = 52)) - dt).days / 365.0)

    if debug_mode:
        print "--------------------------------------------------"
        print "symbol: %s" % symbol
        print "--------------------------------------------------"

    if symbol not in hist_unadj_px_dict[dt]:
        print "--------------------------------------------------"
        print "No unadj price: %s" % symbol
        print "--------------------------------------------------"
        return irr_mean_ci_tuple

    reporting_curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["reporting_currency"].get(symbol,config["reporting_currency"]["default"]),dt)
    price_curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["price_currency"].get(symbol,config["price_currency"]["default"]),dt)

    ###################################################
    eps_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*4+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_eps_dict.get(symbol,[])))

    ###################################################
    # annualization factor
    ###################################################
    eps_dt_list = sorted(list(set(map(lambda x: x[0], eps_list))))
    if debug_mode:
        print "eps_dt_list: %s" % eps_dt_list
    annualization_factor = get_annualization_factor(eps_dt_list)
    ###################################################

    ###################################################
    if len(eps_list) >= 5:
        eps_val_list = map(lambda x: x[1], eps_list)
        eps_chg_list = map(lambda x: x[1]-x[0], zip(eps_val_list[:-1],eps_val_list[1:]))
        eps_chg_list = sorted(eps_chg_list)[1:][:-1]

        m = sum(eps_chg_list)/len(eps_chg_list)
        sd = np.std(np.asarray(eps_chg_list))

        ann_sd = math.sqrt(annualization_factor) * sd

        if debug_mode:
            print "eps_val_list: %s" % eps_val_list
            print "annualization_factor: %s" % annualization_factor
            print "m: %s" % m
            print "sd: %s" % sd
            print "ann_sd: %s" % ann_sd
    else:
        ann_sd = 0.0

    if len(eps_list) > 0:
        five_yr_dt_bound = eps_list[-1][0] - timedelta(weeks = int(52*5))
        fil_eps_list = map(lambda x: x[1], filter(lambda x: x[0] >= five_yr_dt_bound, eps_list))
        annualized_avg_eps = sum(fil_eps_list) / len(fil_eps_list) * annualization_factor
    else:
        annualized_avg_eps = 0.0

    cur_eps = annualized_avg_eps

    if debug_mode:
        print "cur_eps: %s" % cur_eps

    next_eps_sd = cur_eps * ann_sd_sym_rtn
    # next_eps_sd = ann_sd
    irr_mean_ci_tuple = cal_irr_mean_ci_MonteCarlo_1D(hist_unadj_px_dict[dt][symbol],
                                                   cur_eps,
                                                   next_eps_sd,
                                                   bps,
                                                   # 100,
                                                   10,
                                                   1000,
                                                   confidence_level)
    if irr_mean_ci_tuple is not None:
        irr_mean_ci_tuple = map(lambda x: adj_irr_by_cf_time(x, time_to_next_cf), irr_mean_ci_tuple)

    return irr_mean_ci_tuple

def remove_files_in_folder(prep_data_folder):
    for f in glob.glob(prep_data_folder+"/*"):
        # print "os.remove(%s)" % f
        os.remove(f)


def calc_irr_mean_cov_after_20170309_prep(config,prep_data_folder,dt,symbol_list,hist_bps_dict,hist_totliabps_dict,hist_eps_dict,hist_roa_dict,hist_outshr_dict,NUM_OF_MONTE_CARLO,num_of_fut_divd_periods,delay_months,debug_mode):
    def replace_date_with_YM(date_value_list):
        return map(lambda x: ((x[0].year,x[0].month),x[1]), date_value_list)

    def interpolate_values(sym_YM_val_list,annualization_factor_list,just_repeat_val=False):
        interpolated_YM_val_list = []
        for YM_val_list,annualization_factor in zip(sym_YM_val_list,annualization_factor_list):
            YM_hold_list = []
            sym_interpolated_YM_val_list = []
            for YM,val in YM_val_list:
                if val is not None:
                    if len(sym_interpolated_YM_val_list) == 0:
                        interpolated_val = val/(4.0/annualization_factor) if just_repeat_val == False else val
                        if annualization_factor == 2:
                            sym_interpolated_YM_val_list.extend(map(lambda ym: (ym,interpolated_val), YM_hold_list[-1:]))
                        elif annualization_factor == 1:
                            sym_interpolated_YM_val_list.extend(map(lambda ym: (ym,interpolated_val), YM_hold_list[-3:]))
                    else:
                        interpolated_val = val/float(len(YM_hold_list)+1.0) if just_repeat_val == False else val
                        sym_interpolated_YM_val_list.extend(map(lambda ym: (ym, interpolated_val), YM_hold_list))
                    sym_interpolated_YM_val_list.append((YM, interpolated_val))
                    YM_hold_list = []
                else:
                    YM_hold_list.append(YM)
            interpolated_YM_val_list.append(sym_interpolated_YM_val_list)
        return interpolated_YM_val_list

    def standardize_yearly_val(sym_YM_val_list,yr_end_month,just_repeat_val=False):
        yearly_val = []
        prev_val_list = []
        for YM,val in sym_YM_val_list:
            prev_val_list.append(val)
            if YM[1] == yr_end_month:
                val = sum(prev_val_list)*4.0/len(prev_val_list) if just_repeat_val == False else prev_val_list[-1]
                yearly_val.append((YM[0],val))
                prev_val_list = []
        # ###################################################
        # # remaining items
        # ###################################################
        # if len(prev_val_list) > 0 and len(yearly_val) > 0:
        #     n = len(prev_val_list)
        #     val_same_period_last_yr = sum(map(lambda x: x[1], sym_YM_val_list)[-4-n:-4])
        #     if abs(val_same_period_last_yr) > 0.0001:
        #         val_most_current = sum(prev_val_list)
        #         projected_val = val_most_current / val_same_period_last_yr * yearly_val[-1][1]
        #         projected_year = (yearly_val[-1][0])+1
        #         yearly_val.append((projected_year,projected_val))
        return yearly_val

    curcy_converter = CurrencyConverter(config["currency_rate"])
    ###################################################
    w_a_dict = {}
    w_e_dict = {}
    bv_rlzn_yr_dict = {}
    bv_rcvy_rate_dict = {}

    annualization_factor_dict = {}

    ###################################################
    # annualization factor
    ###################################################
    annualization_factor_dict = dict(map(lambda sym: (sym,get_annualization_factor(map(lambda x: x[0], hist_bps_dict.get(sym,[])))), symbol_list))
    if debug_mode:
        print "annualization_factor: %s" % annualization_factor_dict
    ###################################################

    ###################################################
    for sym in symbol_list:
        sym_indus_grp = config["industry_group"].get(sym,config["industry_group"]["default"])
        w_ig_list = []
        try:
            w_ig_list.append((1.0,str(int(sym_indus_grp))))
        except Exception, e:
            w_ig_list.extend(map(lambda x: (float(x.split(':')[1]),str(x.split(':')[0])), sym_indus_grp))

        w_a_w_e_bv_list = []
        for w,ig in w_ig_list:
            w_a_w_e_bv_list.append(tuple(map(lambda x: w*x, map(float, config["expected_rtn_ast_ext_bvrlzn_bvrcvy"].get(ig, config["expected_rtn_ast_ext_bvrlzn_bvrcvy"]["0"])))))

        w_a_dict[sym]           = sum(map(lambda x: x[0], w_a_w_e_bv_list))
        w_e_dict[sym]           = sum(map(lambda x: x[1], w_a_w_e_bv_list))
        bv_rlzn_yr_dict[sym]    = sum(map(lambda x: x[2], w_a_w_e_bv_list))
        bv_rcvy_rate_dict[sym]  = sum(map(lambda x: x[3], w_a_w_e_bv_list))
        if debug_mode:
            print "sym config: %s" % ' '.join(map(str, [sym, w_a_dict[sym], w_e_dict[sym], bv_rlzn_yr_dict[sym], bv_rcvy_rate_dict[sym]]))
    ###################################################

    reporting_YM_list = [j for i in map(lambda y: map(lambda m: (1980+y,m), [3,6,9,12]), range(50)) for j in i]
    last_fundl_avb_date = shift_back_n_months(dt,delay_months)
    last_fundl_avb_YM = (last_fundl_avb_date.year,last_fundl_avb_date.month)
    reporting_YM_list = filter(lambda ym: ym <= last_fundl_avb_YM, reporting_YM_list)

    sym_aligned_roa_list = map(lambda sym: map(lambda ym: next(iter(filter(lambda x: x[0] == ym, replace_date_with_YM(hist_roa_dict.get(sym,[])))),(ym,None)), reporting_YM_list), symbol_list)

    sym_aligned_eps_list = map(lambda sym: map(lambda ym: next(iter(filter(lambda x: x[0] == ym, replace_date_with_YM(adj_val_per_share_with_no_of_shares(hist_eps_dict.get(sym,[]),hist_outshr_dict.get(sym,[]))))),(ym,None)), reporting_YM_list), symbol_list)
    sym_aligned_bps_list = map(lambda sym: map(lambda ym: next(iter(filter(lambda x: x[0] == ym, replace_date_with_YM(adj_val_per_share_with_no_of_shares(hist_bps_dict.get(sym,[]),hist_outshr_dict.get(sym,[]))))),(ym,None)), reporting_YM_list), symbol_list)
    sym_aligned_totliabps_list = map(lambda sym: map(lambda ym: next(iter(filter(lambda x: x[0] == ym, replace_date_with_YM(adj_val_per_share_with_no_of_shares(hist_totliabps_dict.get(sym,[]),hist_outshr_dict.get(sym,[]))))),(ym,None)), reporting_YM_list), symbol_list)
    # if debug_mode:
    #     print "sym_aligned_eps_list: %s" % (": ".join(map(str, zip(symbol_list,sym_aligned_eps_list))))

    ###################################################
    # ROA from Bloomberg has to be divided by 100
    ###################################################
    sym_aligned_roa_list = map(lambda x: map(lambda y: (y[0],y[1]/100.0 if y[1] is not None else None), x), sym_aligned_roa_list)
    interpolated_YM_roa_list = interpolate_values(sym_aligned_roa_list,map(lambda s: annualization_factor_dict[s],symbol_list),True)
    interpolated_YM_bps_list = interpolate_values(sym_aligned_bps_list,map(lambda s: annualization_factor_dict[s],symbol_list),True)
    interpolated_YM_totliabps_list = interpolate_values(sym_aligned_totliabps_list,map(lambda s: annualization_factor_dict[s],symbol_list),True)
    interpolated_YM_eps_list = interpolate_values(sym_aligned_eps_list,map(lambda s: annualization_factor_dict[s],symbol_list))
    if debug_mode:
        print "interpolated_YM_eps_list: %s" % (": ".join(map(str, zip(symbol_list,interpolated_YM_eps_list)[0])))
        print "interpolated_YM_bps_list: %s" % (": ".join(map(str, zip(symbol_list,interpolated_YM_bps_list)[0])))
        print "interpolated_YM_totliabps_list: %s" % (": ".join(map(str, zip(symbol_list,interpolated_YM_totliabps_list)[0])))
        print "interpolated_YM_roa_list: %s" % (": ".join(map(str, zip(symbol_list,interpolated_YM_roa_list)[0])))

    ###################################################
    # to make sure we have data for that quarter during live mode / not to exclude those companies that announce results late
    # - final year = dt.date().year - 1
    # - shifting back 9 months
    ###################################################
    quarter_end_list = [(y,q) for y in map(lambda x: dt.year-x, range(2)) for q in [3,6,9,12]]
    dt_back_9_mth = shift_back_n_months(dt,9)
    dt_back_9_mth_ym = (dt_back_9_mth.year, dt_back_9_mth.month)
    final_year,stndzd_mth = max(filter(lambda x: x is not None, map(lambda x: x if x <= dt_back_9_mth_ym else None, quarter_end_list)))

    if debug_mode:
        print "dt: %s final_year: %s standardized month: %s" % (dt,final_year,stndzd_mth)

    YM_eps_stndzd_yr_end_list       = map(lambda eps_list:       standardize_yearly_val(eps_list,      stndzd_mth     ), interpolated_YM_eps_list)
    YM_roa_stndzd_yr_end_list       = map(lambda roa_list:       standardize_yearly_val(roa_list,      stndzd_mth,True), interpolated_YM_roa_list)
    YM_bps_stndzd_yr_end_list       = map(lambda bps_list:       standardize_yearly_val(bps_list,      stndzd_mth,True), interpolated_YM_bps_list)
    YM_totliabps_stndzd_yr_end_list = map(lambda totliabps_list: standardize_yearly_val(totliabps_list,stndzd_mth,True), interpolated_YM_totliabps_list)

    NUM_OF_YEARS = 6 # so 5 years of changes
    YM_eps_stndzd_yr_end_list = map(lambda sym_stndzd_list: map(lambda x: x[1], filter(lambda x: x[0]>(final_year-NUM_OF_YEARS), sym_stndzd_list)), YM_eps_stndzd_yr_end_list)
    YM_bps_stndzd_yr_end_list = map(lambda sym_stndzd_list: map(lambda x: x[1], filter(lambda x: x[0]>(final_year-NUM_OF_YEARS), sym_stndzd_list)), YM_bps_stndzd_yr_end_list)
    YM_totliabps_stndzd_yr_end_list = map(lambda sym_stndzd_list: map(lambda x: x[1], filter(lambda x: x[0]>(final_year-NUM_OF_YEARS), sym_stndzd_list)), YM_totliabps_stndzd_yr_end_list)
    YM_roa_stndzd_yr_end_list = map(lambda sym_stndzd_list: map(lambda x: x[1], filter(lambda x: x[0]>(final_year-NUM_OF_YEARS), sym_stndzd_list)), YM_roa_stndzd_yr_end_list)
    symbol_with_enough_eps_data_set       = set(filter(lambda x: x is not None, map(lambda x: x[0] if len(x[1]) == NUM_OF_YEARS else None, zip(symbol_list,YM_eps_stndzd_yr_end_list))))
    symbol_with_enough_roa_data_set       = set(filter(lambda x: x is not None, map(lambda x: x[0] if len(x[1]) == NUM_OF_YEARS else None, zip(symbol_list,YM_roa_stndzd_yr_end_list))))
    symbol_with_enough_bps_data_set       = set(filter(lambda x: x is not None, map(lambda x: x[0] if len(x[1]) == NUM_OF_YEARS else None, zip(symbol_list,YM_bps_stndzd_yr_end_list))))
    symbol_with_enough_totliabps_data_set = set(filter(lambda x: x is not None, map(lambda x: x[0] if len(x[1]) == NUM_OF_YEARS else None, zip(symbol_list,YM_totliabps_stndzd_yr_end_list))))
    symbol_with_enough_fundl_set = symbol_with_enough_eps_data_set.intersection(symbol_with_enough_bps_data_set).intersection(symbol_with_enough_totliabps_data_set).intersection(symbol_with_enough_roa_data_set)
    symbol_with_enough_fundl_list = sorted(filter(lambda s: s in symbol_with_enough_fundl_set, symbol_list))

    if debug_mode:
        print "symbol_with_enough_fundl_list: %s %s" % (dt,symbol_with_enough_fundl_list)

    ###################################################
    # return if no data to work with!
    ###################################################
    if len(symbol_with_enough_fundl_list) == 0:
        return None,None,None,None

    interpolated_YM_eps_list  = map(lambda x: x[1], filter(lambda x: x[0] in symbol_with_enough_fundl_set, zip(symbol_list,interpolated_YM_eps_list)))
    interpolated_YM_bps_list  = map(lambda x: x[1], filter(lambda x: x[0] in symbol_with_enough_fundl_set, zip(symbol_list,interpolated_YM_bps_list)))
    interpolated_YM_totliabps_list  = map(lambda x: x[1], filter(lambda x: x[0] in symbol_with_enough_fundl_set, zip(symbol_list,interpolated_YM_totliabps_list)))
    interpolated_YM_roa_list  = map(lambda x: x[1], filter(lambda x: x[0] in symbol_with_enough_fundl_set, zip(symbol_list,interpolated_YM_roa_list)))
    YM_eps_stndzd_yr_end_list = map(lambda x: x[1], filter(lambda x: x[0] in symbol_with_enough_fundl_set, zip(symbol_list,YM_eps_stndzd_yr_end_list)))
    YM_bps_stndzd_yr_end_list = map(lambda x: x[1], filter(lambda x: x[0] in symbol_with_enough_fundl_set, zip(symbol_list,YM_bps_stndzd_yr_end_list)))
    YM_totliabps_stndzd_yr_end_list = map(lambda x: x[1], filter(lambda x: x[0] in symbol_with_enough_fundl_set, zip(symbol_list,YM_totliabps_stndzd_yr_end_list)))
    YM_roa_stndzd_yr_end_list = map(lambda x: x[1], filter(lambda x: x[0] in symbol_with_enough_fundl_set, zip(symbol_list,YM_roa_stndzd_yr_end_list)))

    if debug_mode:
        print "YM_eps_stndzd_yr_end_list: %s" % (zip(symbol_with_enough_fundl_list,YM_eps_stndzd_yr_end_list)[:5])
        print "YM_bps_stndzd_yr_end_list: %s" % (zip(symbol_with_enough_fundl_list,YM_bps_stndzd_yr_end_list)[:5])
        print "YM_totliabps_stndzd_yr_end_list: %s" % (zip(symbol_with_enough_fundl_list,YM_totliabps_stndzd_yr_end_list)[:5])
        print "YM_roa_stndzd_yr_end_list: %s" % (zip(symbol_with_enough_fundl_list,YM_roa_stndzd_yr_end_list)[:5])

    YM_eps_chg_stndzd_yr_end_list = map(lambda eps_list: map(lambda y: y[0]-y[1], zip(eps_list[1:],eps_list[:-1])), YM_eps_stndzd_yr_end_list)
    YM_bps_chg_stndzd_yr_end_list = map(lambda bps_list: map(lambda y: y[0]-y[1], zip(bps_list[1:],bps_list[:-1])), YM_bps_stndzd_yr_end_list)
    YM_totliabps_chg_stndzd_yr_end_list = map(lambda totliabps_list: map(lambda y: y[0]-y[1], zip(totliabps_list[1:],totliabps_list[:-1])), YM_totliabps_stndzd_yr_end_list)
    YM_roa_chg_stndzd_yr_end_list = map(lambda roa_list: map(lambda y: y[0]-y[1], zip(roa_list[1:],roa_list[:-1])), YM_roa_stndzd_yr_end_list)

    eps_roa_chg_cov_matrix = np.cov(np.array(YM_eps_chg_stndzd_yr_end_list+YM_roa_chg_stndzd_yr_end_list))

    ###################################################
    # for error checking only
    ###################################################
    eps_cor_matrix = np.corrcoef(np.array(YM_eps_stndzd_yr_end_list))
    if debug_mode:
        print "symbol_with_enough_fundl_list: %s" % symbol_with_enough_fundl_list
        print "eps_cor_matrix: %s" % zip(symbol_with_enough_fundl_list,eps_cor_matrix.tolist()[0])
    ###################################################

    ###################################################
    NUM_OF_QUARTERS_FOR_CUR_EARG = 12
    reporting_curcy_conv_rate_list = map(lambda s: curcy_converter.get_conv_rate_to_hkd(config["reporting_currency"].get(s,config["reporting_currency"]["default"]),dt), symbol_with_enough_fundl_list)
    price_curcy_conv_rate_list = map(lambda s: curcy_converter.get_conv_rate_to_hkd(config["price_currency"].get(s,config["price_currency"]["default"]),dt), symbol_with_enough_fundl_list)
    cur_eps_list = map(lambda sym_YM_val_list: sum(map(lambda x: (NUM_OF_QUARTERS_FOR_CUR_EARG-x[0])*x[1], enumerate(map(lambda y: y[1], list(reversed(sym_YM_val_list))[:NUM_OF_QUARTERS_FOR_CUR_EARG])))) / (NUM_OF_QUARTERS_FOR_CUR_EARG*(NUM_OF_QUARTERS_FOR_CUR_EARG+1)/2.0) * 4.0  , interpolated_YM_eps_list)
    ###################################################
    # ROA from Bloomberg is already annualized
    ###################################################
    cur_roa_list = map(lambda sym_YM_val_list: sum(map(lambda x: (NUM_OF_QUARTERS_FOR_CUR_EARG-x[0])*x[1], enumerate(map(lambda y: y[1], list(reversed(sym_YM_val_list))[:NUM_OF_QUARTERS_FOR_CUR_EARG])))) / (NUM_OF_QUARTERS_FOR_CUR_EARG*(NUM_OF_QUARTERS_FOR_CUR_EARG+1)/2.0)        , interpolated_YM_roa_list)
    sym_bps_list = map(lambda sym_YM_val_list: sum(map(lambda x: (NUM_OF_QUARTERS_FOR_CUR_EARG-x[0])*x[1], enumerate(map(lambda y: y[1], list(reversed(sym_YM_val_list))[:NUM_OF_QUARTERS_FOR_CUR_EARG])))) / (NUM_OF_QUARTERS_FOR_CUR_EARG*(NUM_OF_QUARTERS_FOR_CUR_EARG+1)/2.0)        , interpolated_YM_bps_list)
    sym_totliabps_list = map(lambda sym_YM_val_list: sum(map(lambda x: (NUM_OF_QUARTERS_FOR_CUR_EARG-x[0])*x[1], enumerate(map(lambda y: y[1], list(reversed(sym_YM_val_list))[:NUM_OF_QUARTERS_FOR_CUR_EARG])))) / (NUM_OF_QUARTERS_FOR_CUR_EARG*(NUM_OF_QUARTERS_FOR_CUR_EARG+1)/2.0)  , interpolated_YM_totliabps_list)

    w_a_list = map(lambda s: w_a_dict[s], symbol_with_enough_fundl_list)
    w_e_list = map(lambda s: w_e_dict[s], symbol_with_enough_fundl_list)
    bv_rlzn_yr_list = map(lambda s: bv_rlzn_yr_dict[s], symbol_with_enough_fundl_list)
    bv_rcvy_rate_list = map(lambda s: bv_rcvy_rate_dict[s], symbol_with_enough_fundl_list)

    if debug_mode:
        print "reporting_curcy_conv_rate_list: %s" % zip(symbol_with_enough_fundl_list,reporting_curcy_conv_rate_list)
        print "price_curcy_conv_rate_list: %s" % zip(symbol_with_enough_fundl_list,price_curcy_conv_rate_list)
        print "cur_eps_list (original currency): %s" % zip(symbol_with_enough_fundl_list,cur_eps_list)
        print "cur_roa_list (original currency): %s" % zip(symbol_with_enough_fundl_list,cur_roa_list)
        print "sym_bps_list (original currency): %s" % zip(symbol_with_enough_fundl_list,sym_bps_list)
        print "sym_totliabps_list (original currency): %s" % zip(symbol_with_enough_fundl_list,sym_totliabps_list)

    ###################################################
    with open(prep_data_folder+"/symbol_with_enough_fundl.pkl", "wb") as symbol_with_enough_fundl_file:
         cPickle.dump(symbol_with_enough_fundl_list, symbol_with_enough_fundl_file)
    with open(prep_data_folder+"/w_a_dict.pkl", "wb") as w_a_dict_file:
         cPickle.dump(w_a_dict, w_a_dict_file)
    with open(prep_data_folder+"/w_e_dict.pkl", "wb") as w_e_dict_file:
         cPickle.dump(w_e_dict, w_e_dict_file)

    ###################################################
    # going concern
    ###################################################
    def goingconcern_montecarlo_prep(tup):
        n,num_montecarlo = tup
        fe = open(prep_data_folder+"/extnl_drvr_dvd_rlzn_path_samples"+"_"+str(n)+".csv", "wb")
        fa = open(prep_data_folder+"/asset_drvr_dvd_rlzn_path_samples"+"_"+str(n)+".csv", "wb")
        for _ in itertools.repeat(None, num_montecarlo):
            rand_matrix = np.random.multivariate_normal([0]*len(YM_eps_chg_stndzd_yr_end_list+YM_roa_chg_stndzd_yr_end_list), eps_roa_chg_cov_matrix, num_of_fut_divd_periods).T.tolist()
            extnl_drvr_dvd_rlzn_path_list = map(lambda x: get_divd_rlzn_external_driver(x[0],x[1],x[2]), zip(cur_eps_list,rand_matrix[:len(rand_matrix)/2],sym_bps_list))
            asset_drvr_dvd_rlzn_path_list = map(lambda x: get_divd_rlzn_asset_driver(x[0],x[1],x[2],x[3]), zip(cur_roa_list,rand_matrix[len(rand_matrix)/2:],sym_bps_list,sym_totliabps_list))
            fe.write('|'.join(map(lambda x: ','.join(map(str,x)), extnl_drvr_dvd_rlzn_path_list)))
            fe.write('\n')
            fa.write('|'.join(map(lambda x: ','.join(map(str,x)), asset_drvr_dvd_rlzn_path_list)))
            fa.write('\n')
        fe.close()
        fa.close()

    def liquidation_montecarlo_prep(tup):
        n,num_montecarlo = tup
        fliq = open(prep_data_folder+"/liq_drvr_dvd_rlzn_path_samples"+"_"+str(n)+".csv", "wb")
        for _ in itertools.repeat(None, num_montecarlo):
            cash_flow_list = map(lambda x: [0]*int(max(random.randint(0,2*int(x[0]))-1,0)) + [np.random.uniform(max(1.0-2*(1.0-x[1]),0.001),0.999)*max(x[2],0)], zip(bv_rlzn_yr_list,bv_rcvy_rate_list,sym_bps_list))
            fliq.write('|'.join(map(lambda x: ','.join(map(str,x)), cash_flow_list)))
            fliq.write('\n')
        fliq.close()

    ###################################################
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("gmail.com",80))
    local_ip = (s.getsockname()[0])
    s.close()
    num_of_jobs = int(mp.cpu_count()*float(dict(map(lambda x: x.split(':'), config["general"]["percentage_of_cpu_cores_to_use"]))[local_ip]))
    num_of_files = NUM_OF_MONTE_CARLO/500
    ###################################################
    with Pool(num_of_jobs) as p:
        p.map(goingconcern_montecarlo_prep, map(lambda x: (x,int(NUM_OF_MONTE_CARLO/num_of_files)), range(num_of_files)))
    with Pool(num_of_jobs) as p:
        p.map(liquidation_montecarlo_prep, map(lambda x: (x,int(NUM_OF_MONTE_CARLO/num_of_files)), range(num_of_files)))

def calc_irr_mean_cov_after_20170309_live(config,prep_data_folder,dt,symbol_with_enough_fundl_list,hist_unadj_px_dict,debug_mode):

    curcy_converter = CurrencyConverter(config["currency_rate"])
    reporting_curcy_conv_rate_list = map(lambda s: curcy_converter.get_conv_rate_to_hkd(config["reporting_currency"].get(s,config["reporting_currency"]["default"]),dt), symbol_with_enough_fundl_list)
    price_curcy_conv_rate_list = map(lambda s: curcy_converter.get_conv_rate_to_hkd(config["price_currency"].get(s,config["price_currency"]["default"]),dt), symbol_with_enough_fundl_list)
    # print "reporting_curcy_conv_rate_list: %s" % zip(symbol_with_enough_fundl_list,reporting_curcy_conv_rate_list)
    # print "price_curcy_conv_rate_list: %s" % zip(symbol_with_enough_fundl_list,price_curcy_conv_rate_list)

    sym_hist_unadj_px_list = map(lambda x: hist_unadj_px_dict[dt][x[0]]*x[2]/x[1] if x[0] in hist_unadj_px_dict[dt] else None, zip(symbol_with_enough_fundl_list,reporting_curcy_conv_rate_list,price_curcy_conv_rate_list))
    if debug_mode:
        print "sym_hist_unadj_px_list (reporting currency): %s" % zip(symbol_with_enough_fundl_list,reporting_curcy_conv_rate_list,price_curcy_conv_rate_list,sym_hist_unadj_px_list)

    ###################################################
    with open(prep_data_folder+"/w_a_dict.pkl", "rb") as w_a_dict_file:
         w_a_dict = cPickle.load(w_a_dict_file)
    with open(prep_data_folder+"/w_e_dict.pkl", "rb") as w_e_dict_file:
         w_e_dict = cPickle.load(w_e_dict_file)
    w_a_list = map(lambda s: w_a_dict[s], symbol_with_enough_fundl_list)
    w_e_list = map(lambda s: w_e_dict[s], symbol_with_enough_fundl_list)

    ###################################################
    # Monte Carlo (going concern)
    ###################################################
    extnl_drvr_dvd_rlzn_path_sample_file_list = sorted(filter(lambda x: fnmatch.fnmatch(x, "extnl_drvr_*.csv"), os.listdir(prep_data_folder)))
    asset_drvr_dvd_rlzn_path_sample_file_list = sorted(filter(lambda x: fnmatch.fnmatch(x, "asset_drvr_*.csv"), os.listdir(prep_data_folder)))
    # print zip(extnl_drvr_dvd_rlzn_path_sample_file_list,asset_drvr_dvd_rlzn_path_sample_file_list)

    def goingconcern_irr_part(tup):
        extnl_drvr_dvd_rlzn_path_sample_file,asset_drvr_dvd_rlzn_path_sample_file = tup

        irr_goingconcern_part_list = []
        fe = open(prep_data_folder+'/'+extnl_drvr_dvd_rlzn_path_sample_file,'r')
        fa = open(prep_data_folder+'/'+asset_drvr_dvd_rlzn_path_sample_file,'r')

        while True:
            fe_line = fe.readline().strip()
            fa_line = fa.readline().strip()

            if not fe_line or not fa_line:
                break

            extnl_drvr_dvd_rlzn_path_list = map(lambda x: map(lambda x: float(x), x.split(',')), fe_line.split('|'))
            asset_drvr_dvd_rlzn_path_list = map(lambda x: map(lambda x: float(x), x.split(',')), fa_line.split('|'))

            # print "len sym_hist_unadj_px_list: %s" % len(sym_hist_unadj_px_list)
            # print "len extnl_drvr_dvd_rlzn_path_list: %s" % len(extnl_drvr_dvd_rlzn_path_list)
            # print datetime.now()
            # sym_irr_extnl_drvr_list = map(lambda x: np.irr([-x[0]] + x[1]) if (x[0] > 0.001 and len(x[1]) > 0) else -1.0, zip(sym_hist_unadj_px_list,extnl_drvr_dvd_rlzn_path_list))
            # sym_irr_extnl_drvr_list = map(lambda x: np.irr([-x[0]] + x[1]), zip(sym_hist_unadj_px_list,extnl_drvr_dvd_rlzn_path_list))
            sym_irr_extnl_drvr_list = map(lambda x: irr_newton([-x[0]] + x[1]), zip(sym_hist_unadj_px_list,extnl_drvr_dvd_rlzn_path_list))
            sym_irr_extnl_drvr_list = map(lambda x: -1.0 if math.isnan(x) else x, sym_irr_extnl_drvr_list)

            # sym_irr_asset_drvr_list = map(lambda x: np.irr([-x[0]] + x[1]) if (x[0] > 0.001 and len(x[1]) > 0) else -1.0, zip(sym_hist_unadj_px_list,asset_drvr_dvd_rlzn_path_list))
            # sym_irr_asset_drvr_list = map(lambda x: np.irr([-x[0]] + x[1]), zip(sym_hist_unadj_px_list,asset_drvr_dvd_rlzn_path_list))
            sym_irr_asset_drvr_list = map(lambda x: irr_newton([-x[0]] + x[1]), zip(sym_hist_unadj_px_list,asset_drvr_dvd_rlzn_path_list))
            sym_irr_asset_drvr_list = map(lambda x: -1.0 if math.isnan(x) else x, sym_irr_asset_drvr_list)

            ###################################################
            # weighted by business nature
            ###################################################
            irr_goingconcern_part_list.append((map(lambda x: x[0]*x[1]+x[2]*x[3], zip(w_e_list,sym_irr_extnl_drvr_list,w_a_list,sym_irr_asset_drvr_list)),sym_irr_extnl_drvr_list,sym_irr_asset_drvr_list))

        fe.close()
        fa.close()
        return irr_goingconcern_part_list


    ###################################################
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("gmail.com",80))
    local_ip = (s.getsockname()[0])
    s.close()
    num_of_jobs = int(mp.cpu_count()*float(dict(map(lambda x: x.split(':'), config["general"]["percentage_of_cpu_cores_to_use"]))[local_ip]))
    ###################################################

    # dvd_rlzn_sample_file_list = zip(extnl_drvr_dvd_rlzn_path_sample_file_list,asset_drvr_dvd_rlzn_path_sample_file_list)
    # irr_goingconcern_list = []
    # while len(dvd_rlzn_sample_file_list) > 0:
    #     ele = num_of_jobs*2
    #     with Pool(num_of_jobs) as p:
    #         p_list = p.map(goingconcern_irr_part, dvd_rlzn_sample_file_list[:ele])
    #         irr_goingconcern_list.extend(p_list)
    #         dvd_rlzn_sample_file_list = dvd_rlzn_sample_file_list[ele:]
    # irr_goingconcern_sample_list = [j for i in map(lambda x: map(lambda y: y[0], x), irr_goingconcern_list) for j in i]
    # irr_extnl_drvr_sample_list = [j for i in map(lambda x: map(lambda y: y[1], x), irr_goingconcern_list) for j in i]
    # irr_asset_drvr_sample_list = [j for i in map(lambda x: map(lambda y: y[2], x), irr_goingconcern_list) for j in i]

    if debug_mode:
        print "Start calculating going concern IRR... %s" % (datetime.now())
    with Pool(num_of_jobs) as p:
        irr_goingconcern_list = [j for i in p.map(goingconcern_irr_part, zip(extnl_drvr_dvd_rlzn_path_sample_file_list,asset_drvr_dvd_rlzn_path_sample_file_list)) for j in i]
    irr_goingconcern_sample_list = map(lambda x: x[0], irr_goingconcern_list)
    irr_extnl_drvr_sample_list = map(lambda x: x[1], irr_goingconcern_list)
    irr_asset_drvr_sample_list = map(lambda x: x[2], irr_goingconcern_list)
    if debug_mode:
        print "Finished calculating going concern IRR... %s" % (datetime.now())


    # print irr_goingconcern_sample_list

    ###################################################
    if debug_mode:
        irr_goingconcern_corrcoef = np.corrcoef(np.array(irr_goingconcern_sample_list).T).tolist()
        print "len(irr_goingconcern_corrcoef): %s" % len(irr_goingconcern_corrcoef)
        print "len(irr_goingconcern_corrcoef[0]): %s" % len(irr_goingconcern_corrcoef[0])
        print "irr_goingconcern_corrcoef: %s" % (zip(symbol_with_enough_fundl_list,irr_goingconcern_corrcoef[0]))
    ###################################################

    # irr_goingconcern_mean_list = np.mean(np.array(irr_goingconcern_sample_list).T, axis=1).tolist()
    # irr_extnl_drvr_mean_list = np.mean(np.array(irr_extnl_drvr_sample_list).T, axis=1).tolist()
    # irr_asset_drvr_mean_list = np.mean(np.array(irr_asset_drvr_sample_list).T, axis=1).tolist()
    irr_goingconcern_mean_list = np.mean(np.array(irr_goingconcern_sample_list), axis=0).tolist()
    irr_extnl_drvr_mean_list = np.mean(np.array(irr_extnl_drvr_sample_list), axis=0).tolist()
    irr_asset_drvr_mean_list = np.mean(np.array(irr_asset_drvr_sample_list), axis=0).tolist()

    ###################################################
    # Monte Carlo (liquidation or M&A)
    ###################################################
    liq_drvr_dvd_rlzn_path_sample_file_list = sorted(filter(lambda x: fnmatch.fnmatch(x, "liq_drvr_*.csv"), os.listdir(prep_data_folder)))
    # print liq_drvr_dvd_rlzn_path_sample_file_list

    def liquidation_irr_part(liq_drvr_dvd_rlzn_path_sample_file):
        irr_liq_part_list = []
        fliq = open(prep_data_folder+'/'+liq_drvr_dvd_rlzn_path_sample_file,'r')

        while True:
            fliq_line = fliq.readline().strip()
            if not fliq_line:
                break
            liq_drvr_dvd_rlzn_path_list = map(lambda x: map(lambda x: float(x), x.split(',')), fliq_line.split('|'))
            # sym_irr_liq_drvr_list = map(lambda x: np.irr([-x[0]] + x[1]) if (x[0] > 0.001 and len(x[1]) > 0) else -1.0, zip(sym_hist_unadj_px_list,liq_drvr_dvd_rlzn_path_list))
            # sym_irr_liq_drvr_list = map(lambda x: np.irr([-x[0]] + x[1]), zip(sym_hist_unadj_px_list,liq_drvr_dvd_rlzn_path_list))
            sym_irr_liq_drvr_list = map(lambda x: math.exp(math.log(x[1][-1]/x[0])/len(x[1]))-1.0 if (x[1][-1] > 0.001 and x[0] > 0.001) else -1.0, zip(sym_hist_unadj_px_list,liq_drvr_dvd_rlzn_path_list))
            sym_irr_liq_drvr_list = map(lambda x: -1.0 if math.isnan(x) else x, sym_irr_liq_drvr_list)
            irr_liq_part_list.append(sym_irr_liq_drvr_list)

        fliq.close()
        return irr_liq_part_list


    # dvd_rlzn_sample_file_list = liq_drvr_dvd_rlzn_path_sample_file_list
    # irr_liquidation_sample_list = []
    # while len(dvd_rlzn_sample_file_list) > 0:
    #     ele = num_of_jobs*2
    #     with Pool(num_of_jobs) as p:
    #         p_list = p.map(liquidation_irr_part, dvd_rlzn_sample_file_list[:ele])
    #         irr_liquidation_sample_list.extend(p_list)
    #         dvd_rlzn_sample_file_list = dvd_rlzn_sample_file_list[ele:]
    # irr_liquidation_sample_list = [j for i in irr_liquidation_sample_list for j in i]

    if debug_mode:
        print "Start calculating liquidation IRR... %s" % (datetime.now())
    with Pool(num_of_jobs) as p:
        irr_liquidation_sample_list = [j for i in p.map(liquidation_irr_part, liq_drvr_dvd_rlzn_path_sample_file_list) for j in i]
    if debug_mode:
        print "Finished calculating liquidation IRR... %s" % (datetime.now())


    # if debug_mode:
    #     print "irr_liquidation_sample_list: %s" % irr_liquidation_sample_list

    ###################################################
    if debug_mode:
        irr_liquidation_corrcoef = np.corrcoef(np.array(irr_liquidation_sample_list).T).tolist()
        print "len(irr_liquidation_corrcoef): %s" % len(irr_liquidation_corrcoef)
        print "len(irr_liquidation_corrcoef[0]): %s" % len(irr_liquidation_corrcoef[0])
        # print "irr_liquidation_corrcoef: %s" % (zip(symbol_with_enough_fundl_list,irr_liquidation_corrcoef[0]))
    ###################################################

    # irr_liquidation_mean_list = np.mean(np.array(irr_liquidation_sample_list).T, axis=1).tolist()
    irr_liquidation_mean_list = np.mean(np.array(irr_liquidation_sample_list), axis=0).tolist()

    ###################################################
    # Combine the results from various earnings drivers
    ###################################################
    choice_list = map(lambda x: x[0] >= x[1], zip(irr_goingconcern_mean_list,irr_liquidation_mean_list))

    irr_goingconcern_sample_list = (np.array(irr_goingconcern_sample_list).T).tolist()
    irr_liquidation_sample_list = (np.array(irr_liquidation_sample_list).T).tolist()

    irr_combined_sample_list = map(lambda x: x[1] if x[0] == True else x[2], zip(choice_list,irr_goingconcern_sample_list,irr_liquidation_sample_list))

    irr_combined_mean_list = np.mean(np.array(irr_combined_sample_list), axis=1).tolist()
    irr_combined_cov_matrix = np.cov(irr_combined_sample_list).tolist()

    if debug_mode:
        print "irr_extnl_drvr_mean_list irr_asset_drvr_mean_list irr_goingconcern_mean_list irr_liquidation_mean_list irr_combined_mean_list"
        print '\n'.join(map(str, zip(symbol_with_enough_fundl_list,irr_extnl_drvr_mean_list,irr_asset_drvr_mean_list,irr_goingconcern_mean_list,irr_liquidation_mean_list,irr_combined_mean_list)))

    ###################################################
    # output IRR
    ###################################################
    with open(prep_data_folder+"/k_samples_combined.csv", "w") as k_sample_file:
        k_sample_file.write('\n'.join(map(lambda x: x[0]+':'+','.join(map(str,x[1])), zip(symbol_with_enough_fundl_list,irr_combined_sample_list))))
    with open(prep_data_folder+"/k_samples_extnl_drvr.csv", "w") as k_sample_file:
        k_sample_file.write('\n'.join(map(lambda x: x[0]+':'+','.join(map(str,x[1])), zip(symbol_with_enough_fundl_list,irr_extnl_drvr_sample_list))))
    with open(prep_data_folder+"/k_samples_asset_drvr.csv", "w") as k_sample_file:
        k_sample_file.write('\n'.join(map(lambda x: x[0]+':'+','.join(map(str,x[1])), zip(symbol_with_enough_fundl_list,irr_asset_drvr_sample_list))))
    with open(prep_data_folder+"/k_samples_liquidation.csv", "w") as k_sample_file:
        k_sample_file.write('\n'.join(map(lambda x: x[0]+':'+','.join(map(str,x[1])), zip(symbol_with_enough_fundl_list,irr_liquidation_sample_list))))
    ###################################################

    irr_combined_ci_list = []
    for sym_sample_list in irr_combined_sample_list:
        sorted_sample_list = sorted(sym_sample_list)
        n = len(sorted_sample_list)
        irr_combined_ci_list.append((sorted_sample_list[int(float(n)*0.05)],sorted_sample_list[int(float(n)*0.95)]))

    ###################################################
    return irr_combined_mean_list,irr_combined_cov_matrix,irr_combined_ci_list


def preprocess_industry_groups(industry_group_dict):
    industry_group_list = []
    for k,v in sorted(industry_group_dict.items(),key=lambda x: x[0]):
        try:
            i = int(v)
            industry_group_list.append((k,i))
        except Exception, e:
            industry_group_list.extend(map(lambda x: (k,int(x.split(":")[0])), v))
    return map(lambda x: (x[0],str(x[1])), filter(lambda x: x[1] > 0, industry_group_list)) + map(lambda x: (x[1],str(x[0]+300)), enumerate(map(lambda x: x[0], filter(lambda x: x[1] == 0, industry_group_list))))

def get_industry_groups(industry_group_list):
    industry_groups_set_list = []
    for grp, it_lstup in itertools.groupby(sorted(industry_group_list, key=lambda x: x[1]), lambda x: x[1]):
        industry_groups_set_list.append(set(map(lambda x: x[0], list(it_lstup))))
    return industry_groups_set_list

###################################################
def get_port_and_hdg_cov_matrix(aug_cov_matrix,sol_list,hedging_symbol_list):
    # print "aug_cov_matrix: %s" % aug_cov_matrix
    # print "sol_list: %s" % sol_list
    h_n = len(hedging_symbol_list)
    s_n = len(sol_list)
    m = aug_cov_matrix
    for i in range(h_n):
        m = np.delete(m, 0, 0)
        m = np.delete(m, 0, 1)

    sol_vec = np.asarray(sol_list)
    sol_vec_T = np.matrix(sol_vec).T

    var_port = float((sol_vec * m) * sol_vec_T)
    sd_port = math.sqrt(var_port)
    # print "sd_port: %s" % sd_port

    cov_port_hedge_list = []
    for h in range(h_n):
        cov_port_hedge_list.append(sum(map(lambda x: float(x[0])*float(x[1]), zip(aug_cov_matrix.tolist()[h][h_n:],sol_list))))

    sd1=math.sqrt(aug_cov_matrix.tolist()[0][0])
    # print "sd(hedge): %s" % sd1
    # print "correl_port_hedge: %s" % (cov_port_hedge_list[0]/sd1/sd_port)
    cov_matrix_hedge = map(lambda x: x[:-s_n], aug_cov_matrix.tolist()[:-s_n])

    cov_matrix_port_hedge = []
    cov_matrix_port_hedge.append([var_port] + cov_port_hedge_list)

    for i in range(h_n):
        cov_matrix_port_hedge.append([cov_port_hedge_list[i]] + cov_matrix_hedge[i])

    return np.asarray(cov_matrix_port_hedge)


def risk_adj_rtn_hedge(expected_rtn_list,cov_matrix,sharpe_risk_aversion_factor):
    def iif(cond, iftrue=1.0, iffalse=0.0):
        if cond:
            return iftrue
        else:
            return iffalse

    n = len(expected_rtn_list)-1

    ###################################################
    # P and q determine the objective function to minimize
    # which in cvxopt is defined as $.5 x^T P x + q^T x$
    P = cvxopt.matrix(2.0 * sharpe_risk_aversion_factor * cov_matrix)
    q = cvxopt.matrix(map(lambda x: -x, expected_rtn_list))

    ###################################################
    # G and h determine the inequality constraints in the
    # form $G x \leq h$. We write $w_i \geq 0$ as $-1 \times x_i \leq 0$
    # and also add a (superfluous) $x_i \leq 1$ constraint
    ###################################################
    # G x <= h
    ###################################################
    G = cvxopt.matrix(
                [[( 1.0 if j==i else 0.0) for j in range(n+1)] for i in range(n+1)] +
                [[(-1.0 if j==i else 0.0) for j in range(n+1)] for i in range(n+1)]
                ).trans()
    h = cvxopt.matrix([1.0] + n * [0.0] + [-1.0] + n * [1.0])
    ###################################################

    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h)
    except:
        return None

    if sol['status'] != 'optimal':
        return None

    return list(sol['x'])


def log_optimal_hedge(expected_rtn_list,cov_matrix):
    def iif(cond, iftrue=1.0, iffalse=0.0):
        if cond:
            return iftrue
        else:
            return iffalse

    n = len(expected_rtn_list)-1

    ###################################################
    # P and q determine the objective function to minimize
    # which in cvxopt is defined as $.5 x^T P x + q^T x$
    P = cvxopt.matrix(cov_matrix)
    q = cvxopt.matrix(map(lambda x: -x, expected_rtn_list))

    ###################################################
    # G and h determine the inequality constraints in the
    # form $G x \leq h$. We write $w_i \geq 0$ as $-1 \times x_i \leq 0$
    # and also add a (superfluous) $x_i \leq 1$ constraint
    ###################################################
    # G x <= h
    ###################################################
    G = cvxopt.matrix(
                [[( 1.0 if j==i else 0.0) for j in range(n+1)] for i in range(n+1)] +
                [[(-1.0 if j==i else 0.0) for j in range(n+1)] for i in range(n+1)]
                ).trans()
    h = cvxopt.matrix([1.0] + n * [0.0] + [-1.0] + n * [1.0])
    ###################################################

    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h)
    except:
        return None

    if sol['status'] != 'optimal':
        return None

    return list(sol['x'])

def sharpe_hedge(expected_rtn_list,cov_matrix):
    def iif(cond, iftrue=1.0, iffalse=0.0):
        if cond:
            return iftrue
        else:
            return iffalse

    n = len(expected_rtn_list)-1

    ###################################################
    # P and q determine the objective function to minimize
    # which in cvxopt is defined as $.5 x^T P x + q^T x$

    ###################################################
    P = cvxopt.matrix(2.0 * cov_matrix)
    q = cvxopt.matrix([0.0 for i in range(n+1)])
    ###################################################
    # G x <= h
    ###################################################
    b1_list = [ 1.0] + n * [0.0]
    b2_list = [-1.0] + n * [1.0]
    G = cvxopt.matrix(
                [[(( 1.0 if j==i else 0.0) - b1_list[i]) for j in range(n+1)] for i in range(n+1)] +
                [[((-1.0 if j==i else 0.0) - b2_list[i]) for j in range(n+1)] for i in range(n+1)]
                ).trans()
    h = cvxopt.matrix([0.0]*(2*(n+1)))

    ###################################################
    # A and b determine the equality constraints defined as A x = b
    ###################################################
    A = cvxopt.matrix( [expected_rtn_list] ).trans()
    b = cvxopt.matrix([ 1.0 ])

    ###################################################
    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except:
        return None

    if sol['status'] != 'optimal':
        return None

    ###################################################
    # transform back to the original space
    ###################################################
    y_list = list(sol['x'])
    sy = sum(y_list)
    return map(lambda y: y / sy, y_list)

def minvar_hedge(expected_rtn_list,cov_matrix):
    n = len(expected_rtn_list)-1

    ###################################################
    # P and q determine the objective function to minimize
    # which in cvxopt is defined as $.5 x^T P x + q^T x$
    P = cvxopt.matrix(cov_matrix)
    q = cvxopt.matrix([0.0] * (n+1))

    ###################################################
    # G x <= h
    ###################################################
    G = cvxopt.matrix(
                [[( 1.0 if j==i else 0.0) for j in range(n+1)] for i in range(n+1)] +
                [[(-1.0 if j==i else 0.0) for j in range(n+1)] for i in range(n+1)]
                ).trans()
    h = cvxopt.matrix([1.0] + n * [0.0] + [-1.0] + n * [1.0])
    ###################################################

    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h)
    except:
        return None

    if sol['status'] != 'optimal':
        return None

    return list(sol['x'])

if __name__ == "__main__":

    ###################################################
    # test get_divd_rlzn_external_driver
    ###################################################
    irr_list = []
    while len(irr_list) < 10000:
        cur_px = 68.1
        cur_eps = 7.47
        rand_chg_list = np.random.normal(0, 20, 100).tolist()
        # rand_chg_list = [1.3565494538069889, -0.7328438048764819, -0.9911325496274774, 0.7216603971011765, -1.7010436939161022, -0.05459450695457012, 1.0595518908975685, 1.2771844239978554, 0.3906346389750634, -0.7567610690562201]
        bps = 102
        dvd_rlzn_path_list = get_divd_rlzn_external_driver(cur_eps,rand_chg_list,bps)
        # print "rand_chg_list: %s" % map(lambda x: round(x,3), rand_chg_list)
        # print "eps: %s" % [round(cur_eps+sum(rand_chg_list[:(i+1)]),3) for i in range(len(rand_chg_list))]
        # print "divd: %s" % map(lambda x: round(x,3), dvd_rlzn_path_list)
        irr = np.irr([-cur_px]+dvd_rlzn_path_list)

        if math.isnan(irr):
            irr_list.append(-1.0)
        else:
            irr_list.append(irr)

        # if not math.isnan(irr):
        #     irr_list.append(irr)

    print "mean irr: %s" % np.mean(irr_list)


    # ###################################################
    # # test get_divd_rlzn_asset_driver
    # ###################################################
    # irr_list = []
    # while len(irr_list) < 10000:
    #     cur_px = 68.1
    #     cur_roa = 0.049
    #     rand_chg_list = np.random.normal(0, 0.01, 100).tolist()
    #     # rand_chg_list = [-0.0076027477677689335, -0.035993399498083986, 0.05761109772056621, -0.049718955849337464, -0.11439030986747334, -0.07305513665411592, -0.06987823955241451, 0.028665312545241586, 0.0031546672341396225, 0.046790743318825205]
    #     bps = 100
    #     liabps = 40
    #     dvd_rlzn_path_list = get_divd_rlzn_asset_driver(cur_roa,rand_chg_list,bps,liabps)
    #     # print "rand_chg_list: %s" % map(lambda x: round(x,3), rand_chg_list)
    #     # print "divd: %s" % map(lambda x: round(x,3), dvd_rlzn_path_list)
    #     irr = np.irr([-cur_px]+dvd_rlzn_path_list)
    #     if math.isnan(irr):
    #         irr_list.append(-1.0)
    #     else:
    #         irr_list.append(irr)
    #
    # print "mean irr: %s" % np.mean(irr_list)
