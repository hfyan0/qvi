#!/usr/bin/env python
import sys
import math
import numpy as np
import cvxopt
from cvxopt import blas, solvers
from datetime import datetime, timedelta
import scipy.optimize
import itertools
import random
import time

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
        r_ls = calc_return_list(ts)[-LOOKBACK_DAYS:]
        return np.std(np.asarray(r_ls))

def get_annualization_factor(date_list):
    if len(date_list) < 3:
        return 0.0
    else:
        dl = sorted(map(lambda x: (x[0]-x[1]).days, zip(date_list[1:],date_list[:-1])))
        median_diff = (dl[(len(dl)/2)-1] + dl[(len(dl)/2)])/2.0
        if median_diff <= 4:
            return 252
        elif median_diff <= 7:
            return 52
        elif median_diff <= 35:
            return 12
        elif median_diff <= 30*3:
            return 4
        elif median_diff <= 35*6:
            return 2
        else:
            return 1

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
    annualized_adj_sd_list = map(lambda x: x[0]+x[1], zip(annualized_sd_list,specific_riskiness_list))
    D = np.diag(annualized_adj_sd_list)
    # print "D: %s" % str(D)
    # print "correl_matrix: %s" % str(correl_matrix)
    # print "cov_matrix: %s" % str(D*correl_matrix*D)
    return ((D * correl_matrix * D),annualized_sd_list,annualized_adj_sd_list)

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
    max_industry_weight_constraint_list = map(lambda x: max_weight_industry, industry_groups_list)

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

def log_optimal_growth(symbol_list,expected_rtn_list,cov_matrix,max_weight_list,industry_groups_list,max_weight_industry,portfolio_change_inertia=None,hatred_for_small_size=None,current_weight_list=None):
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
                ] +
                map(lambda ig_set: map(lambda s: 1.0 if (s in ig_set) else 0.0, symbol_list), industry_groups_list)
                ).trans()
    h = cvxopt.matrix([ max_weight_list[j/2] * iif(j % 2) for j in range(2*n) ] + map(lambda x: max_weight_industry, industry_groups_list))
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

def calc_expected_return_before_201703(config,dt,symbol_list,hist_bps_dict,hist_unadj_px_dict,hist_operincm_dict,hist_totasset_dict,hist_totliabps_dict,hist_costofdebt_dict,hist_stattaxrate_dict,hist_oper_eps_dict,hist_eps_dict,hist_roa_dict,delay_months,debug_mode):
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

        bps_list = map(lambda z: z[1], filter(lambda y: y[0] > shift_back_n_months(dt,5*12+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_bps_dict.get(sym,[]))))
        ###################################################
        if len(bps_list) >= 3 and sym in hist_unadj_px_dict[dt]:
            bps_r = calc_return_list(bps_list)
            m = sum(bps_r)/len(bps_r)
            sd = np.std(np.asarray(bps_r))
            conser_bp_ratio = max(reporting_curcy_conv_rate * bps_list[-1] * (1 + min(m,0.0) - float(config["general"]["bp_stdev"]) * sd) / price_curcy_conv_rate / hist_unadj_px_dict[dt][sym], 0.0)
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

            m = sum(oper_roa_list)/len(oper_roa_list)
            sd = np.std(np.asarray(oper_roa_list))
            conser_oper_roa = annualization_factor * (min(m,oper_roa_list[-1]) - float(config["general"]["roa_drvr_stdev"]) * sd)
            if debug_mode:
                print "oper_roa_list: %s" % oper_roa_list
                print "annualization_factor: %s" % annualization_factor
                print "m: %s" % m
                print "sd: %s" % sd
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
            m = sum(roa_list)/len(roa_list)
            sd = np.std(np.asarray(roa_list))
            ###################################################
            # Bloomberg's ROA is already annualized
            ###################################################
            conser_roa = (min(m,roa_list[-1]) - float(config["general"]["roa_drvr_stdev"]) * sd) / 100.0
            if debug_mode:
                print "roa_list (annualized %%): %s" % roa_list
                print "m: %s" % m
                print "sd: %s" % sd
                print "conser_roa (annualized): %s" % conser_roa
        else:
            conser_roa = 0.0

        bps_list = filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_bps_dict.get(sym,[]))
        bps = bps_list[-1][1] if len(bps_list) > 0 else 0.0

        costofdebt_list = filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_costofdebt_dict.get(sym,[]))
        totliabps_list = filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_totliabps_dict.get(sym,[]))
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
        oper_eps_chg_list = map(lambda x: x[0][1]-x[1][1], zip(oper_eps_list[1:],oper_eps_list[:-1]))
        if len(oper_eps_chg_list) >= 5:
            oper_eps_list = map(lambda z: z[1][1], sorted(sorted(enumerate(oper_eps_list), key=lambda x: x[1][1])[1:][:-1], key=lambda y: y[0]))
            oper_eps_chg_list = sorted(oper_eps_chg_list)[1:][:-1]
            m = sum(oper_eps_chg_list)/len(oper_eps_chg_list)
            sd = np.std(np.asarray(oper_eps_chg_list))
            conser_oper_eps = (oper_eps_list[-1] + min(m,0.0) - float(config["general"]["ext_drvr_stdev"]) * sd) * annualization_factor
            if debug_mode:
                print "oper_eps_list: %s" % oper_eps_list
                print "oper_eps_chg_list: %s" % oper_eps_chg_list
                print "annualization_factor: %s" % annualization_factor
                print "m: %s" % m
                print "sd: %s" % sd
                print "conser_oper_eps (annualized): %s" % conser_oper_eps
        else:
            conser_oper_eps = 0.0

        ###################################################
        # external driver: eps
        ###################################################
        eps_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*4+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_eps_dict.get(sym,[])))
        eps_chg_list = map(lambda x: x[0][1]-x[1][1], zip(eps_list[1:],eps_list[:-1]))
        if len(eps_chg_list) >= 5:
            eps_list = map(lambda z: z[1][1], sorted(sorted(enumerate(eps_list), key=lambda x: x[1][1])[1:][:-1], key=lambda y: y[0]))
            eps_chg_list = sorted(eps_chg_list)[1:][:-1]
            m = sum(eps_chg_list)/len(eps_chg_list)
            sd = np.std(np.asarray(eps_chg_list))
            conser_eps = (eps_list[-1] + min(m,0.0) - float(config["general"]["ext_drvr_stdev"]) * sd) * annualization_factor
            if debug_mode:
                print "eps_list: %s" % eps_list
                print "eps_chg_list: %s" % eps_chg_list
                print "annualization_factor: %s" % annualization_factor
                print "m: %s" % m
                print "sd: %s" % sd
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
        return list(set(find_all_roots_brentq(f, 0.001, 1, pars=(irr_with_1st_cf_in_1yr, time_b4_1st_cf))))[0]
    else:
        return list(set(find_all_roots_brentq(f, -1, -0.001, pars=(irr_with_1st_cf_in_1yr, time_b4_1st_cf))))[0]

def cal_irr_mean_ci_MonteCarlo(cur_px,next_eps,sigma,bps,num_terms,num_times_montecarlo,confidence_interval):
    irr_result_list = []
    for _ in itertools.repeat(None, num_times_montecarlo):
        rand_eps_chg_list = np.random.normal(0, sigma, num_terms).tolist()

        eps_rlzn_list = map(lambda x: next_eps + sum(rand_eps_chg_list[:x]), range(num_terms))

        dvd_rlzn_list = []
        for i in range(num_terms):
            if eps_rlzn_list[i] > 0.0:
                dvd = max(sum(eps_rlzn_list[:(i+1)]) - sum(dvd_rlzn_list), 0.0)
                dvd_rlzn_list.append(dvd)
            else:
                if (sum(eps_rlzn_list) - sum(dvd_rlzn_list) < bps):
                    break
                else:
                    dvd_rlzn_list.append(0.0)

        irr = np.irr([-cur_px] + dvd_rlzn_list)
        if not math.isnan(irr):
            irr_result_list.append(round(irr,5))
    sorted_irr_list = sorted(irr_result_list)
    if len(sorted_irr_list) < 30:
        return None,None,None
    ci_leftside_pctg = (100.0-confidence_interval)/2.0/100.0
    ci_rightside_pctg = 1.0 - ci_leftside_pctg

    irr_mean = round(sum(sorted_irr_list)/len(sorted_irr_list),5)
    # irr_median = sorted_irr_list[len(sorted_irr_list)/2]
    irr_5pctl = sorted_irr_list[int(len(sorted_irr_list)*ci_leftside_pctg)]
    irr_95pctl = sorted_irr_list[int(len(sorted_irr_list)*ci_rightside_pctg)]
    irr_true_mean = next_eps/cur_px
    irr_true_5pctl = irr_true_mean - (irr_mean - irr_5pctl)
    irr_true_95pctl = irr_true_mean - (irr_mean - irr_95pctl)
    return [irr_true_mean,irr_true_5pctl,irr_true_95pctl]

def calc_irr_mean_ci(config,dt,symbol,ann_sd_sym_rtn,hist_unadj_px_dict,hist_eps_dict,hist_bps_dict,delay_months,debug_mode):
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
    if len(eps_list) > 0:
        one_yr_dt_bound = eps_list[-1][0] - timedelta(weeks = float(52.0/12.0*13.0))
        cur_yr_eps = sum(map(lambda x: x[1], filter(lambda x: x[0] >= one_yr_dt_bound, eps_list)))
    else:
        cur_yr_eps = 0.0

    if len(eps_list) >= 5:
        eps_list = map(lambda x: x[1], eps_list)
        eps_chg_list = map(lambda x: x[1]-x[0], zip(eps_list[:-1],eps_list[1:]))
        eps_chg_list = sorted(eps_chg_list)[1:][:-1]

        m = sum(eps_chg_list)/len(eps_chg_list)
        sd = np.std(np.asarray(eps_chg_list))

        ann_sd = math.sqrt(annualization_factor) * sd

        if debug_mode:
            print "eps_list: %s" % eps_list
            print "annualization_factor: %s" % annualization_factor
            print "m: %s" % m
            print "sd: %s" % sd
            print "ann_sd: %s" % ann_sd
    else:
        ann_sd = 0.0

    next_eps = cur_yr_eps

    if debug_mode:
        print "next_eps: %s" % next_eps

    next_eps_sd = next_eps * ann_sd_sym_rtn
    # next_eps_sd = ann_sd
    irr_mean_ci_tuple = cal_irr_mean_ci_MonteCarlo(hist_unadj_px_dict[dt][symbol],
                                                   next_eps,
                                                   next_eps_sd,
                                                   bps,
                                                   # 100,
                                                   10,
                                                   1000,
                                                   95)
    if irr_mean_ci_tuple is not None:
        irr_mean_ci_tuple = map(lambda x: adj_irr_by_cf_time(x, time_to_next_cf), irr_mean_ci_tuple)

    return irr_mean_ci_tuple

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

