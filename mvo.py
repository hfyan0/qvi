#!/usr/bin/env python
import sys
import math
import numpy as np
import cvxopt
from cvxopt import blas, solvers
from datetime import datetime, timedelta
import scipy.optimize
from itertools import groupby

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
    for d,it_lstup in groupby(sorted(read_file(filename), key=lambda x: x[0]), lambda x: x[0]):
        rtn_dict[datetime.strptime(d,"%Y-%m-%d").date()] = dict(map(lambda y: (y[1],float(y[2])), filter(lambda x: abs(float(x[2])) > 0.0001, list(it_lstup))))
    return rtn_dict

def get_hist_data_key_sym(filename):
    rtn_dict = {}
    for s,it_lstup in groupby(sorted(read_file(filename), key=lambda x: x[1]), lambda x: x[1]):
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
    if len(date_list) < 30:
        return 9999.9
    else:
        min_diff = min(map(lambda x: (x[0]-x[1]).days, zip(date_list[1:],date_list[:-1])))
        if min_diff <= 4:
            return 252
        elif min_diff >= 5 and min_diff <= 7:
            return 52
        elif min_diff >= 20 and min_diff <= 31:
            return 12
        else:
            return 252

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

def calc_expected_return(config,dt,symbol_list,hist_bps_dict,hist_unadj_px_dict,hist_operincm_dict,hist_totasset_dict,hist_totliabps_dict,hist_costofdebt_dict,hist_stattaxrate_dict,hist_oper_eps_dict,delay_months=0):

    curcy_converter = CurrencyConverter(config["currency_rate"])
    ###################################################
    # book-to-price
    ###################################################
    bp_list = []
    conser_bp_ratio_list = []
    conser_bp_ratio_dict = {}
    for sym in symbol_list:
        curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["reporting_currency"][sym],dt)
        # print sym,dt,curcy_conv_rate

        bps_list = map(lambda z: z[1], filter(lambda y: y[0] > shift_back_n_months(dt,5*12+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_bps_dict[sym])))
        ###################################################
        if len(bps_list) >= 3:
            bps_r = calc_return_list(bps_list)
            m = sum(bps_r)/len(bps_r)
            sd = np.std(np.asarray(bps_r))
            conser_bp_ratio = max(curcy_conv_rate * bps_list[-1] * (1 + m - float(config["general"]["bp_stdev"]) * sd) / hist_unadj_px_dict[dt][sym], 0.0)
            conser_bp_ratio_list.append(conser_bp_ratio)
            conser_bp_ratio_dict[sym] = conser_bp_ratio
        else:
            conser_bp_ratio_list.append(0.0)
            conser_bp_ratio_dict[sym] = 0.0
        ###################################################
        if len(bps_list) > 0:
            bp_list.append(bps_list[-1] / hist_unadj_px_dict[dt][sym])
        else:
            bp_list.append(0.0)
        ###################################################

    expected_rtn_list = []
    expected_rtn_asset_driver_list = []
    expected_rtn_external_driver_list = []
    expected_rtn_bv_list = []
    for sym in symbol_list:
        curcy_conv_rate = curcy_converter.get_conv_rate_to_hkd(config["reporting_currency"][sym],dt)
        # print sym,dt,curcy_conv_rate
        ###################################################
        w_a,w_e = map(float, config["expected_rtn_ast_ext"][sym])
        # print "sym: %s %s %s" % (sym, w_a, w_e)

        ###################################################
        # asset driver
        ###################################################
        conser_oper_roa = None

        oper_roa_list = []
        oper_incm_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*4+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_operincm_dict.get(sym,[])))
        for oi_dt,oper_incm in oper_incm_list:
            totasset_list = filter(lambda x: x[0] <= shift_back_n_months(oi_dt,6), hist_totasset_dict.get(sym,[]))
            if len(totasset_list) > 0:
                oper_roa_list.append((oi_dt,oper_incm/totasset_list[-1][1]))

        oper_roa_list = map(lambda x: x[1], oper_roa_list)

        ###################################################
        oper_date_list = map(lambda x: x[0], oper_incm_list)
        if len(oper_date_list) >= 3:
            annualization_factor = round(365.0/min(map(lambda x: (x[0]-x[1]).days, zip(oper_date_list[1:],oper_date_list[:-1]))),0)
        else:
            annualization_factor = 0.0
        ###################################################

        if len(oper_roa_list) >= 5:
            oper_roa_list = sorted(oper_roa_list)[1:][:-1]
            m = sum(oper_roa_list)/len(oper_roa_list)
            sd = np.std(np.asarray(oper_roa_list))
            conser_oper_roa = annualization_factor * (m - float(config["general"]["roa_drvr_stdev"]) * sd)
            # print "sym: %s" % sym
            # print "oper_roa_list: %s" % oper_roa_list
            # print "annualization_factor: %s" % annualization_factor
            # print "m: %s" % m
            # print "sd: %s" % sd
            # print "conser_oper_roa: %s" % conser_oper_roa
        else:
            conser_oper_roa = 0.0

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

        expected_rtn_asset_driver_list.append(w_a*(1-taxrate)*(conser_oper_roa*(totliabps+bps)-iL)*curcy_conv_rate/hist_unadj_px_dict[dt][sym])

        ###################################################
        # external driver
        ###################################################
        oper_eps_list = filter(lambda y: y[0] > shift_back_n_months(dt,12*4+delay_months), filter(lambda x: x[0] <= shift_back_n_months(dt,delay_months), hist_oper_eps_dict.get(sym,[])))
        oper_eps_chg_list = map(lambda x: x[0][1]-x[1][1], zip(oper_eps_list[1:],oper_eps_list[:-1]))
        if len(oper_eps_chg_list) >= 5:
            oper_eps_list = map(lambda z: z[1][1], sorted(sorted(enumerate(oper_eps_list), key=lambda x: x[1][1])[1:][:-1], key=lambda y: y[0]))
            oper_eps_chg_list = sorted(oper_eps_chg_list)[1:][:-1]
            m = sum(oper_eps_chg_list)/len(oper_eps_chg_list)
            sd = np.std(np.asarray(oper_eps_chg_list))
            conser_oper_eps = (oper_eps_list[-1] + m - float(config["general"]["ext_drvr_stdev"]) * sd) * annualization_factor
            # print "sym: %s" % sym
            # print "oper_eps_list: %s" % oper_eps_list
            # print "oper_eps_chg_list: %s" % oper_eps_chg_list
            # print "annualization_factor: %s" % annualization_factor
            # print "m: %s" % m
            # print "sd: %s" % sd
        else:
            conser_oper_eps = 0.0

        expected_rtn_external_driver_list.append(w_e*(1-taxrate)*(conser_oper_eps-iL)*curcy_conv_rate/hist_unadj_px_dict[dt][sym])

        ###################################################
        # liquidation to realize book value
        ###################################################
        expected_rtn_bv_list.append(math.pow(conser_bp_ratio_dict[sym],1.0/float(config["general"]["default_liquidation_years"]))-1)

    expected_rtn_list = map(lambda x: max(x[0]+x[1],x[2]), zip(expected_rtn_asset_driver_list,expected_rtn_external_driver_list,expected_rtn_bv_list))

    return expected_rtn_list
