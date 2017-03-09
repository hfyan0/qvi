#!/usr/bin/env python

from mvo import cal_irr_mean_var_Reinschmidt,cal_irr_mean_ci_MonteCarlo,adj_irr_by_cf_time

# corr_matrix = [
# [1, 0, 0, 0, 0, 0],
# [0, 1, 0.92, 0.92, 0.92, 0.89],
# [0, 0.92, 1, 0.99, 0.99, 0.95],
# [0, 0.92, 0.99, 1, 0.99, 0.95],
# [0, 0.92, 0.99, 0.99, 1, 0.95],
# [0, 0.89, 0.95, 0.95, 0.95, 1]]
#
# print cal_irr_mean_var_Reinschmidt([-600000,50000,400000,300000,200000,200000],[50000,53852,100499,100499,100499,104881],corr_matrix,None)


print cal_irr_mean_ci_MonteCarlo(23.4,1.22,1.22*0.2,100,1000,95)

# print adj_irr_by_cf_time(-0.03,0.5)
