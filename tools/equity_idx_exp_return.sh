#!/bin/bash
for SYM in AEX AS51 CAC DAX HSCEI HSI INDU NDX NKY SMI SPX UKX
do
    python equity_idx_exp_return.py $SYM > ../expected_returns/exp_rtn_"$SYM".csv
done
