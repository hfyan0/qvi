#!/bin/bash
DATA_PATH=/home/qy/Dropbox/dataENF/blmg/data_adj/
ETF_INFO_FOLDER=etf_info/
FUNDL_INFO_FOLDER=fundl_info/
HK_SYMBOL_LIST=$(cat ../live/config.ini | egrep 'traded_symbols_hk|reserved_symbols_hk' | tr ',' ' ' | tr -d \" | awk -F= '{print $2}')
US_SYMBOL_LIST=$(cat ../live/config.ini | egrep 'traded_symbols_us|reserved_symbols_us' | tr ',' ' ' | tr -d \" | awk -F= '{print $2}')
SYMBOL_LIST="US_SYMBOL_LIST"
