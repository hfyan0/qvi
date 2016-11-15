#!/bin/bash
source common.sh
for SYMBOL in $SYMBOL_LIST
do
  OUTFILE="$ETF_INFO_FOLDER/$SYMBOL.html"

  wget -O $OUTFILE http://www.etf.com/$SYMBOL
done
