#!/bin/bash
source common.sh
for SYMBOL in $SYMBOL_LIST
do
  OUTFILE="$SYMBOL.html"

  wget -O $OUTFILE http://www.etf.com/$SYMBOL
done
