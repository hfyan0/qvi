#!/bin/bash
source common.sh
for SYMBOL in $SYMBOL_LIST
do
  OUTFILE="$SYMBOL.html"

  echo $OUTFILE
  # cat $OUTFILE | grep -A 1 "Price / Earn"
  cat $OUTFILE | grep -A 1 "Expense Ratio"
done
