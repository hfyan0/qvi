#!/bin/bash
source common.sh
for SYMBOL in $SYMBOL_LIST
do
    echo $SYMBOL
    cat $DATA_PATH/$SYMBOL".csv" | tail -n 2 | head -n 1 | awk -F, '{print $6}'
done
