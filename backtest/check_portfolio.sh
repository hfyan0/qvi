#!/bin/bash
if [[ $# -lt 2 ]]
then
    echo "Usage: $(basename $0) [backtest output file] [YYYY-MM-DD]"
    exit
fi

cat $1 | grep "^$2" | ctl | grep -v pos | tail -n +8 | csel : 3 1 | sort -n | tac | grep -v "^0.0:" | sed -e 's/:/\t/' 
