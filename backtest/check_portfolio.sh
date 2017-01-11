#!/bin/bash
if [[ $# -eq 0 ]]
then
    echo "Usage: $(basename $0) [backtest output file] [YYYY-MM-DD]"
    exit
fi

cat $1 | grep "^$2" | ctl | csel : 3 1 | tail -n +4 | sort -n | tac | grep -v "^0.0:" | sed -e 's/:/\t/' 
