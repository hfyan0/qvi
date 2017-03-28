#!/bin/bash

if [[ $# -eq 0 ]]
then
    echo "Usage: $(basename $0) [name of output file]"
    exit
fi

HOME_FOLDER="$HOME/Dropbox/nirvana/mvo/"
BACKTEST_FOLDER="$HOME/Dropbox/nirvana/mvo/backtest/"
BACKTEST_OUTPUT=$BACKTEST_FOLDER/$1

cd $BACKTEST_FOLDER
cat config_actual.ini > config.ini
# python backtest.py | grep -v Terminated > $BACKTEST_OUTPUT
python backtest.py > $BACKTEST_OUTPUT
