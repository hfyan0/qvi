#!/bin/bash

HOME_FOLDER="$HOME/Dropbox/nirvana/mvo/"
BACKTEST_FOLDER="$HOME/Dropbox/nirvana/mvo/backtest/"
BACKTEST_OUTPUT=$BACKTEST_FOLDER/out

cd $BACKTEST_FOLDER
cat config_actual.ini > config.ini
python backtest.py | grep -v Terminated > $BACKTEST_OUTPUT
