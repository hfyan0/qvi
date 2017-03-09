#!/bin/bash

HOME_FOLDER="$HOME/Dropbox/nirvana/mvo/"
BACKTEST_FOLDER="$HOME/Dropbox/nirvana/mvo/backtest/"

cd $BACKTEST_FOLDER
cat config_test_ci.ini > config.ini
python test_ci.py
