#!/bin/bash

if [[ $1 == 'h' ]]
then
    echo "Usage: $(basename $0) [p] [HK|US]"
    exit
fi

HOME_FOLDER="$HOME/Dropbox/nirvana/mvo/"
LIVE_FOLDER="$HOME/Dropbox/nirvana/mvo/live/"

cd $LIVE_FOLDER
if [[ $1 == 'p' ]]
then
    cat config_actual.ini > config.ini
    cat current_positions_sunny.csv > current_positions.csv

    cd $HOME_FOLDER/tools
    if [[ -n $2 ]]
    then
        ./download_google_px.sh $2
    else
        ./download_google_px.sh
    fi
    ./add_current_px_manually.sh

elif [[ $1 -eq 2007 ]]
then
    cat current_prices_20071101.csv > current_prices.csv
    cat config_20071101.ini > config.ini

elif [[ $1 -eq 2008 ]]
then
    cat current_prices_20081027.csv > current_prices.csv
    cat config_20081027.ini > config.ini
else
    cat current_prices_*.csv > current_prices.csv
    cat current_positions_sunny.csv > current_positions.csv
    cat config_actual.ini > config.ini
    cd $HOME_FOLDER/tools
    ./add_current_px_manually.sh
fi

cd $LIVE_FOLDER
python calc_expected_return.py
python live.py | grep -v Terminated 2>&1
