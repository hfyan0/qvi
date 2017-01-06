#!/bin/bash

if [[ $1 == 'h' ]]
then
    echo "Usage: $(basename $0) [p] [HK|US]"
    exit
fi

HOME_FOLDER="$HOME/Dropbox/nirvana/mvo/"
LIVE_FOLDER="$HOME/Dropbox/nirvana/mvo/live/"

if [[ $1 == 'p' ]]
then
    cd $LIVE_FOLDER
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
else
    cd $LIVE_FOLDER
    cat current_prices_*.csv > current_prices.csv
    cat current_positions_sunny.csv > current_positions.csv
    cat config_actual.ini > config.ini
    cd $HOME_FOLDER/tools
    ./add_current_px_manually.sh
fi

cd $LIVE_FOLDER
python calc_expected_return.py
python live.py | grep -v Terminated 2>&1
