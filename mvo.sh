#!/bin/bash

if [[ $1 == 'h' ]]
then
    echo "Usage: $(basename $0) [p] [HK|US]"
    exit
fi

if [[ $1 == 'p' ]]
then
    cd tools
    if [[ -n $2 ]]
    then
        ./download_google_px.sh $2
    else
        ./download_google_px.sh
    fi
    ./add_current_px_manually.sh
    cd ..

    cat config_actual.ini > config.ini
    cat current_positions_actual.csv > current_positions.csv

elif [[ $1 -eq 2007 ]]
then
    cat current_prices_20071101.csv > current_prices.csv
    cat config_20071101.ini > config.ini

elif [[ $1 -eq 2008 ]]
then
    cat current_prices_20081027.csv > current_prices.csv
    cat config_20081027.ini > config.ini
else
    cat config_actual.ini > config.ini
fi

python calc_expected_return.py

python mvo.py | grep -v Terminated 2>&1
