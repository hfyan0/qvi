#!/bin/bash

if [[ $1 == 'p' ]]
then
    cd tools
    ./download_google_px.sh
    cd ..
fi

./calc_expected_return.py > expected_returns.csv
# cat expected_returns.csv

python mvo.py | grep -v Terminated 2>&1
