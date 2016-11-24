#!/bin/bash

if [[ $1 == 'p' ]]
then
    cd tools
    ./download_google_px.sh
    ./add_current_px_manually.sh
    cd ..
fi

python calc_expected_return.py

python mvo.py | grep -v Terminated 2>&1
