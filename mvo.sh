#!/bin/bash

# cd tools
# ./download_google_px.sh
# cd ..

./calc_expected_return.py > expected_returns.csv

python mvo.py | grep -v Terminated 2>&1
