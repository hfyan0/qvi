#!/bin/bash

if [[ $# -eq 0 ]]
then
    echo "Usage: $(basename $0) [stock code without leading zero]"
    exit
fi
cd tools/
./download_etnet_fundl.sh $1
