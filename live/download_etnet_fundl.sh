#!/bin/bash

HOME_FOLDER="$HOME/Dropbox/nirvana/mvo/"
LIVE_FOLDER="$HOME/Dropbox/nirvana/mvo/live/"

if [[ $# -eq 0 ]]
then
    echo "Usage: $(basename $0) [stock code without leading zero]"
    exit
fi
cd $HOME_FOLDER/tools/
./download_etnet_fundl.sh $1
