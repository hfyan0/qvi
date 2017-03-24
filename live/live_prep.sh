#!/bin/bash

HOME_FOLDER="$HOME/Dropbox/nirvana/mvo/"
LIVE_FOLDER="$HOME/Dropbox/nirvana/mvo/live/"

cat config_sunny.ini > config.ini
cd $LIVE_FOLDER
python live_prep.py | grep -v Terminated 2>&1
