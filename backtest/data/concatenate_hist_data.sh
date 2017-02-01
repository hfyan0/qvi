#!/bin/bash

for i in $(lg _us.csv | sed -e 's/_us.csv//' ); do cat "$i"_*.csv > $i".csv"; done
