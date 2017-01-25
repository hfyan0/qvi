#!/bin/bash

for i in $(lg _us.csv | sed -e 's/_us.csv//' ); do cat "$i"_*.csv > $i".csv"; done

cat hist_adj_px_hk_fut.csv >> hist_adj_px.csv
