#!/bin/bash
source common.sh

DOMAIN="www.google.com.hk"
TMPFILE="$ETF_INFO_FOLDER/tmpfile"
EXCHGLIST_ALL="HK US"

if [[ $# -gt 0 ]]
then
    EXCHGLIST="$@"
    for e in $EXCHGLIST
    do
        cat /dev/null > "../current_prices_"$e".csv"
    done
else
    EXCHGLIST=$EXCHGLIST_ALL
fi

for e in $EXCHGLIST
do
    if [[ $e == "HK" ]]
    then
        SYMBOL_LIST="$HK_SYMBOL_LIST"
        EXCHANGE="HKG"
    elif [[ $e == "US" ]]
    then
        SYMBOL_LIST="$US_SYMBOL_LIST"
        EXCHANGE="NYSEARCA"
    fi

    for SYMBOL in $SYMBOL_LIST
    do
        wget -O $TMPFILE "https://$DOMAIN/finance?q="$EXCHANGE"%3A"$SYMBOL"&hl=en"
        echo -n "$SYMBOL," >> "../current_prices_"$e".csv"
        cat $TMPFILE | grep ref_ | head -n 1 | sed -e 's/<\/.*$//' | sed -e 's/^.*>//' >> "../current_prices_"$e".csv"
    done
done

cat ../current_prices_* > ../current_prices.csv

rm -f $TMPFILE
