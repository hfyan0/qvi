#!/bin/bash
source common.sh

DOMAIN="www.google.com"
TMPFILE="$ETF_INFO_FOLDER/tmpfile"
EXCHGLIST_ALL="HK US"
PX_FILE_PREFIX="../live/current_prices_"
PX_FILE="../live/current_prices.csv"

if [[ $# -gt 0 ]]
then
    EXCHGLIST="$@"
else
    EXCHGLIST=$EXCHGLIST_ALL
fi

for e in $EXCHGLIST
do
    cat /dev/null > "$PX_FILE_PREFIX"$e".csv"
done

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
        CURPX=$(cat $TMPFILE | grep ref_ | head -n 1 | sed -e 's/<\/.*$//' | sed -e 's/^.*>//')
        if [[ -z $CURPX ]]
        then
            CURPX="0.00"
        fi
        echo "$SYMBOL,$CURPX" >> "$PX_FILE_PREFIX"$e".csv"
    done
done

cat $PX_FILE_PREFIX* > $PX_FILE

rm -f $TMPFILE
