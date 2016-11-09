#!/bin/bash

HK_SYMBOL_LIST="0011 2388 0052 0341 0808 0823 2800"
US_SYMBOL_LIST=""
# US_SYMBOL_LIST="VYM EFA QQQ EWA EWG SPY VOO VNQ JNK EMB TLT MBB EZA IVV"

CUR_PRICE_FILE="current_prices"
TMPFILE="tmpfile"
cat /dev/null > $CUR_PRICE_FILE

for i in "HK" "US"
do
    if [[ $i == "HK" ]]
    then
        SYMBOL_LIST="$HK_SYMBOL_LIST"
        EXCHANGE="HKG"
    elif [[ $i == "US" ]]
    then
        SYMBOL_LIST="$US_SYMBOL_LIST"
        EXCHANGE="NYSEARCA"
    fi

    for SYMBOL in $SYMBOL_LIST
    do
        wget -O $TMPFILE "https://www.google.com/finance?q="$EXCHANGE"%3A"$SYMBOL"&hl=en"
        echo -n "$SYMBOL," >> $CUR_PRICE_FILE
        cat $TMPFILE | grep ref_ | head -n 1 | sed -e 's/<\/.*$//' | sed -e 's/^.*>//' >> $CUR_PRICE_FILE
    done
done

rm -f $TMPFILE
